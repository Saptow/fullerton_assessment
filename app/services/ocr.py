import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel

from app.constants import (
    CLASSIFICATION_PROMPT_TEMPLATE,
    EXTRACTION_PROMPT_TEMPLATE,
    PDF_CONTENT_TYPE,
)
from app.core.configs import settings
from app.schemas import (
    ClassificationOutput,
    DOCUMENT_SCHEMAS,
    DOCUMENT_TYPE_DESCRIPTION,
    DocumentClassificationError,
    OCRResponse,
    OCRResult,
    create_extraction_output_model,
)


logger = logging.getLogger(__name__)


class OCRService:
    """OCR pipeline using PyMuPDF rendering and OpenAI structured outputs."""

    pdf_render_scale = 4  # Balances PDF render quality against processing time.
    min_image_width = 1200  # Upscales small images enough for readable OCR.
    binarization_threshold = 150  # Splits foreground text from background noise.
    confidence_threshold = 0.5  # Below this, extracted fields become null.
    classification_confidence_threshold = 0.9  # Below this, the type is unsupported.
    null_extracted_values = {"", "unsure", "unknown", "n/a", "na", "none", "null"}

    def __init__(
        self,
        *,
        openai_client: Any | None = None,
    ) -> None:
        self._openai_client = openai_client

    # Helper for interacting with OpenAI's structured output responses, handling both text and image inputs.
    def generate_with_model(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        output_model: type[BaseModel],
    ) -> dict[str, Any]:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI()

        logger.debug(
            "Calling OpenAI structured output model=%s output_model=%s image_count=%s",
            settings.openai_model,
            output_model.__name__,
            len(images),
        )

        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                    "detail": settings.openai_image_detail,
                }
            )

        response = self._openai_client.responses.parse(
            model=settings.openai_model,
            input=[{"role": "user", "content": content}],
            text_format=output_model,
        )
        logger.debug("OpenAI structured output parsed as %s", output_model.__name__)
        return response.output_parsed.model_dump(mode="json")

    # Main method to extract text from the uploaded document.
    def extract_text(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> OCRResponse:
        started_at = time.perf_counter()
        cleaning_started_at = time.perf_counter()
        logger.debug(
            "Starting OCR extraction filename=%s content_type=%s",
            filename,
            content_type,
        )

        # Load the uploaded document into one or more RGB page images.
        images: list[Image.Image] = []
        is_pdf = (
            content_type == PDF_CONTENT_TYPE
            or Path(filename).suffix.lower() == ".pdf"
        )

        if is_pdf:
            with fitz.open(stream=file_bytes, filetype="pdf") as document:
                for page in document:
                    matrix = fitz.Matrix(
                        self.pdf_render_scale,
                        self.pdf_render_scale,
                    )
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    images.append(image)
        else:
            image = Image.open(BytesIO(file_bytes))
            images.append(image.convert("RGB"))

        logger.debug("Loaded %s page image(s)", len(images))

        # Normalize contrast and text edges before sending images to OpenAI vision.
        cleaned_images: list[Image.Image] = []
        for image in images:
            cleaned = image.convert("RGB")
            cleaned = ImageOps.grayscale(cleaned)
            cleaned = ImageOps.autocontrast(cleaned)

            if cleaned.width < self.min_image_width:
                scale = self.min_image_width / cleaned.width
                cleaned = cleaned.resize(
                    (self.min_image_width, int(cleaned.height * scale)),
                    Image.Resampling.LANCZOS,
                )

            cleaned = cleaned.filter(ImageFilter.SHARPEN)
            cleaned = cleaned.point(
                lambda pixel: 255 if pixel > self.binarization_threshold else 0
            )
            cleaned_images.append(cleaned.convert("RGB"))

        cleaning_elapsed = time.perf_counter() - cleaning_started_at
        logger.debug("Image cleaning completed in %.2fs", cleaning_elapsed)

        # First OpenAI pass: classify the document before spending on extraction.
        classification_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            document_type_descriptions=DOCUMENT_TYPE_DESCRIPTION,
        )
        classification_started_at = time.perf_counter()
        classification_payload = self.generate_with_model(
            classification_prompt,
            cleaned_images,
            output_model=ClassificationOutput,
        )
        classification_elapsed = time.perf_counter() - classification_started_at
        document_type = classification_payload.get("document_type")
        classification_confidence = classification_payload.get("confidence", 0)
        logger.debug(
            "Classification completed document_type=%s confidence=%s elapsed=%.2fs",
            document_type,
            classification_confidence,
            classification_elapsed,
        )

        try:
            classification_confidence = float(classification_confidence)
        except (TypeError, ValueError) as exc:
            raise DocumentClassificationError(
                "Document classification confidence was missing or invalid.",
                payload=classification_payload,
            ) from exc

        if (
            document_type not in DOCUMENT_SCHEMAS
            or classification_confidence < self.classification_confidence_threshold
        ):
            raise DocumentClassificationError(
                "Document classification did not meet the confidence threshold.",
                payload=classification_payload,
            )

        # Second OpenAI pass: extract only the fields for the selected schema.
        schema = DOCUMENT_SCHEMAS[document_type]
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            document_type=document_type,
            schema_json=json.dumps(schema.model_json_schema()),
        )
        extraction_output_model = create_extraction_output_model(schema)
        extraction_started_at = time.perf_counter()
        extraction_payload = self.generate_with_model(
            extraction_prompt,
            cleaned_images,
            output_model=extraction_output_model,
        )
        extraction_elapsed = time.perf_counter() - extraction_started_at
        logger.debug(
            "Extraction completed document_type=%s elapsed=%.2fs",
            document_type,
            extraction_elapsed,
        )

        fields = extraction_payload.get("fields", extraction_payload)
        if not isinstance(fields, dict):
            fields = {}

        # Gate low-confidence values before final Pydantic normalization.
        values: dict[str, Any] = {}
        for field_name in schema.model_fields:
            field_payload = fields.get(field_name)

            if field_payload is None:
                value = None
            elif not isinstance(field_payload, dict):
                value = field_payload
            elif field_payload.get("confidence", 0) < self.confidence_threshold:
                value = None
            else:
                value = field_payload.get("value")

            if (
                isinstance(value, str)
                and value.strip().lower() in self.null_extracted_values
            ):
                logger.debug("Normalizing placeholder value to null field=%s", field_name)
                value = None

            values[field_name] = value

        return OCRResponse(
            result=OCRResult(
                document_type=document_type,
                total_time=round(time.perf_counter() - started_at, 2),
                cleaning_elapsed=round(cleaning_elapsed, 2),
                classification_elapsed=round(classification_elapsed, 2),
                extraction_elapsed=round(extraction_elapsed, 2),
                final_json=schema(**values),
            )
        )
