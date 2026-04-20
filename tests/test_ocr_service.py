import json
from io import BytesIO

import fitz
from PIL import Image

import pytest

from app.core.configs import settings
from app.schemas import (
    DocumentClassificationError,
    Receipt,
    create_extraction_output_model,
)
from app.services.ocr import OCRService


class FakeOpenAIResponse:
    def __init__(self, parsed) -> None:
        self.output_parsed = parsed


class FakeResponses:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def parse(self, *, model, input, text_format):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "text_format": text_format,
            }
        )
        return FakeOpenAIResponse(text_format.model_validate(self.payloads.pop(0)))


class FakeOpenAIClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.responses = FakeResponses(payloads)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (80, 40), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 50), "receipt")
    return document.tobytes()


def _service_with_fake_openai(fake_client: FakeOpenAIClient) -> OCRService:
    return OCRService(openai_client=fake_client)


# Covers the happy path: classification call, extraction call, image payloads,
# structured output models, and confidence-threshold post-processing.
def test_service_uses_two_stage_openai_prompts_and_structured_outputs() -> None:
    fake_client = FakeOpenAIClient(
        payloads=[
            {"document_type": "receipt", "confidence": 0.96},
            {
                "fields": {
                    "claimant_name": {"value": "Jane Tan", "confidence": 0.91},
                    "provider_name": {"value": "unsure", "confidence": 0.91},
                    "tax_amount": {"value": 80, "confidence": 0.49},
                    "total_amount": {"value": 1200, "confidence": 0.95},
                }
            },
        ]
    )
    service = _service_with_fake_openai(fake_client)

    response = service.extract_text(
        file_bytes=_png_bytes(),
        filename="receipt.png",
        content_type="image/png",
    )

    calls = fake_client.responses.calls
    assert len(calls) == 2
    assert calls[0]["model"] == settings.openai_model
    assert calls[0]["text_format"].__name__ == "ClassificationOutput"

    classification_content = calls[0]["input"][0]["content"]
    classification_prompt = classification_content[0]["text"]
    assert classification_content[0]["type"] == "input_text"
    assert classification_content[1]["type"] == "input_image"
    assert classification_content[1]["image_url"].startswith("data:image/png;base64,")
    assert "referral_letter" in classification_prompt
    assert "medical_certificate" in classification_prompt
    assert "receipt" in classification_prompt
    assert "Clinical referral letter" in classification_prompt
    assert "Medical certificate (MC)" in classification_prompt
    assert "Healthcare billing receipt" in classification_prompt
    assert '"referral_letter|medical_certificate|receipt"' not in classification_prompt
    assert "Choose exactly one document_type value" in classification_prompt

    extraction_content = calls[1]["input"][0]["content"]
    extraction_prompt = extraction_content[0]["text"]
    assert "document_type=receipt" in extraction_prompt
    assert "claimant_name" in extraction_prompt
    assert "tax_amount" in extraction_prompt
    assert calls[1]["text_format"].__name__ == "ReceiptExtractionOutput"

    assert response.result.document_type == "receipt"
    assert response.result.cleaning_elapsed >= 0
    assert response.result.classification_elapsed >= 0
    assert response.result.extraction_elapsed >= 0
    assert response.result.final_json.claimant_name == "Jane Tan"
    assert response.result.final_json.provider_name is None
    assert response.result.final_json.tax_amount is None
    assert response.result.final_json.total_amount == 1200


# Covers the dynamic extraction wrapper schema used for OpenAI structured outputs.
def test_extraction_output_model_is_derived_from_pydantic_model() -> None:
    schema = create_extraction_output_model(Receipt).model_json_schema()

    assert "claimant_name" in json.dumps(schema)
    assert "tax_amount" in json.dumps(schema)


# Covers PDF rendering into page images before sending them to OpenAI vision.
def test_service_converts_pdf_pages_to_images() -> None:
    fake_client = FakeOpenAIClient(
        payloads=[
            {"document_type": "medical_certificate", "confidence": 0.92},
            {
                "fields": {
                    "claimant_name": {"value": "Jane Tan", "confidence": 0.9},
                    "date_of_mc": {"value": "08/04/2026", "confidence": 0.9},
                }
            },
        ]
    )
    service = _service_with_fake_openai(fake_client)

    response = service.extract_text(
        file_bytes=_pdf_bytes(),
        filename="medical_certificate.pdf",
        content_type="application/pdf",
    )

    classification_content = fake_client.responses.calls[0]["input"][0]["content"]
    assert len([item for item in classification_content if item["type"] == "input_image"]) == 1
    assert response.result.document_type == "medical_certificate"
    assert response.result.final_json.claimant_name == "Jane Tan"
    assert response.result.final_json.date_of_mc == "08/04/2026"


# Covers rejecting classifications below the configured confidence threshold.
def test_service_rejects_low_confidence_classification() -> None:
    fake_client = FakeOpenAIClient(
        payloads=[{"document_type": "receipt", "confidence": 0.49}]
    )
    service = _service_with_fake_openai(fake_client)

    with pytest.raises(DocumentClassificationError):
        service.extract_text(
            file_bytes=_png_bytes(),
            filename="receipt.png",
            content_type="image/png",
        )

    assert len(fake_client.responses.calls) == 1
