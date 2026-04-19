from pathlib import Path

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.constants import (
    OCR_EXTRACTION_RESPONSES,
    SUPPORTED_CONTENT_TYPES,
    SUPPORTED_EXTENSIONS,
)
from app.schemas import DocumentClassificationError, OCRResponse
from app.services.ocr import OCRService

router = APIRouter()

# Main endpoint for OCR extraction with comprehensive validation and error handling
@router.post(
    "/ocr",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    responses=OCR_EXTRACTION_RESPONSES,
    tags=["ocr"],
)
async def create_ocr_extraction(
    file: UploadFile | None = File(
        default=None,
        description="PDF, JPG, or PNG document to process.",
    ),
) -> OCRResponse | JSONResponse:
    """
    Main endpoint for OCR extraction with comprehensive validation and error handling.
    - Validates against the current set of supported document types, check README for details for supported types.
    - Returns structured error responses for various failure scenarios (missing file, invalid MIME type, unsupported document type, and internal processing errors).
    """
    
    # Basic validation of the uploaded file
    if file is None or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "file_missing"},
        )

    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "invalid_mime_type"},
        )

    if Path(file.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "unsupported_document_type"},
        )

    # Read the file contents into memory (consider message queues for large files in production using Redis or similar)
    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "file_missing"},
        )

    # Main OCR processing logic with error handling
    try:
        ocr_service = OCRService()
        return ocr_service.extract_text(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_file",
            content_type=file.content_type,
        )
    except DocumentClassificationError:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "unsupported_document_type"},
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_server_error"},
        )
