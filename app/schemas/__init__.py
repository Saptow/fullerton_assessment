from app.schemas.error import DocumentClassificationError, ErrorResponse
from app.schemas.health import HealthResponse
from app.schemas.ocr import (
    ClassificationOutput,
    DOCUMENT_SCHEMAS,
    DOCUMENT_TYPE_DESCRIPTION,
    ExtractedFieldValue,
    MedicalCertificate,
    OCRResponse,
    OCRResult,
    Receipt,
    ReferralLetter,
    create_extraction_output_model,
)

__all__ = [
    "ErrorResponse",
    "DocumentClassificationError",
    "HealthResponse",
    "DOCUMENT_SCHEMAS",
    "DOCUMENT_TYPE_DESCRIPTION",
    "ClassificationOutput",
    "ExtractedFieldValue",
    "MedicalCertificate",
    "OCRResponse",
    "OCRResult",
    "Receipt",
    "ReferralLetter",
    "create_extraction_output_model",
]
