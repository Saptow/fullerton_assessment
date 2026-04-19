from app.schemas import ErrorResponse

# Upload validation constants used by the FastAPI route and OCR service.
PDF_CONTENT_TYPE = "application/pdf"
JPEG_CONTENT_TYPE = "image/jpeg"
PNG_CONTENT_TYPE = "image/png"

SUPPORTED_CONTENT_TYPES = {
    PDF_CONTENT_TYPE,
    JPEG_CONTENT_TYPE,
    PNG_CONTENT_TYPE,
}
SUPPORTED_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
}

# Prompt templates for the two-step model workflow.
CLASSIFICATION_PROMPT_TEMPLATE = """
# Role
You are a strict JSON document classifier for healthcare OCR.

# Task
Read the uploaded document image and classify it as exactly one document_type, the one you have the highest confidence in and best matches content.

# Allowed document_type Values
- referral_letter
- medical_certificate
- receipt

# Document Type Descriptions
{document_type_descriptions}

# Few-shot Examples
Example input clue: A healthcare tax invoice, receipt number, GST/tax, payment amount, or total paid.
Example output: {{"document_type": "receipt", "confidence": 0.95}}

Example input clue: A medical certificate, unfit for work/school/duty, MC days, diagnosis, or leave dates.
Example output: {{"document_type": "medical_certificate", "confidence": 0.95}}

Example input clue: A referral note or letter requesting specialist review, consultation, investigation, or follow-up.
Example output: {{"document_type": "referral_letter", "confidence": 0.95}}

# Rules
- Return JSON only.
- Do not return prose, markdown, explanation, or extracted document text.
- Do not describe signatures, dates, patient details, or document content.
- The document_type value must be one of the allowed values.
- Choose exactly one document_type value. Do not combine values.
- Do not return a combined or pipe-separated list as the document_type value.
- The confidence value must be a number between 0 and 1.
- Return exactly one object with document_type and confidence.

# Required JSON Keys
- document_type
- confidence
"""

EXTRACTION_PROMPT_TEMPLATE = """
# Role
You are a strict JSON field extractor for healthcare OCR.

# Task
Read the uploaded document image and extract structured fields.

# Document Type
document_type={document_type}

# Schema
{schema_json}

# Few-shot Examples
These examples show the output style only. Do not copy example values.

Receipt-style output:
{{"fields": {{"provider_name": {{"value": "Example Clinic", "confidence": 0.95}}, "tax_amount": {{"value": 80, "confidence": 0.9}}, "total_amount": {{"value": 1200, "confidence": 0.95}}}}}}

Medical-certificate-style output:
{{"fields": {{"claimant_name": {{"value": "Example Patient", "confidence": 0.92}}, "date_of_mc": {{"value": "08/04/2026", "confidence": 0.95}}, "mc_days": {{"value": 2, "confidence": 0.9}}}}}}

Referral-letter-style output:
{{"fields": {{"claimant_name": {{"value": "Example Patient", "confidence": 0.92}}, "provider_name": {{"value": "Example Specialist Clinic", "confidence": 0.9}}, "signature_presence": {{"value": true, "confidence": 0.85}}}}}}

# Rules
- Return JSON only.
- Do not return prose, markdown, explanation, or copied document text outside JSON.
- Use only fields from the schema.
- For every field, return an object with value, and confidence between 0 and 1.
- Use high confidence when the value is clearly visible in the document.
- Use low confidence only when the value is missing, unreadable, or ambiguous.
"""

# Swagger UI response metadata for /ocr.
OCR_EXTRACTION_RESPONSES = {
    200: {"description": "OCR extraction completed successfully."},
    400: {
        "description": "No file or invalid MIME type.",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "file_missing": {"value": {"error": "file_missing"}},
                    "invalid_mime_type": {"value": {"error": "invalid_mime_type"}},
                }
            }
        },
    },
    422: {
        "description": "Unsupported document type.",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {"error": "unsupported_document_type"}
            }
        },
    },
    500: {
        "description": "Unhandled exception.",
        "model": ErrorResponse,
        "content": {
            "application/json": {"example": {"error": "internal_server_error"}}
        },
    },
}
