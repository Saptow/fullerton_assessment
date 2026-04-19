import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


# Shared value types for final OCR responses.
UnsureValue = Literal["unsure"]
TextValue = str | UnsureValue | None
IntegerValue = int | UnsureValue | None
BooleanValue = bool | UnsureValue

# OpenAI extraction can emit decimals before final schema normalization.
ExtractedFieldPrimitive = str | int | float | bool | None


DATE_INPUT_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
)
DATE_OUTPUT_FORMAT = "%d/%m/%Y"
AMOUNT_DESCRIPTION = (
    "Currency symbols and separators removed; decimals rounded to nearest integer."
)


# Schema validators normalize model output before API serialization.
def _validate_provider_name(value: TextValue) -> TextValue:
    # Fullerton Health is the submitter in this assessment, not the provider.
    if isinstance(value, str) and value != "unsure" and "Fullerton Health" in value:
        return None
    return value


def _validate_date(value: Any) -> TextValue:
    if value is None or value == "unsure":
        return value

    if isinstance(value, datetime):
        return value.strftime(DATE_OUTPUT_FORMAT)

    if isinstance(value, date):
        return value.strftime(DATE_OUTPUT_FORMAT)

    if not isinstance(value, str):
        raise ValueError("date must be in DD/MM/YYYY format")

    text = value.strip()
    if not text:
        return None

    for format_ in DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(text, format_).strftime(DATE_OUTPUT_FORMAT)
        except ValueError:
            pass

    match = re.match(r"^(\d{4}-\d{1,2}-\d{1,2})[T\s]", text)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").strftime(
                DATE_OUTPUT_FORMAT
            )
        except ValueError:
            pass

    match = re.match(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{4})[T\s]", text)
    if match is not None:
        separator = "/" if "/" in match.group(1) else "-"
        try:
            return datetime.strptime(
                match.group(1),
                f"%d{separator}%m{separator}%Y",
            ).strftime(DATE_OUTPUT_FORMAT)
        except ValueError:
            pass

    raise ValueError("date must be in DD/MM/YYYY format")


def _validate_amount(value: Any) -> IntegerValue:
    if value is None or value == "unsure":
        return value

    if isinstance(value, bool):
        raise ValueError("amount must be numeric")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        number = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        is_negative = text.startswith("(") and text.endswith(")")
        text = re.sub(r"[^0-9.\-]", "", text)
        if not text or text in {"-", ".", "-."}:
            return None

        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("amount must be numeric") from exc

        if is_negative:
            number = -abs(number)
    else:
        raise ValueError("amount must be numeric")

    return int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


# Document schemas; allow null and "unsure" for missing or gated fields.
class ReferralLetter(BaseModel):
    claimant_name: TextValue = Field(default=None, description="Patient name.")
    provider_name: TextValue = Field(
        default=None,
        description="Provider / lab name.",
    )
    signature_presence: BooleanValue = Field(
        default=False,
        description=(
            "Some indication of doctor signoff or signature, such as a "
            "signature image, handwritten name, or 'signed' text."
        ),
    )
    total_amount_paid: IntegerValue = Field(
        default=None,
        description=AMOUNT_DESCRIPTION,
    )
    total_approved_amount: IntegerValue = Field(
        default=None,
        description=AMOUNT_DESCRIPTION,
    )
    total_requested_amount: IntegerValue = Field(
        default=None,
        description=AMOUNT_DESCRIPTION,
    )

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: TextValue) -> TextValue:
        return _validate_provider_name(value)

    @field_validator("signature_presence", mode="before")
    @classmethod
    def default_missing_signature_presence(
        cls,
        value: BooleanValue | None,
    ) -> BooleanValue:
        if value is None:
            return False
        return value

    @field_validator(
        "total_amount_paid",
        "total_approved_amount",
        "total_requested_amount",
        mode="before",
    )
    @classmethod
    def validate_amounts(cls, value: Any) -> IntegerValue:
        return _validate_amount(value)


class MedicalCertificate(BaseModel):
    claimant_name: TextValue = Field(default=None, description="Claimant name.")
    claimant_address: TextValue = Field(default=None, description="Claimant address.")
    claimant_date_of_birth: TextValue = Field(
        default=None,
        description="Claimant date of birth in DD/MM/YYYY format.",
        examples=["31/12/1990"],
    )
    diagnosis_name: TextValue = Field(default=None, description="Diagnosis.")
    discharge_date_time: TextValue = Field(
        default=None,
        description="Discharge date in DD/MM/YYYY format.",
        examples=["08/04/2026"],
    )
    icd_code: TextValue = Field(default=None, description="ICD code.")
    provider_name: TextValue = Field(
        default=None,
        description="Provider / lab name.",
    )
    submission_date_time: TextValue = Field(
        default=None,
        description="Admission datetime in DD/MM/YYYY format.",
        examples=["08/04/2026"],
    )
    date_of_mc: TextValue = Field(
        default=None,
        description="Date of MC in DD/MM/YYYY format.",
        examples=["08/04/2026"],
    )
    mc_days: IntegerValue = Field(
        default=None,
        description="Integer number of MC days.",
    )

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: TextValue) -> TextValue:
        return _validate_provider_name(value)

    @field_validator(
        "claimant_date_of_birth",
        "discharge_date_time",
        "submission_date_time",
        "date_of_mc",
        mode="before",
    )
    @classmethod
    def validate_dates(cls, value: TextValue) -> TextValue:
        return _validate_date(value)


class Receipt(BaseModel):
    claimant_name: TextValue = Field(default=None, description="Claimant name.")
    claimant_address: TextValue = Field(default=None, description="Claimant address.")
    claimant_date_of_birth: TextValue = Field(
        default=None,
        description="Claimant date of birth in DD/MM/YYYY format.",
        examples=["31/12/1990"],
    )
    provider_name: TextValue = Field(
        default=None,
        description="Provider / lab name.",
    )
    tax_amount: IntegerValue = Field(
        default=None,
        description=AMOUNT_DESCRIPTION,
    )
    total_amount: IntegerValue = Field(
        default=None,
        description=AMOUNT_DESCRIPTION,
    )

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: TextValue) -> TextValue:
        return _validate_provider_name(value)

    @field_validator("claimant_date_of_birth", mode="before")
    @classmethod
    def validate_dates(cls, value: TextValue) -> TextValue:
        return _validate_date(value)

    @field_validator("tax_amount", "total_amount", mode="before")
    @classmethod
    def validate_amounts(cls, value: Any) -> IntegerValue:
        return _validate_amount(value)


# Document type metadata used by classification, extraction, and final responses.
DOCUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "referral_letter": ReferralLetter,
    "medical_certificate": MedicalCertificate,
    "receipt": Receipt,
}

DOCUMENT_TYPE_DESCRIPTION = """
- referral_letter: Clinical referral letter with patient details, referral reason, clinical notes, and doctor signoff/signature.
- medical_certificate: Medical certificate (MC) stating unfitness for work, school, or duty, usually with leave dates, MC days, diagnosis, and provider details.
- receipt: Healthcare billing receipt, tax invoice, or payment document with provider/patient details, line items, tax, totals, and payment or visit date.
""".strip()

DocumentType = Annotated[
    Literal["referral_letter", "medical_certificate", "receipt"],
    Field(description=DOCUMENT_TYPE_DESCRIPTION),
]
ExtractedDocument = ReferralLetter | MedicalCertificate | Receipt




# Structured outputs
class ClassificationOutput(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0, le=1)


class ExtractedFieldValue(BaseModel):
    value: ExtractedFieldPrimitive = None
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Confidence for this extracted value. Use high confidence when the "
            "value is clearly visible; use low confidence only when missing, "
            "unreadable, or ambiguous."
        ),
    )


def create_extraction_output_model(
    document_schema: type[BaseModel],
) -> type[BaseModel]:
    """Create the structured-output schema for a selected document type."""

    fields_model = create_model(
        f"{document_schema.__name__}ExtractedFields",
        **{
            field_name: (
                ExtractedFieldValue | None,
                Field(default=None, description=field.description),
            )
            for field_name, field in document_schema.model_fields.items()
        },
    )
    return create_model(
        f"{document_schema.__name__}ExtractionOutput",
        fields=(fields_model, Field(...)),
    )





# API response schemas.
class OCRResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_type: DocumentType = Field(examples=["referral_letter"])
    total_time: float | None = Field(default=None, ge=0, examples=[3.04])
    cleaning_elapsed: float | None = Field(default=None, ge=0, examples=[0.12])
    classification_elapsed: float | None = Field(default=None, ge=0, examples=[1.2])
    extraction_elapsed: float | None = Field(default=None, ge=0, examples=[1.72])
    final_json: ExtractedDocument = Field(alias="finalJson")


class OCRResponse(BaseModel):
    message: str = Field(default="Processing completed.")
    result: OCRResult
