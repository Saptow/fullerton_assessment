from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.schemas import MedicalCertificate, Receipt, ReferralLetter


def test_medical_certificate_accepts_dd_mm_yyyy_dates() -> None:
    certificate = MedicalCertificate(
        claimant_date_of_birth="31/12/1990",
        discharge_date_time="08/04/2026",
        submission_date_time="08/04/2026",
        date_of_mc="08/04/2026",
    )

    assert certificate.claimant_date_of_birth == "31/12/1990"
    assert certificate.discharge_date_time == "08/04/2026"
    assert certificate.submission_date_time == "08/04/2026"
    assert certificate.date_of_mc == "08/04/2026"


def test_medical_certificate_rejects_invalid_dates() -> None:
    with pytest.raises(ValidationError):
        MedicalCertificate(claimant_date_of_birth="31/02/2026")


def test_medical_certificate_normalizes_equivalent_date_formats() -> None:
    certificate = MedicalCertificate(
        claimant_date_of_birth=date(1990, 12, 31),
        discharge_date_time=datetime(2026, 4, 8, 14, 30),
        submission_date_time="2026-04-08T14:30:00",
        date_of_mc="8-4-2026",
    )

    assert certificate.claimant_date_of_birth == "31/12/1990"
    assert certificate.discharge_date_time == "08/04/2026"
    assert certificate.submission_date_time == "08/04/2026"
    assert certificate.date_of_mc == "08/04/2026"


def test_receipt_accepts_null_date() -> None:
    receipt = Receipt(claimant_date_of_birth=None)

    assert receipt.claimant_date_of_birth is None


def test_receipt_normalizes_iso_date() -> None:
    receipt = Receipt(claimant_date_of_birth="2026-04-08")

    assert receipt.claimant_date_of_birth == "08/04/2026"


def test_provider_name_fullerton_health_normalizes_to_null() -> None:
    referral_letter = ReferralLetter(provider_name="Fullerton Health")
    medical_certificate = MedicalCertificate(provider_name="Fullerton Health Clinic")
    receipt = Receipt(provider_name="Fullerton Health Pte Ltd")

    assert referral_letter.provider_name is None
    assert medical_certificate.provider_name is None
    assert receipt.provider_name is None


def test_receipt_amounts_round_decimals_instead_of_stripping_them() -> None:
    receipt = Receipt(
        tax_amount="S$80.50",
        total_amount="1,234.49",
    )

    assert receipt.tax_amount == 81
    assert receipt.total_amount == 1234


def test_referral_letter_amounts_round_decimals_instead_of_stripping_them() -> None:
    referral_letter = ReferralLetter(
        total_amount_paid=1200.5,
        total_approved_amount="$1,199.49",
        total_requested_amount=None,
    )

    assert referral_letter.total_amount_paid == 1201
    assert referral_letter.total_approved_amount == 1199
    assert referral_letter.total_requested_amount is None


def test_referral_letter_defaults_null_signature_presence_to_false() -> None:
    referral_letter = ReferralLetter(signature_presence=None)

    assert referral_letter.signature_presence is False
