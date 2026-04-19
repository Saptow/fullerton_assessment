from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.schemas import (
    DocumentClassificationError,
    OCRResponse,
    OCRResult,
    ReferralLetter,
)
from app.services.ocr import OCRService


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_endpoint_accepts_multipart_form_data(monkeypatch) -> None:
    class FakeOCRService:
        def extract_text(self, **kwargs) -> OCRResponse:
            return OCRResponse(
                result=OCRResult(
                    document_type="referral_letter",
                    total_time=0.0,
                    final_json=ReferralLetter(),
                )
            )

    monkeypatch.setattr(routes, "OCRService", FakeOCRService)

    response = client.post(
        "/ocr",
        files={
            "file": ("sample.pdf", b"%PDF-1.7 test content", "application/pdf"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Processing completed."
    assert body["result"]["document_type"] == "referral_letter"
    assert body["result"]["total_time"] >= 0
    assert body["result"]["cleaning_elapsed"] is None
    assert body["result"]["classification_elapsed"] is None
    assert body["result"]["extraction_elapsed"] is None
    assert body["result"]["finalJson"] == {
        "claimant_name": None,
        "provider_name": None,
        "signature_presence": False,
        "total_amount_paid": None,
        "total_approved_amount": None,
        "total_requested_amount": None,
    }


def test_ocr_endpoint_rejects_empty_files() -> None:
    response = client.post(
        "/ocr",
        files={
            "file": ("empty.pdf", b"", "application/pdf"),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"error": "file_missing"}


def test_ocr_endpoint_rejects_missing_files() -> None:
    response = client.post("/ocr")

    assert response.status_code == 400
    assert response.json() == {"error": "file_missing"}


def test_ocr_endpoint_rejects_invalid_mime_type() -> None:
    response = client.post(
        "/ocr",
        files={
            "file": ("sample.pdf", b"Example OCR input", "application/octet-stream"),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_mime_type"}


def test_ocr_endpoint_rejects_unsupported_document_type() -> None:
    response = client.post(
        "/ocr",
        files={
            "file": ("sample.csv", b"Example OCR input", "application/pdf"),
        },
    )

    assert response.status_code == 422
    assert response.json() == {"error": "unsupported_document_type"}


def test_ocr_endpoint_rejects_low_confidence_classification(monkeypatch) -> None:
    def raise_classification_error(self, **kwargs) -> None:
        raise DocumentClassificationError("low confidence")

    monkeypatch.setattr(OCRService, "extract_text", raise_classification_error)

    response = client.post(
        "/ocr",
        files={
            "file": ("sample.png", b"Example OCR input", "image/png"),
        },
    )

    assert response.status_code == 422
    assert response.json() == {"error": "unsupported_document_type"}


def test_ocr_endpoint_handles_service_errors(monkeypatch) -> None:
    def raise_error(self, **kwargs) -> None:
        raise RuntimeError("OCR engine failed")

    monkeypatch.setattr(OCRService, "extract_text", raise_error)

    response = client.post(
        "/ocr",
        files={
            "file": ("sample.png", b"Example OCR input", "image/png"),
        },
    )

    assert response.status_code == 500
    assert response.json() == {"error": "internal_server_error"}
