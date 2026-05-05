from pathlib import Path

from app.schemas.classification import InputKind
from app.services.classification_service import ClassificationService


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def assert_kind(result, expected_kind: InputKind):
    assert result.kind == expected_kind, {
        "actual_kind": result.kind,
        "expected_kind": expected_kind,
        "confidence": result.confidence,
        "routing_key": result.routing_key,
        "features": result.features,
        "warnings": result.warnings,
    }


def test_classifies_mobile_app_screenshot_as_screenshot():
    file_path = FIXTURES_DIR / "screenshot.png"

    result = ClassificationService().classify(
        file_path=str(file_path),
        mime_type="image/png",
    )

    assert_kind(result, InputKind.SCREENSHOT)
    assert result.routing_key == "image.screenshot.ocr"
    assert result.needs_ocr is True
    assert result.needs_preprocessing is False
    assert result.confidence >= 0.80


def test_classifies_camera_photo_as_camera_photo():
    file_path = FIXTURES_DIR / "camera_photo.jpeg"

    result = ClassificationService().classify(
        file_path=str(file_path),
        mime_type="image/jpeg",
    )

    assert_kind(result, InputKind.CAMERA_PHOTO)
    assert result.routing_key == "image.camera_photo.preprocessing"
    assert result.needs_ocr is True
    assert result.needs_preprocessing is True
    assert result.confidence >= 0.75


def test_classifies_digital_pdf_as_real_pdf():
    file_path = FIXTURES_DIR / "real_pdf.pdf"

    result = ClassificationService().classify(
        file_path=str(file_path),
        mime_type="application/pdf",
    )

    assert_kind(result, InputKind.REAL_PDF)
    assert result.routing_key == "pdf.real.extract_text"
    assert result.needs_ocr is False
    assert result.needs_preprocessing is False
    assert result.confidence >= 0.85

    assert result.features["page_count"] >= 1
    assert result.features["avg_text_chars_per_page"] > 300
    assert result.features["text_page_ratio"] >= 0.60


def test_classifies_scanned_pdf_as_scanned_pdf():
    file_path = FIXTURES_DIR / "scanned_pdf.pdf"

    result = ClassificationService().classify(
        file_path=str(file_path),
        mime_type="application/pdf",
    )

    assert_kind(result, InputKind.SCANNED_PDF)
    assert result.routing_key == "pdf.scanned.ocr"
    assert result.needs_ocr is True
    assert result.needs_preprocessing is True
    assert result.confidence >= 0.85

    assert result.features["page_count"] >= 1
    assert result.features["avg_text_chars_per_page"] < 80
    assert result.features["large_image_page_ratio"] >= 0.60