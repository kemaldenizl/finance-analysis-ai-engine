from pathlib import Path

from app.services.mime_detector import MimeDetector


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_detects_png_mime_type():
    file_bytes = (FIXTURES_DIR / "screenshot.png").read_bytes()

    mime_type = MimeDetector.detect(file_bytes)

    assert mime_type == "image/png"


def test_detects_jpeg_mime_type():
    file_bytes = (FIXTURES_DIR / "camera_photo.jpeg").read_bytes()

    mime_type = MimeDetector.detect(file_bytes)

    assert mime_type == "image/jpeg"


def test_detects_pdf_mime_type():
    file_bytes = (FIXTURES_DIR / "real_pdf.pdf").read_bytes()

    mime_type = MimeDetector.detect(file_bytes)

    assert mime_type == "application/pdf"