from app.schemas.classification import InputKind


class RoutingService:
    @staticmethod
    def get_routing_key(kind: InputKind) -> str:
        mapping = {
            InputKind.REAL_PDF: "pdf.real.extract_text",
            InputKind.SCANNED_PDF: "pdf.scanned.ocr",
            InputKind.HYBRID_PDF: "pdf.hybrid.extract_and_ocr",
            InputKind.SCREENSHOT: "image.screenshot.ocr",
            InputKind.CAMERA_PHOTO: "image.camera_photo.preprocessing",
            InputKind.UNKNOWN: "manual_review",
            InputKind.UNSUPPORTED: "unsupported",
        }

        return mapping[kind]

    @staticmethod
    def needs_ocr(kind: InputKind) -> bool:
        return kind in {
            InputKind.SCANNED_PDF,
            InputKind.HYBRID_PDF,
            InputKind.SCREENSHOT,
            InputKind.CAMERA_PHOTO,
        }

    @staticmethod
    def needs_preprocessing(kind: InputKind) -> bool:
        return kind in {
            InputKind.SCANNED_PDF,
            InputKind.HYBRID_PDF,
            InputKind.CAMERA_PHOTO,
        }