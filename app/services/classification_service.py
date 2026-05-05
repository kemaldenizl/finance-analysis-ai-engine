from app.schemas.classification import ClassificationResult, InputKind
from app.services.image_classifier import ImageClassifier
from app.services.mime_detector import MimeDetector
from app.services.pdf_classifier import PDFClassifier
from app.services.routing_service import RoutingService


class ClassificationService:
    def __init__(self):
        self.pdf_classifier = PDFClassifier()
        self.image_classifier = ImageClassifier()

    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        if MimeDetector.is_pdf(mime_type):
            return self.pdf_classifier.classify(
                file_path=file_path,
                mime_type=mime_type,
            )

        if MimeDetector.is_image(mime_type):
            return self.image_classifier.classify(
                file_path=file_path,
                mime_type=mime_type,
            )

        kind = InputKind.UNSUPPORTED

        return ClassificationResult(
            kind=kind,
            confidence=0.0,
            needs_ocr=False,
            needs_preprocessing=False,
            routing_key=RoutingService.get_routing_key(kind),
            features={
                "mime_type": mime_type,
            },
            warnings=[
                "unsupported_mime_type",
            ],
        )
