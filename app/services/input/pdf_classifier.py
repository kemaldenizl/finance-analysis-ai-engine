import fitz

from app.schemas.classification import ClassificationResult, InputKind
from app.services.input.routing_service import RoutingService


class PDFClassifier:
    MIN_TEXT_CHARS_FOR_TEXT_PAGE = 80
    STRONG_AVG_TEXT_THRESHOLD = 300
    LARGE_IMAGE_PAGE_RATIO_THRESHOLD = 0.60
    LARGE_IMAGE_AREA_THRESHOLD = 0.65

    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            return self._unknown_result(
                reason=f"Could not open PDF: {str(exc)}",
                mime_type=mime_type,
            )

        page_count = len(doc)

        if page_count == 0:
            return self._unknown_result(
                reason="PDF has zero pages",
                mime_type=mime_type,
            )

        total_chars = 0
        total_words = 0
        pages_with_text = 0
        pages_with_large_image = 0

        total_image_area = 0.0
        total_page_area = 0.0

        page_features = []

        for page_index, page in enumerate(doc):
            text = page.get_text("text") or ""
            words = page.get_text("words") or []

            char_count = len(text.strip())
            word_count = len(words)

            total_chars += char_count
            total_words += word_count

            if char_count >= self.MIN_TEXT_CHARS_FOR_TEXT_PAGE:
                pages_with_text += 1

            page_area = page.rect.width * page.rect.height
            total_page_area += page_area

            image_area_for_page = 0.0

            images = page.get_images(full=True)

            for image in images:
                xref = image[0]

                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []

                for rect in rects:
                    image_area_for_page += rect.width * rect.height

            image_area_ratio_for_page = image_area_for_page / page_area if page_area > 0 else 0.0

            if image_area_ratio_for_page >= self.LARGE_IMAGE_AREA_THRESHOLD:
                pages_with_large_image += 1

            total_image_area += image_area_for_page

            page_features.append(
                {
                    "page_number": page_index + 1,
                    "char_count": char_count,
                    "word_count": word_count,
                    "image_count": len(images),
                    "image_area_ratio": round(image_area_ratio_for_page, 4),
                }
            )

        avg_text_chars = total_chars / page_count
        avg_words = total_words / page_count

        text_page_ratio = pages_with_text / page_count
        large_image_page_ratio = pages_with_large_image / page_count

        global_image_area_ratio = total_image_area / total_page_area if total_page_area > 0 else 0.0

        features = {
            "mime_type": mime_type,
            "page_count": page_count,
            "avg_text_chars_per_page": round(avg_text_chars, 2),
            "avg_words_per_page": round(avg_words, 2),
            "text_page_ratio": round(text_page_ratio, 4),
            "large_image_page_ratio": round(large_image_page_ratio, 4),
            "global_image_area_ratio": round(global_image_area_ratio, 4),
            "pages": page_features[:5],
        }

        kind, confidence, warnings = self._decide(
            avg_text_chars=avg_text_chars,
            text_page_ratio=text_page_ratio,
            large_image_page_ratio=large_image_page_ratio,
            global_image_area_ratio=global_image_area_ratio,
        )

        routing_key = RoutingService.get_routing_key(kind)

        return ClassificationResult(
            kind=kind,
            confidence=confidence,
            needs_ocr=RoutingService.needs_ocr(kind),
            needs_preprocessing=RoutingService.needs_preprocessing(kind),
            routing_key=routing_key,
            features=features,
            warnings=warnings,
        )

    def _decide(
        self,
        avg_text_chars: float,
        text_page_ratio: float,
        large_image_page_ratio: float,
        global_image_area_ratio: float,
    ) -> tuple[InputKind, float, list[str]]:
        warnings = []

        if avg_text_chars >= self.STRONG_AVG_TEXT_THRESHOLD and text_page_ratio >= 0.60:
            return InputKind.REAL_PDF, 0.94, warnings

        if avg_text_chars < 80 and large_image_page_ratio >= self.LARGE_IMAGE_PAGE_RATIO_THRESHOLD:
            return InputKind.SCANNED_PDF, 0.92, warnings

        if text_page_ratio > 0.15 and large_image_page_ratio > 0.15:
            warnings.append("mixed_text_and_image_pdf")
            return InputKind.HYBRID_PDF, 0.78, warnings

        if global_image_area_ratio > 0.50 and avg_text_chars < 150:
            warnings.append("likely_scanned_pdf_but_low_confidence")
            return InputKind.SCANNED_PDF, 0.74, warnings

        warnings.append("pdf_classification_uncertain")
        return InputKind.UNKNOWN, 0.50, warnings

    def _unknown_result(self, reason: str, mime_type: str) -> ClassificationResult:
        kind = InputKind.UNKNOWN

        return ClassificationResult(
            kind=kind,
            confidence=0.0,
            needs_ocr=False,
            needs_preprocessing=False,
            routing_key=RoutingService.get_routing_key(kind),
            features={
                "mime_type": mime_type,
                "reason": reason,
            },
            warnings=[reason],
        )
