from dataclasses import dataclass
from pathlib import Path

from app.services.extraction.ocr_text_extractor import OcrLine, OcrTextExtractor
from app.services.extraction.pdf_transaction_parser import PdfTransactionParser


@dataclass
class OcrVariantInput:
    page: int
    variant: str
    storage_url: str
    is_preferred: bool = False
    purpose: str | None = None


@dataclass
class OcrVariantResult:
    page: int
    variant: str
    image_path: str
    psm: int
    lines: list[OcrLine]
    ocr_score: float
    final_score: float
    error: str | None = None


class OcrVariantSelector:
    VARIANT_PRIORITY = {
        "enhanced_grayscale": 100,
        "ocr_grayscale": 100,
        "rendered_original": 90,
        "normalized_original": 90,
        "original": 80,
        "color": 75,
        "grayscale": 70,
        "gray": 70,
        "thresholded": 55,
        "threshold": 55,
        "binary": 50,
        "bw": 50,
    }

    PSM_CANDIDATES = [6, 11, 4]

    def __init__(self):
        self.ocr_extractor = OcrTextExtractor()
        self.transaction_parser = PdfTransactionParser()

    def select_best_for_page(
        self,
        variants: list[OcrVariantInput],
    ) -> tuple[OcrVariantResult, list[OcrVariantResult]]:
        results: list[OcrVariantResult] = []

        for variant_input in variants:
            for psm in self.PSM_CANDIDATES:
                try:
                    lines = self.ocr_extractor.extract_lines(
                        image_path=variant_input.storage_url,
                        page=variant_input.page,
                        variant=variant_input.variant,
                        psm=psm,
                    )

                    ocr_score = self.score_lines(lines)

                    final_score = (
                        ocr_score
                        + self.variant_priority(variant_input.variant) / 100
                        + (0.15 if variant_input.is_preferred else 0.0)
                    )

                    results.append(
                        OcrVariantResult(
                            page=variant_input.page,
                            variant=variant_input.variant,
                            image_path=variant_input.storage_url,
                            psm=psm,
                            lines=lines,
                            ocr_score=round(ocr_score, 4),
                            final_score=round(final_score, 4),
                        )
                    )

                except Exception as exc:
                    results.append(
                        OcrVariantResult(
                            page=variant_input.page,
                            variant=variant_input.variant,
                            image_path=variant_input.storage_url,
                            psm=psm,
                            lines=[],
                            ocr_score=-1.0,
                            final_score=-1.0,
                            error=str(exc),
                        )
                    )

        if not results:
            raise RuntimeError("No OCR variant result generated")

        results.sort(key=lambda item: item.final_score, reverse=True)

        return results[0], results

    def score_lines(self, lines: list[OcrLine]) -> float:
        if not lines:
            return -1.0

        text = "\n".join(line.text for line in lines)

        date_count = sum(
            1
            for line in lines
            if self.transaction_parser.find_date(line.text)
        )

        money_count = sum(
            len(self.transaction_parser.find_money_values(line.text))
            for line in lines
        )

        candidate_count = sum(
            1
            for line in lines
            if self.transaction_parser.find_date(line.text)
            and self.transaction_parser.find_money_values(line.text)
        )

        avg_ocr_confidence = sum(line.ocr_confidence for line in lines) / len(lines)
        text_length = len(text)

        score = 0.0
        score += candidate_count * 8
        score += date_count * 2
        score += money_count * 1.5
        score += avg_ocr_confidence * 15
        score += min(text_length / 500, 8)

        return round(score, 4)

    def variant_priority(self, variant: str) -> int:
        lower = variant.lower()

        for key, priority in self.VARIANT_PRIORITY.items():
            if key in lower:
                return priority

        return 60

    def debug_result(self, result: OcrVariantResult) -> dict:
        return {
            "image": Path(result.image_path).name,
            "variant": result.variant,
            "psm": result.psm,
            "ocr_score": result.ocr_score,
            "final_score": result.final_score,
            "line_count": len(result.lines),
            "error": result.error,
        }