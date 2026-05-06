import re
from dataclasses import dataclass

import fitz
import numpy as np

from app.services.preprocessing.finance_region_detector import FinanceRegionDetector
from app.services.preprocessing.image_utils import ImageUtils


@dataclass
class PageRelevance:
    page_number: int
    keep: bool
    score: float
    reason: str
    text_score: float = 0.0
    visual_score: float = 0.0


class PageRelevanceDetector:
    FINANCE_KEYWORDS = [
        "işlem tarihi",
        "dönem içi işlemler",
        "tutar",
        "toplam",
        "harcama",
        "harcamalar",
        "borcunuz",
        "dönem borcunuz",
        "son ödeme",
        "minimum ödeme",
        "kart numarası",
        "ekstre",
        "bakiye",
        "ödeme",
        "limit",
        "bonus",
        "tl",
        "usd",
        "eur",
    ]

    NEGATIVE_KEYWORDS = [
        "sözleşme değişikliği",
        "ücret bilgileri",
        "kartın yurt dışında kullanımı",
        "ekstreniz ile ilgili açıklamalar",
        "ödeme ve faiz bilgileri",
        "yıllık kart ücretleri",
        "kampanya",
    ]

    def __init__(self):
        self.region_detector = FinanceRegionDetector()

    def analyze_real_pdf_page(
        self,
        doc: fitz.Document,
        page_index: int,
        rendered_image: np.ndarray,
    ) -> PageRelevance:
        page = doc[page_index]
        text = page.get_text("text") or ""

        text_score = self._score_text(text)
        visual_score = self._score_visual(rendered_image)

        score = 0.70 * text_score + 0.30 * visual_score

        keep = score >= 0.35

        reason = "real_pdf_financial_page" if keep else "real_pdf_non_financial_page"

        return PageRelevance(
            page_number=page_index + 1,
            keep=keep,
            score=round(score, 4),
            reason=reason,
            text_score=round(text_score, 4),
            visual_score=round(visual_score, 4),
        )

    def analyze_image_page(
        self,
        image: np.ndarray,
        page_number: int,
        source_kind: str,
    ) -> PageRelevance:
        visual_score = self._score_visual(image)

        keep = visual_score >= 0.35

        reason = (
            f"{source_kind}_visual_financial_page"
            if keep
            else f"{source_kind}_visual_non_financial_page"
        )

        return PageRelevance(
            page_number=page_number,
            keep=keep,
            score=round(visual_score, 4),
            reason=reason,
            visual_score=round(visual_score, 4),
        )

    def _score_text(self, text: str) -> float:
        normalized = self._normalize_text(text)

        if not normalized.strip():
            return 0.0

        keyword_hits = sum(1 for keyword in self.FINANCE_KEYWORDS if keyword in normalized)
        negative_hits = sum(1 for keyword in self.NEGATIVE_KEYWORDS if keyword in normalized)

        money_hits = len(
            re.findall(
                r"\\b\\d{1,3}(?:\\.\\d{3})*,\\d{2}\\s*(?:tl|usd|eur|czk|pln)?\\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

        date_hits = len(
            re.findall(
                r"\\b\\d{1,2}\\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\\s+\\d{4}\\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

        line_count = len([line for line in normalized.splitlines() if line.strip()])

        score = 0.0

        score += min(keyword_hits * 0.06, 0.30)
        score += min(money_hits * 0.025, 0.35)
        score += min(date_hits * 0.03, 0.20)

        if line_count >= 15:
            score += 0.10

        # Sadece açıklama/terms sayfası gibi görünen sayfaları düşür.
        if negative_hits >= 3 and money_hits < 5:
            score -= 0.35

        if "dönem içi işlemler" in normalized or "işlem tarihi" in normalized:
            score += 0.25

        if "ekstre özeti" in normalized:
            score += 0.10

        return max(0.0, min(score, 1.0))

    def _score_visual(self, image: np.ndarray) -> float:
        gray = ImageUtils.to_grayscale(image)
        height, width = gray.shape[:2]

        binary = cv2_safe_threshold(gray)

        foreground_density = float(np.count_nonzero(binary) / binary.size)

        horizontal_kernel = np.ones((1, max(width // 25, 20)), np.uint8)
        vertical_kernel = np.ones((max(height // 60, 15), 1), np.uint8)

        horizontal_lines = cv2_safe_open(binary, horizontal_kernel)
        vertical_lines = cv2_safe_open(binary, vertical_kernel)

        line_density = float(
            (np.count_nonzero(horizontal_lines) + np.count_nonzero(vertical_lines))
            / binary.size
        )

        region = self.region_detector.detect(image, source_kind="generic")
        region_area_ratio = (region.bbox[2] * region.bbox[3]) / max(width * height, 1)

        score = 0.0

        if 0.01 <= foreground_density <= 0.30:
            score += 0.25

        score += min(line_density * 10, 0.30)
        score += min(region_area_ratio * 0.70, 0.30)

        if region.score >= 0.5:
            score += 0.15

        return max(0.0, min(score, 1.0))

    def _normalize_text(self, text: str) -> str:
        return text.lower().replace("ı", "i")


def cv2_safe_threshold(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]


def cv2_safe_open(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)