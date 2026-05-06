from dataclasses import dataclass

import cv2
import numpy as np

from app.services.preprocessing.image_utils import ImageUtils


@dataclass
class FinanceRegion:
    bbox: tuple[int, int, int, int]
    score: float
    reason: str


class FinanceRegionDetector:
    def detect(self, image: np.ndarray, source_kind: str) -> FinanceRegion:
        if source_kind == "real_pdf":
            return self._detect_for_pdf_preview(image)

        if source_kind == "screenshot":
            return self._detect_for_screenshot(image)

        if source_kind in {"camera_photo", "scanned_pdf", "hybrid_pdf"}:
            return self._detect_for_document_image(image)

        return self._full_region(image, "fallback_full_region")

    def _detect_for_screenshot(self, image: np.ndarray) -> FinanceRegion:
        height, width = image.shape[:2]

        # Mobil screenshotlarda üst status/header ve alt nav genellikle gereksizdir.
        top_cut = int(height * 0.12)
        bottom_cut = int(height * 0.90)

        candidate = image[top_cut:bottom_cut, :]

        content_bbox = self._detect_dense_content_bbox(candidate)

        if content_bbox is None:
            return FinanceRegion(
                bbox=(0, top_cut, width, bottom_cut - top_cut),
                score=0.55,
                reason="screenshot_default_middle_region",
            )

        x, y, w, h = content_bbox

        return FinanceRegion(
            bbox=(x, y + top_cut, w, h),
            score=0.72,
            reason="screenshot_dense_financial_region",
        )

    def _detect_for_pdf_preview(self, image: np.ndarray) -> FinanceRegion:
        height, width = image.shape[:2]

        # Real PDF'de footer ve banka header/logolarını kırpmak için geniş ana içerik bölgesi.
        top_cut = int(height * 0.06)
        bottom_cut = int(height * 0.92)

        candidate = image[top_cut:bottom_cut, :]

        content_bbox = self._detect_dense_content_bbox(candidate)

        if content_bbox is None:
            return FinanceRegion(
                bbox=(0, top_cut, width, bottom_cut - top_cut),
                score=0.60,
                reason="pdf_default_content_region",
            )

        x, y, w, h = content_bbox

        return FinanceRegion(
            bbox=(x, y + top_cut, w, h),
            score=0.78,
            reason="pdf_dense_financial_region",
        )

    def _detect_for_document_image(self, image: np.ndarray) -> FinanceRegion:
        doc_image, document_found = ImageUtils.crop_document_if_found(image)

        if document_found:
            content_bbox = self._detect_dense_content_bbox(doc_image)

            if content_bbox is None:
                h, w = doc_image.shape[:2]
                return FinanceRegion(
                    bbox=(0, 0, w, h),
                    score=0.65,
                    reason="document_contour_found_full_document",
                )

            return FinanceRegion(
                bbox=content_bbox,
                score=0.82,
                reason="document_contour_found_dense_region",
            )

        trimmed, trim_bbox = ImageUtils.trim_white_border(image)

        content_bbox = self._detect_dense_content_bbox(trimmed)

        if content_bbox is None:
            h, w = image.shape[:2]
            return FinanceRegion(
                bbox=(0, 0, w, h),
                score=0.35,
                reason="document_region_fallback_full_image",
            )

        x, y, w, h = content_bbox

        if trim_bbox is not None:
            trim_x, trim_y, _, _ = trim_bbox
            x += trim_x
            y += trim_y

        return FinanceRegion(
            bbox=(x, y, w, h),
            score=0.58,
            reason="document_contour_missing_dense_region_fallback",
        )

    def _detect_dense_content_bbox(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = ImageUtils.to_grayscale(image)

        # Arka planı at, yazı/çizgi içeren pikselleri yakala.
        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        height, width = binary.shape[:2]

        # Header logo/banner gibi çok geniş koyu blokları azaltmak için yatay büyük lekeleri aç.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 12, 20), 2))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(height // 30, 20)))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        text_like = cv2.bitwise_or(binary, horizontal_lines)
        text_like = cv2.bitwise_or(text_like, vertical_lines)

        # Küçük metinleri aynı region'a bağla.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        connected = cv2.dilate(text_like, kernel, iterations=2)

        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        boxes: list[tuple[int, int, int, int, float]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < width * height * 0.005:
                continue

            if h < height * 0.025:
                continue

            density = self._foreground_density(binary, x, y, w, h)
            score = self._score_box(x, y, w, h, width, height, density)

            boxes.append((x, y, w, h, score))

        if not boxes:
            return None

        # Çok küçük parçaları değil, finansal tablo/metin yoğunluğu olan ana alanları birleştir.
        selected = [box for box in boxes if box[4] >= 0.15]

        if not selected:
            selected = sorted(boxes, key=lambda item: item[4], reverse=True)[:5]

        x1 = min(box[0] for box in selected)
        y1 = min(box[1] for box in selected)
        x2 = max(box[0] + box[2] for box in selected)
        y2 = max(box[1] + box[3] for box in selected)

        pad_x = int(width * 0.025)
        pad_y = int(height * 0.025)

        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, width)
        y2 = min(y2 + pad_y, height)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2 - x1, y2 - y1)

    def _foreground_density(
        self,
        binary: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        region = binary[y : y + height, x : x + width]

        if region.size == 0:
            return 0.0

        return float(np.count_nonzero(region) / region.size)

    def _score_box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
        density: float,
    ) -> float:
        area_ratio = (width * height) / max(image_width * image_height, 1)

        vertical_center = y + height / 2
        center_preference = 1.0 - abs((vertical_center / image_height) - 0.55)

        table_like_bonus = 0.0

        if width > image_width * 0.45:
            table_like_bonus += 0.10

        if height > image_height * 0.10:
            table_like_bonus += 0.10

        density_score = min(density * 8, 1.0)

        return float(
            0.40 * area_ratio
            + 0.25 * center_preference
            + 0.25 * density_score
            + table_like_bonus
        )

    def _full_region(self, image: np.ndarray, reason: str) -> FinanceRegion:
        height, width = image.shape[:2]

        return FinanceRegion(
            bbox=(0, 0, width, height),
            score=0.25,
            reason=reason,
        )