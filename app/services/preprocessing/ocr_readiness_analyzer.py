import cv2
import numpy as np

from app.services.preprocessing.image_utils import ImageUtils


class OcrReadinessAnalyzer:
    def analyze(self, image: np.ndarray) -> dict:
        gray = ImageUtils.to_grayscale(image)

        height, width = gray.shape[:2]

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        foreground_density = float(np.count_nonzero(binary) / binary.size)

        component_count, avg_component_area = self._connected_component_stats(binary)
        border_artifact_score = self._border_artifact_score(binary)

        score = self._score(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            foreground_density=foreground_density,
            component_count=component_count,
            avg_component_area=avg_component_area,
            border_artifact_score=border_artifact_score,
        )

        return {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "foreground_density": round(foreground_density, 4),
            "component_count": component_count,
            "avg_component_area": round(avg_component_area, 4),
            "border_artifact_score": round(border_artifact_score, 4),
            "ocr_readiness_score": round(score, 4),
        }

    def _connected_component_stats(self, binary: np.ndarray) -> tuple[int, float]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        areas = []

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if 2 <= area <= 5000:
                areas.append(float(area))

        if not areas:
            return 0, 0.0

        return len(areas), float(sum(areas) / len(areas))

    def _border_artifact_score(self, binary: np.ndarray) -> float:
        height, width = binary.shape[:2]

        border = int(max(width, height) * 0.025)

        top = binary[:border, :]
        bottom = binary[-border:, :]
        left = binary[:, :border]
        right = binary[:, -border:]

        border_pixels = (
            np.count_nonzero(top)
            + np.count_nonzero(bottom)
            + np.count_nonzero(left)
            + np.count_nonzero(right)
        )

        total_border_area = top.size + bottom.size + left.size + right.size

        return float(border_pixels / max(total_border_area, 1))

    def _score(
        self,
        blur_score: float,
        brightness: float,
        contrast: float,
        foreground_density: float,
        component_count: int,
        avg_component_area: float,
        border_artifact_score: float,
    ) -> float:
        score = 0.0

        if blur_score >= 500:
            score += 0.20
        elif blur_score >= 180:
            score += 0.12
        else:
            score += 0.04

        if 80 <= brightness <= 235:
            score += 0.15
        else:
            score += 0.06

        if 30 <= contrast <= 95:
            score += 0.20
        else:
            score += 0.08

        if 0.01 <= foreground_density <= 0.22:
            score += 0.20
        else:
            score += 0.07

        if component_count >= 80:
            score += 0.15
        elif component_count >= 20:
            score += 0.08

        if 3 <= avg_component_area <= 400:
            score += 0.10
        else:
            score += 0.04

        if border_artifact_score <= 0.08:
            score += 0.10
        elif border_artifact_score <= 0.18:
            score += 0.04
        else:
            score -= 0.08

        return max(0.0, min(score, 1.0))