import cv2
import numpy as np

from app.services.preprocessing.image_utils import ImageUtils


class QualityAnalyzer:
    def analyze(self, image: np.ndarray) -> dict:
        gray = ImageUtils.to_grayscale(image)

        height, width = gray.shape[:2]

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        edges = cv2.Canny(gray, 80, 200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        dark_pixel_ratio = float(np.count_nonzero(gray < 30) / gray.size)
        bright_pixel_ratio = float(np.count_nonzero(gray > 225) / gray.size)

        quality_score = self._score(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            edge_density=edge_density,
            dark_pixel_ratio=dark_pixel_ratio,
            bright_pixel_ratio=bright_pixel_ratio,
        )

        return {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "edge_density": round(edge_density, 4),
            "dark_pixel_ratio": round(dark_pixel_ratio, 4),
            "bright_pixel_ratio": round(bright_pixel_ratio, 4),
            "quality_score": round(quality_score, 4),
        }

    def _score(
        self,
        blur_score: float,
        brightness: float,
        contrast: float,
        edge_density: float,
        dark_pixel_ratio: float,
        bright_pixel_ratio: float,
    ) -> float:
        score = 0.0

        if blur_score >= 500:
            score += 0.25
        elif blur_score >= 200:
            score += 0.15
        else:
            score += 0.05

        if 70 <= brightness <= 200:
            score += 0.20
        else:
            score += 0.08

        if contrast >= 45:
            score += 0.25
        elif contrast >= 25:
            score += 0.15
        else:
            score += 0.05

        if 0.02 <= edge_density <= 0.25:
            score += 0.20
        else:
            score += 0.08

        if dark_pixel_ratio < 0.35 and bright_pixel_ratio < 0.70:
            score += 0.10
        else:
            score += 0.04

        return min(score, 1.0)