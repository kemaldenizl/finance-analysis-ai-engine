from dataclasses import dataclass

import numpy as np

from app.services.preprocessing.image_utils import ImageUtils
from app.services.preprocessing.ocr_readiness_analyzer import OcrReadinessAnalyzer


@dataclass
class ImageVariant:
    variant: str
    purpose: str
    image: np.ndarray
    operations: list[str]
    quality: dict
    is_preferred_candidate: bool = False
    warnings: list[str] | None = None


class VariantBuilder:
    def __init__(self):
        self.ocr_analyzer = OcrReadinessAnalyzer()

    def build_screenshot_variants(self, image: np.ndarray) -> list[ImageVariant]:
        normalized = ImageUtils.normalize_original(image, max_side=2800)
        grayscale = ImageUtils.soft_enhance_grayscale(normalized)

        return [
            ImageVariant(
                variant="normalized_original",
                purpose="color_preservation",
                image=normalized,
                operations=["normalize_original"],
                quality=self.ocr_analyzer.analyze(normalized),
                is_preferred_candidate=False,
            ),
            ImageVariant(
                variant="ocr_grayscale",
                purpose="primary_ocr",
                image=grayscale,
                operations=["grayscale", "soft_contrast_enhancement"],
                quality=self.ocr_analyzer.analyze(grayscale),
                is_preferred_candidate=True,
            ),
        ]

    def build_camera_photo_variants(self, image: np.ndarray) -> list[ImageVariant]:
        normalized = ImageUtils.normalize_original(image, max_side=3200)

        denoised = ImageUtils.denoise(normalized)
        enhanced = ImageUtils.enhance_contrast(denoised)
        deskewed, angle = ImageUtils.deskew(enhanced)

        operations = ["normalize_original", "denoise", "contrast_enhancement_clahe"]

        if angle:
            operations.append(f"deskew:{round(angle, 4)}")
        else:
            operations.append("deskew_skipped")

        thresholded = ImageUtils.adaptive_threshold(deskewed)
        thresholded = ImageUtils.remove_small_noise(thresholded)

        return [
            ImageVariant(
                variant="normalized_original",
                purpose="color_preservation",
                image=normalized,
                operations=["normalize_original"],
                quality=self.ocr_analyzer.analyze(normalized),
                is_preferred_candidate=False,
            ),
            ImageVariant(
                variant="enhanced_grayscale",
                purpose="primary_ocr",
                image=deskewed,
                operations=operations,
                quality=self.ocr_analyzer.analyze(deskewed),
                is_preferred_candidate=True,
            ),
            ImageVariant(
                variant="thresholded",
                purpose="secondary_ocr",
                image=thresholded,
                operations=operations + ["adaptive_threshold", "small_noise_removal"],
                quality=self.ocr_analyzer.analyze(thresholded),
                is_preferred_candidate=False,
            ),
        ]

    def build_scanned_pdf_variants(self, image: np.ndarray) -> list[ImageVariant]:
        normalized = ImageUtils.normalize_original(image, max_side=3200)

        denoised = ImageUtils.denoise(normalized)
        enhanced = ImageUtils.enhance_contrast(denoised)
        deskewed, angle = ImageUtils.deskew(enhanced)

        operations = ["rendered_original", "denoise", "contrast_enhancement_clahe"]

        if angle:
            operations.append(f"deskew:{round(angle, 4)}")
        else:
            operations.append("deskew_skipped")

        thresholded = ImageUtils.adaptive_threshold(deskewed)
        thresholded = ImageUtils.remove_small_noise(thresholded)

        return [
            ImageVariant(
                variant="rendered_original",
                purpose="fallback_ocr",
                image=normalized,
                operations=["rendered_original"],
                quality=self.ocr_analyzer.analyze(normalized),
                is_preferred_candidate=False,
            ),
            ImageVariant(
                variant="enhanced_grayscale",
                purpose="primary_ocr",
                image=deskewed,
                operations=operations,
                quality=self.ocr_analyzer.analyze(deskewed),
                is_preferred_candidate=True,
            ),
            ImageVariant(
                variant="thresholded",
                purpose="secondary_ocr",
                image=thresholded,
                operations=operations + ["adaptive_threshold", "small_noise_removal"],
                quality=self.ocr_analyzer.analyze(thresholded),
                is_preferred_candidate=False,
            ),
        ]

    def build_real_pdf_preview_variants(self, image: np.ndarray) -> list[ImageVariant]:
        normalized = ImageUtils.normalize_original(image, max_side=2800)
        preview = ImageUtils.soft_enhance_grayscale(normalized)

        return [
            ImageVariant(
                variant="page_preview",
                purpose="debug_preview",
                image=preview,
                operations=["grayscale_preview", "light_contrast_enhancement"],
                quality=self.ocr_analyzer.analyze(preview),
                is_preferred_candidate=False,
            )
        ]