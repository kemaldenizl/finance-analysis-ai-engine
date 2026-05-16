import numpy as np
from PIL import Image

from app.services.preprocessing.image_utils import ImageUtils
from app.services.preprocessing.quality_analyzer import QualityAnalyzer


class BasePreprocessor:
    def __init__(self):
        self.quality_analyzer = QualityAnalyzer()

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
        raise NotImplementedError


class ScreenshotPreprocessor(BasePreprocessor):
    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
        operations: list[str] = []
        warnings: list[str] = []

        image = ImageUtils.resize_if_too_large(image, max_side=2800)

        quality_before = self.quality_analyzer.analyze(image)

        gray = ImageUtils.to_grayscale(image)
        operations.append("grayscale")

        enhanced = ImageUtils.enhance_contrast(gray)
        operations.append("contrast_enhancement_clahe")

        # Screenshot'larda adaptive threshold bazen UI detaylarını fazla sertleştirir.
        # Bu yüzden sadece düşük kontrast varsa threshold uygula.
        if quality_before["contrast"] < 35:
            processed = ImageUtils.adaptive_threshold(enhanced)
            operations.append("adaptive_threshold")
        else:
            processed = enhanced

        processed, angle = ImageUtils.deskew(processed)
        if angle:
            operations.append(f"deskew:{round(angle, 4)}")

        quality_after = self.quality_analyzer.analyze(processed)

        return processed, operations, quality_before, quality_after, warnings


class CameraPhotoPreprocessor(BasePreprocessor):
    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
        operations: list[str] = []
        warnings: list[str] = []

        image = ImageUtils.resize_if_too_large(image, max_side=3200)

        quality_before = self.quality_analyzer.analyze(image)

        cropped, crop_found = ImageUtils.crop_document_if_found(image)

        if crop_found:
            image = cropped
            operations.append("document_crop_and_perspective_correction")
        else:
            warnings.append("document_contour_not_found")
            operations.append("document_crop_skipped")

        image = ImageUtils.denoise(image)
        operations.append("denoise")

        enhanced = ImageUtils.enhance_contrast(image)
        operations.append("contrast_enhancement_clahe")

        deskewed, angle = ImageUtils.deskew(enhanced)
        if angle:
            operations.append(f"deskew:{round(angle, 4)}")
        else:
            operations.append("deskew_skipped")

        thresholded = ImageUtils.adaptive_threshold(deskewed)
        operations.append("adaptive_threshold")

        cleaned = ImageUtils.remove_small_noise(thresholded)
        operations.append("small_noise_removal")

        quality_after = self.quality_analyzer.analyze(cleaned)

        if quality_after["quality_score"] < 0.45:
            warnings.append("low_quality_after_preprocessing")

        return cleaned, operations, quality_before, quality_after, warnings


class ScannedPdfPagePreprocessor(BasePreprocessor):
    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
        operations: list[str] = []
        warnings: list[str] = []

        image = ImageUtils.resize_if_too_large(image, max_side=3200)

        quality_before = self.quality_analyzer.analyze(image)

        denoised = ImageUtils.denoise(image)
        operations.append("denoise")

        enhanced = ImageUtils.enhance_contrast(denoised)
        operations.append("contrast_enhancement_clahe")

        deskewed, angle = ImageUtils.deskew(enhanced)

        if angle:
            operations.append(f"deskew:{round(angle, 4)}")
        else:
            operations.append("deskew_skipped")

        thresholded = ImageUtils.adaptive_threshold(deskewed)
        operations.append("adaptive_threshold")

        cleaned = ImageUtils.remove_small_noise(thresholded)
        operations.append("small_noise_removal")

        quality_after = self.quality_analyzer.analyze(cleaned)

        if quality_after["quality_score"] < 0.45:
            warnings.append("low_quality_after_preprocessing")

        return cleaned, operations, quality_before, quality_after, warnings


class RealPdfPagePreprocessor(BasePreprocessor):
    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
        operations: list[str] = []
        warnings: list[str] = []

        image = ImageUtils.resize_if_too_large(image, max_side=2800)

        quality_before = self.quality_analyzer.analyze(image)

        # Real PDF için ağır threshold yapmıyoruz.
        # Çünkü Stage 3 çoğunlukla text layer’dan veri çıkaracak.
        gray = ImageUtils.to_grayscale(image)
        operations.append("grayscale_preview")

        enhanced = ImageUtils.enhance_contrast(gray)
        operations.append("light_contrast_enhancement")

        quality_after = self.quality_analyzer.analyze(enhanced)

        return enhanced, operations, quality_before, quality_after, warnings