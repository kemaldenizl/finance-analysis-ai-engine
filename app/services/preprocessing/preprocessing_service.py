from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings
from app.schemas.preprocessing import PreprocessedPageOutput, PreprocessingResult
from app.services.preprocessing.image_utils import ImageUtils
from app.services.preprocessing.pdf_renderer import PDFRenderer
from app.services.preprocessing.preprocessors import (
    CameraPhotoPreprocessor,
    RealPdfPagePreprocessor,
    ScannedPdfPagePreprocessor,
    ScreenshotPreprocessor,
)
from app.storage.processed_storage import ProcessedStorage


class PreprocessingService:
    def __init__(self):
        self.pdf_renderer = PDFRenderer()
        self.processed_storage = ProcessedStorage()

    def preprocess(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        if source_kind in {"real_pdf", "scanned_pdf", "hybrid_pdf"}:
            return self._preprocess_pdf(
                input_id=input_id,
                source_path=source_path,
                source_kind=source_kind,
            )

        if source_kind in {"screenshot", "camera_photo"}:
            return self._preprocess_single_image(
                input_id=input_id,
                source_path=source_path,
                source_kind=source_kind,
            )

        return PreprocessingResult(
            input_id=input_id,
            source_kind=source_kind,
            status="skipped",
            output_type="none",
            page_count=0,
            outputs=[],
            warnings=["unsupported_source_kind_for_preprocessing"],
            is_ready_for_extraction=False,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _preprocess_pdf(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        page_images = self.pdf_renderer.render_pages(source_path)

        outputs: list[PreprocessedPageOutput] = []
        all_operations: list[str] = []
        all_warnings: list[str] = []

        quality_before_scores: list[float] = []
        quality_after_scores: list[float] = []

        for index, pil_image in enumerate(page_images, start=1):
            cv_image = ImageUtils.pil_to_cv(pil_image)

            preprocessor = self._get_pdf_page_preprocessor(source_kind)

            processed, operations, quality_before, quality_after, warnings = (
                preprocessor.preprocess_image(cv_image)
            )

            image_bytes = ImageUtils.encode_png(processed)

            storage_key, storage_url = self.processed_storage.save_page_image(
                input_id=input_id,
                page_number=index,
                image_bytes=image_bytes,
                extension=".png",
            )

            height, width = processed.shape[:2]

            outputs.append(
                PreprocessedPageOutput(
                    page_number=index,
                    storage_key=storage_key,
                    storage_url=storage_url,
                    width=width,
                    height=height,
                    operations=operations,
                    quality_before=quality_before,
                    quality_after=quality_after,
                    warnings=warnings,
                )
            )

            all_operations.extend(operations)
            all_warnings.extend(warnings)

            quality_before_scores.append(quality_before.get("quality_score", 0.0))
            quality_after_scores.append(quality_after.get("quality_score", 0.0))

        return PreprocessingResult(
            input_id=input_id,
            source_kind=source_kind,
            status="completed",
            output_type="page_images",
            output_storage_key=f"processed/{input_id}",
            output_storage_url=str(settings.LOCAL_PROCESSED_STORAGE_DIR / input_id),
            page_count=len(outputs),
            outputs=outputs,
            operations=sorted(set(all_operations)),
            quality_before={
                "average_quality_score": self._average(quality_before_scores),
            },
            quality_after={
                "average_quality_score": self._average(quality_after_scores),
            },
            warnings=sorted(set(all_warnings)),
            average_quality_score_before=self._average(quality_before_scores),
            average_quality_score_after=self._average(quality_after_scores),
            is_ready_for_extraction=True,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _preprocess_single_image(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        cv_image = ImageUtils.read_image(source_path)

        preprocessor = self._get_image_preprocessor(source_kind)

        processed, operations, quality_before, quality_after, warnings = (
            preprocessor.preprocess_image(cv_image)
        )

        image_bytes = ImageUtils.encode_png(processed)

        storage_key, storage_url = self.processed_storage.save_single_image(
            input_id=input_id,
            image_bytes=image_bytes,
            extension=".png",
        )

        height, width = processed.shape[:2]

        output = PreprocessedPageOutput(
            page_number=1,
            storage_key=storage_key,
            storage_url=storage_url,
            width=width,
            height=height,
            operations=operations,
            quality_before=quality_before,
            quality_after=quality_after,
            warnings=warnings,
        )

        is_ready = quality_after.get("quality_score", 0.0) >= 0.35

        return PreprocessingResult(
            input_id=input_id,
            source_kind=source_kind,
            status="completed",
            output_type="single_image",
            output_storage_key=storage_key,
            output_storage_url=storage_url,
            page_count=1,
            outputs=[output],
            operations=operations,
            quality_before=quality_before,
            quality_after=quality_after,
            warnings=warnings,
            average_quality_score_before=quality_before.get("quality_score"),
            average_quality_score_after=quality_after.get("quality_score"),
            is_ready_for_extraction=is_ready,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _get_image_preprocessor(self, source_kind: str):
        if source_kind == "screenshot":
            return ScreenshotPreprocessor()

        if source_kind == "camera_photo":
            return CameraPhotoPreprocessor()

        return CameraPhotoPreprocessor()

    def _get_pdf_page_preprocessor(self, source_kind: str):
        if source_kind == "real_pdf":
            return RealPdfPagePreprocessor()

        if source_kind in {"scanned_pdf", "hybrid_pdf"}:
            return ScannedPdfPagePreprocessor()

        return ScannedPdfPagePreprocessor()

    def _average(self, values: list[float]) -> float | None:
        if not values:
            return None

        return round(sum(values) / len(values), 4)