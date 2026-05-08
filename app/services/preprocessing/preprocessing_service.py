from app.core.config import settings
from app.schemas.preprocessing import PreprocessedPageOutput, PreprocessingResult
from app.services.preprocessing.image_utils import ImageUtils
from app.services.preprocessing.pdf_renderer import PDFRenderer
from app.services.preprocessing.variant_builder import ImageVariant, VariantBuilder
from app.storage.processed_storage import ProcessedStorage


class PreprocessingService:
    def __init__(self):
        self.pdf_renderer = PDFRenderer()
        self.processed_storage = ProcessedStorage()
        self.variant_builder = VariantBuilder()

    def preprocess(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        if source_kind == "real_pdf":
            return self._preprocess_real_pdf(
                input_id=input_id,
                source_path=source_path,
                source_kind=source_kind,
            )

        if source_kind in {"scanned_pdf", "hybrid_pdf"}:
            return self._preprocess_scanned_pdf(
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
            preferred_extraction_method="manual_review",
            extraction_risk="high",
            is_ready_for_extraction=False,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _preprocess_single_image(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        image = ImageUtils.read_image(source_path)

        if source_kind == "screenshot":
            variants = self.variant_builder.build_screenshot_variants(image)
            preferred_method = "ocr_multi_variant"
            extraction_risk = "low"
        else:
            variants = self.variant_builder.build_camera_photo_variants(image)
            preferred_method = "ocr_multi_variant"
            extraction_risk = "medium"

        outputs = self._save_variants(
            input_id=input_id,
            page_number=1,
            variants=variants,
        )

        preferred = self._select_preferred_output(outputs)

        return self._build_result(
            input_id=input_id,
            source_kind=source_kind,
            output_type="multi_variant_full_image",
            outputs=outputs,
            page_count=1,
            preferred=preferred,
            preferred_extraction_method=preferred_method,
            extraction_risk=extraction_risk,
            warnings=[],
        )

    def _preprocess_real_pdf(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        doc = self.pdf_renderer.open_document(source_path)
        page_count = len(doc)

        storage_key = self._input_storage_key_from_path(source_path)

        return PreprocessingResult(
            input_id=input_id,
            source_kind=source_kind,
            status="completed",
            output_type="native_pdf_reference",
            output_storage_key=storage_key,
            output_storage_url=source_path,
            preferred_output_storage_key=storage_key,
            preferred_output_storage_url=source_path,
            preferred_output_variant="original_pdf",
            preferred_extraction_method="native_pdf_text",
            extraction_risk="low",
            page_count=page_count,
            outputs=[],
            operations=[
                "preserve_original_pdf",
                "skip_image_rendering",
                "skip_ocr",
            ],
            quality_before={},
            quality_after={
                "native_pdf_available": True,
                "page_count": page_count,
            },
            warnings=[],
            average_quality_score_before=None,
            average_quality_score_after=None,
            ocr_readiness_score=None,
            is_ready_for_extraction=True,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _preprocess_scanned_pdf(
        self,
        input_id: str,
        source_path: str,
        source_kind: str,
    ) -> PreprocessingResult:
        page_images = self.pdf_renderer.render_pages(source_path)

        outputs: list[PreprocessedPageOutput] = []

        for index, pil_image in enumerate(page_images, start=1):
            cv_image = ImageUtils.pil_to_cv(pil_image)

            variants = self.variant_builder.build_scanned_pdf_variants(cv_image)

            outputs.extend(
                self._save_variants(
                    input_id=input_id,
                    page_number=index,
                    variants=variants,
                )
            )

        preferred = self._select_preferred_output(outputs)

        return self._build_result(
            input_id=input_id,
            source_kind=source_kind,
            output_type="multi_variant_full_page_images",
            outputs=outputs,
            page_count=len(page_images),
            preferred=preferred,
            preferred_extraction_method="ocr_multi_variant",
            extraction_risk="medium",
            warnings=[],
        )

    def _save_variants(
        self,
        input_id: str,
        page_number: int,
        variants: list[ImageVariant],
    ) -> list[PreprocessedPageOutput]:
        outputs: list[PreprocessedPageOutput] = []

        preferred_variant_name = self._pick_preferred_variant(variants)

        for variant in variants[: settings.PREPROCESSING_MAX_OUTPUT_VARIANTS_PER_PAGE]:
            image_bytes = ImageUtils.encode_png(variant.image)

            storage_key, storage_url = self.processed_storage.save_page_image(
                input_id=input_id,
                page_number=page_number,
                image_bytes=image_bytes,
                extension=".png",
                variant=variant.variant,
            )

            height, width = variant.image.shape[:2]

            is_preferred = variant.variant == preferred_variant_name

            output = PreprocessedPageOutput(
                page_number=page_number,
                variant=variant.variant,
                purpose=variant.purpose,
                is_preferred=is_preferred,
                storage_key=storage_key,
                storage_url=storage_url,
                width=width,
                height=height,
                operations=variant.operations,
                quality_before={},
                quality_after=variant.quality,
                warnings=variant.warnings or [],
            )

            outputs.append(output)

        return outputs

    def _pick_preferred_variant(self, variants: list[ImageVariant]) -> str | None:
        preferred_candidates = [
            variant for variant in variants if variant.is_preferred_candidate
        ]

        if not preferred_candidates:
            preferred_candidates = variants

        if not preferred_candidates:
            return None

        preferred_candidates.sort(
            key=lambda item: item.quality.get("ocr_readiness_score", 0.0),
            reverse=True,
        )

        return preferred_candidates[0].variant

    def _select_preferred_output(
        self,
        outputs: list[PreprocessedPageOutput],
    ) -> PreprocessedPageOutput | None:
        preferred = [output for output in outputs if output.is_preferred]

        if preferred:
            preferred.sort(
                key=lambda output: output.quality_after.get("ocr_readiness_score", 0.0),
                reverse=True,
            )

            return preferred[0]

        if not outputs:
            return None

        return sorted(
            outputs,
            key=lambda output: output.quality_after.get("ocr_readiness_score", 0.0),
            reverse=True,
        )[0]

    def _build_result(
        self,
        input_id: str,
        source_kind: str,
        output_type: str,
        outputs: list[PreprocessedPageOutput],
        page_count: int,
        preferred: PreprocessedPageOutput | None,
        preferred_extraction_method: str,
        extraction_risk: str,
        warnings: list[str],
    ) -> PreprocessingResult:
        operations = sorted(
            {
                operation
                for output in outputs
                for operation in output.operations
            }
        )

        ocr_scores = [
            output.quality_after.get("ocr_readiness_score", 0.0)
            for output in outputs
        ]

        average_ocr_score = self._average(ocr_scores)

        if preferred is None:
            is_ready = False
        else:
            is_ready = average_ocr_score is None or average_ocr_score >= 0.35

        return PreprocessingResult(
            input_id=input_id,
            source_kind=source_kind,
            status="completed" if outputs else "skipped",
            output_type=output_type,
            output_storage_key=f"processed/{input_id}",
            output_storage_url=str(settings.LOCAL_PROCESSED_STORAGE_DIR / input_id),
            preferred_output_storage_key=preferred.storage_key if preferred else None,
            preferred_output_storage_url=preferred.storage_url if preferred else None,
            preferred_output_variant=preferred.variant if preferred else None,
            preferred_extraction_method=preferred_extraction_method,
            extraction_risk=extraction_risk,
            page_count=page_count,
            outputs=outputs,
            operations=operations,
            quality_before={},
            quality_after={
                "average_ocr_readiness_score": average_ocr_score,
            },
            warnings=sorted(set(warnings)),
            average_quality_score_before=None,
            average_quality_score_after=average_ocr_score,
            ocr_readiness_score=average_ocr_score,
            is_ready_for_extraction=is_ready,
            preprocessing_version=settings.PREPROCESSING_VERSION,
        )

    def _average(self, values: list[float]) -> float | None:
        if not values:
            return None

        return round(sum(values) / len(values), 4)

    def _input_storage_key_from_path(self, source_path: str) -> str:
        marker = "/storage/"

        if marker in source_path:
            return source_path.split(marker, 1)[1]

        return source_path