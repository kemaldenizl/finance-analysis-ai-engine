from typing import Any, Literal

from pydantic import BaseModel, Field


ExtractionRisk = Literal["low", "medium", "high"]
PreferredExtractionMethod = Literal[
    "native_pdf_text",
    "ocr_single_variant",
    "ocr_multi_variant",
    "manual_review",
]


class RegionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


class PreprocessedPageOutput(BaseModel):
    page_number: int

    variant: str = "ocr_optimized"
    purpose: str = "primary_ocr"
    is_preferred: bool = False

    storage_key: str
    storage_url: str

    width: int
    height: int

    region_bbox: RegionBox | None = None
    page_relevance_score: float | None = None
    region_relevance_score: float | None = None

    operations: list[str] = Field(default_factory=list)
    quality_before: dict[str, Any] = Field(default_factory=dict)
    quality_after: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PreprocessingResult(BaseModel):
    input_id: str
    source_kind: str
    status: str = "completed"

    output_type: str

    output_storage_key: str | None = None
    output_storage_url: str | None = None

    preferred_output_storage_key: str | None = None
    preferred_output_storage_url: str | None = None
    preferred_output_variant: str | None = None

    preferred_extraction_method: PreferredExtractionMethod = "ocr_multi_variant"
    extraction_risk: ExtractionRisk = "medium"

    page_count: int
    kept_page_count: int = 0
    skipped_page_count: int = 0

    outputs: list[PreprocessedPageOutput]

    operations: list[str] = Field(default_factory=list)
    quality_before: dict[str, Any] = Field(default_factory=dict)
    quality_after: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    average_quality_score_before: float | None = None
    average_quality_score_after: float | None = None
    ocr_readiness_score: float | None = None

    is_ready_for_extraction: bool = True
    preprocessing_version: str