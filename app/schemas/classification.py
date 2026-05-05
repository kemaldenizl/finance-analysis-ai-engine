from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InputKind(str, Enum):
    REAL_PDF = "real_pdf"
    SCANNED_PDF = "scanned_pdf"
    HYBRID_PDF = "hybrid_pdf"
    SCREENSHOT = "screenshot"
    CAMERA_PHOTO = "camera_photo"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ClassificationResult(BaseModel):
    kind: InputKind
    confidence: float = Field(ge=0.0, le=1.0)

    needs_ocr: bool
    needs_preprocessing: bool

    routing_key: str

    features: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    model_version: str = "rules-v1"