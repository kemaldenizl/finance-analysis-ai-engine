from typing import Any, Literal

from pydantic import BaseModel, Field


Direction = Literal["debit", "credit", "unknown"]


class NormalizedInstallment(BaseModel):
    current: int | None = None
    total: int | None = None
    raw: str | None = None
    unit_amount: float | None = None
    total_amount: float | None = None


class NormalizedMerchant(BaseModel):
    raw: str
    normalized: str
    display_name: str
    confidence: float = 0.0


class NormalizedTransaction(BaseModel):
    transaction_id: str

    date: str
    description: str

    merchant: NormalizedMerchant

    amount: float
    currency: str

    original_amount: float | None = None
    original_currency: str | None = None

    direction: Direction = "debit"

    installment: NormalizedInstallment = Field(default_factory=NormalizedInstallment)

    source: dict[str, Any] = Field(default_factory=dict)

    confidence: float = 0.0
    validation_status: Literal["valid", "warning", "invalid"] = "valid"
    warnings: list[str] = Field(default_factory=list)


class NormalizedSummary(BaseModel):
    transaction_count: int = 0
    duplicate_removed_count: int = 0

    total_debit: float = 0.0
    total_credit: float = 0.0
    net_amount: float = 0.0

    currencies: list[str] = Field(default_factory=list)
    primary_currency: str | None = None

    low_confidence_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0

    average_confidence: float | None = None


class NormalizedDebug(BaseModel):
    input_kind: str | None = None
    document_currency: str | None = None
    total_lines: int | None = None
    transaction_count: int | None = None
    low_confidence_count: int | None = None
    extraction_method: str | None = None
    normalization_version: str = "normalization-v1"


class RowScore(BaseModel):
    transaction_id: str
    score: float
    extraction_confidence: float
    field_confidence: float
    completeness_score: float
    validation_score: float
    flags: list[str] = Field(default_factory=list)


class ScoreSummary(BaseModel):
    overall_confidence: float | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    low_confidence_threshold: float = 0.70
    low_confidence_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0
    validation_passed: bool = True


class NormalizationScores(BaseModel):
    summary: ScoreSummary
    rows: list[RowScore]


class NormalizedResult(BaseModel):
    transactions: list[NormalizedTransaction]
    summary: NormalizedSummary
    debug: NormalizedDebug | None = None


class Stage4Response(BaseModel):
    input_id: str
    status: str = "completed"
    result: NormalizedResult
    scores: NormalizationScores
    warnings: list[str] = Field(default_factory=list)