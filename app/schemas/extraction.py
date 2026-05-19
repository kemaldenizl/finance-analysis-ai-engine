from typing import Any, Literal

from pydantic import BaseModel, Field


Direction = Literal["debit", "credit", "unknown"]


class ExtractedInstallment(BaseModel):
    current: int | None = None
    total: int | None = None
    raw: str | None = None
    unit_amount: float | None = None
    total_amount: float | None = None


class ExtractedTransaction(BaseModel):
    date: str
    description: str

    price: float
    currency: str

    original_price: float | None = None
    original_currency: str | None = None

    installment: ExtractedInstallment = Field(default_factory=ExtractedInstallment)

    direction: Direction = "debit"
    confidence: float = 0.0

    page: int | None = None


class PdfExtractionDebug(BaseModel):
    input_kind: str = "native_pdf"
    document_currency: str | None = None
    total_lines: int = 0
    candidate_line_count: int = 0
    transaction_count: int = 0
    low_confidence_count: int = 0
    rejected_candidate_lines: list[str] = Field(default_factory=list)
    parsed_source_lines: list[dict[str, Any]] = Field(default_factory=list)


class PdfExtractionSummary(BaseModel):
    transaction_count: int = 0
    low_confidence_count: int = 0
    average_confidence: float | None = None
    total_debit: float | None = None
    total_credit: float | None = None
    document_currency: str | None = None


class PdfExtractionResult(BaseModel):
    transactions: list[ExtractedTransaction]
    summary: PdfExtractionSummary
    debug: PdfExtractionDebug | None = None
    warnings: list[str] = Field(default_factory=list)


class PdfExtractionApiResponse(BaseModel):
    input_id: str
    status: str

    stage1: dict[str, Any] = Field(default_factory=dict)
    stage2: dict[str, Any] = Field(default_factory=dict)

    stage3: dict[str, Any] = Field(default_factory=dict)

    result: PdfExtractionResult