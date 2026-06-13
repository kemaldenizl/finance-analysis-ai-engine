from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Direction = Literal["debit", "credit", "unknown"]
ValidationStatus = Literal["valid", "warning", "invalid"]
RiskLevel = Literal["low", "medium", "high"]
AnalysisStatus = Literal["completed", "partial", "failed"]


class MerchantInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw: str | None = None
    normalized: str | None = None
    display_name: str | None = None
    confidence: float | None = None


class InstallmentInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    current: int | None = None
    total: int | None = None
    raw: str | None = None
    unit_amount: float | None = None
    total_amount: float | None = None


class NormalizedTransactionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transaction_id: str | None = None

    date: str
    description: str
    merchant: MerchantInput | None = None

    amount: float = Field(ge=0)
    currency: str

    original_amount: float | None = None
    original_currency: str | None = None

    direction: Direction = "debit"
    installment: InstallmentInput = Field(default_factory=InstallmentInput)

    source: dict[str, Any] = Field(default_factory=dict)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_status: ValidationStatus = "valid"
    warnings: list[str] = Field(default_factory=list)


class NormalizedSummaryInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transaction_count: int | None = None
    duplicate_removed_count: int | None = None

    total_debit: float | None = None
    total_credit: float | None = None
    net_amount: float | None = None

    currencies: list[str] = Field(default_factory=list)
    primary_currency: str | None = None

    low_confidence_count: int | None = None
    invalid_count: int | None = None
    warning_count: int | None = None
    average_confidence: float | None = None


class NormalizedDebugInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_kind: str | None = None
    document_currency: str | None = None
    extraction_method: str | None = None
    normalization_version: str | None = None


class NormalizedPipelineResultInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transactions: list[NormalizedTransactionInput] = Field(default_factory=list)
    summary: NormalizedSummaryInput = Field(default_factory=NormalizedSummaryInput)
    debug: NormalizedDebugInput | None = None

    @model_validator(mode="before")
    @classmethod
    def support_singular_transaction_key(cls, value: Any):
        if isinstance(value, dict):
            if "transactions" not in value and "transaction" in value:
                value["transactions"] = value["transaction"]

        return value


class PurchaseScenarioInput(BaseModel):
    amount: float = Field(gt=0)
    currency: str = "TRY"
    max_installment_months: int = Field(default=12, ge=1, le=36)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_id: str
    status: str = "completed"

    result: NormalizedPipelineResultInput
    scores: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    historical_transactions: list[NormalizedTransactionInput] = Field(default_factory=list)

    question: str | None = None
    purchase_scenario: PurchaseScenarioInput | None = None

    use_llm: bool = True


class CategorizedTransaction(BaseModel):
    transaction_id: str
    category: str
    subcategory: str | None = None
    confidence: float
    method: str
    merchant: str
    amount: float
    currency: str


class CategorySummary(BaseModel):
    category: str
    transaction_count: int
    total_amount: float
    share_of_spend: float


class CategorizationResult(BaseModel):
    transactions: list[CategorizedTransaction]
    summary: list[CategorySummary]
    uncategorized_count: int
    rule_assisted_count: int = 0
    embedding_assisted_count: int = 0
    llm_assisted_count: int = 0


class SpendingProfileResult(BaseModel):
    labels: list[str]
    primary_category: str | None = None
    primary_category_share: float | None = None
    installment_transaction_ratio: float = 0.0
    foreign_currency_transaction_ratio: float = 0.0
    average_transaction_amount: float = 0.0
    largest_transaction_amount: float = 0.0
    observations: list[str] = Field(default_factory=list)


class AnomalyItem(BaseModel):
    transaction_id: str
    anomaly_type: str
    severity: RiskLevel
    score: float
    message: str
    amount: float
    currency: str
    merchant: str


class AnomalyResult(BaseModel):
    anomaly_count: int
    method: str
    items: list[AnomalyItem]
    observations: list[str] = Field(default_factory=list)
    llm_explanation: str | None = None
    explanation_method: str = "deterministic_template_v1"


class SpendingForecastResult(BaseModel):
    status: str
    method: str
    historical_month_count: int = 0
    predicted_next_month_spend: float | None = None
    currency: str | None = None
    confidence: float | None = None
    observations: list[str] = Field(default_factory=list)


class InstallmentOption(BaseModel):
    months: int
    monthly_amount: float
    monthly_burden_ratio: float | None = None
    risk_level: RiskLevel


class InstallmentRecommendationResult(BaseModel):
    status: str
    requested_amount: float | None = None
    currency: str | None = None
    baseline_monthly_spend: float | None = None
    forecast_monthly_spend: float | None = None
    forecast_method: str | None = None
    recommended_months: int | None = None
    options: list[InstallmentOption] = Field(default_factory=list)
    explanation: str | None = None
    explanation_method: str = "deterministic_template_v1"
    warnings: list[str] = Field(default_factory=list)


class AssistantAnswer(BaseModel):
    question: str | None = None
    answer: str | None = None
    intent: str | None = None
    generation_method: str = "deterministic_template_v1"


class AiAnalysisResult(BaseModel):
    categorization: CategorizationResult
    spending_profile: SpendingProfileResult
    anomalies: AnomalyResult
    forecast: SpendingForecastResult
    installment_recommendation: InstallmentRecommendationResult
    assistant: AssistantAnswer
    executive_summary: list[str] = Field(default_factory=list)


class AiAnalysisQuality(BaseModel):
    source_overall_confidence: float | None = None
    usable_transaction_count: int
    low_confidence_transaction_count: int
    invalid_transaction_count: int
    analysis_confidence: float
    warnings: list[str] = Field(default_factory=list)


class AiEngineMetadata(BaseModel):
    analysis_version: str
    llm_enabled: bool
    llm_available: bool
    llm_model: str | None = None
    embedding_enabled: bool
    embedding_model: str | None = None
    anomaly_method: str
    forecast_method: str


class AnalyzeResponse(BaseModel):
    input_id: str
    analysis_id: str | None = None
    status: AnalysisStatus = "completed"
    result: AiAnalysisResult
    quality: AiAnalysisQuality
    engine: AiEngineMetadata
    warnings: list[str] = Field(default_factory=list)


class LlmCategoryDecision(BaseModel):
    merchant: str
    category: str
    subcategory: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class LlmCategoryBatchResponse(BaseModel):
    decisions: list[LlmCategoryDecision]


class LlmNarrativeResponse(BaseModel):
    text: str


class ChatRequest(BaseModel):
    analysis: AnalyzeResponse | None = None
    analysis_id: str | None = None
    question: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    generation_method: str