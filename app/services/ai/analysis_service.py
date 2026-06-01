from app.core.config import settings
from app.schemas.analyze import (
    AiAnalysisQuality,
    AiAnalysisResult,
    AiEngineMetadata,
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.services.ai.anomaly_detection import AnomalyDetectionService
from app.services.ai.categorization import CategorizationService
from app.services.ai.embedding_classifier import EmbeddingCategoryClassifier
from app.services.ai.feature_engineering import FeatureEngineeringService
from app.services.ai.forecast_installment import ForecastInstallmentService
from app.services.ai.llm_report import LLMReportService
from app.services.ai.profiling import SpendingProfileService
from app.services.ai.providers.ollama_provider import OllamaProvider


class AIAnalysisService:
    def __init__(self):
        self.llm_provider = OllamaProvider()

        self.feature_service = FeatureEngineeringService()
        self.categorization_service = CategorizationService(
            llm_provider=self.llm_provider,
        )
        self.profile_service = SpendingProfileService()
        self.anomaly_service = AnomalyDetectionService(
            llm_provider=self.llm_provider,
        )
        self.forecast_service = ForecastInstallmentService(
            llm_provider=self.llm_provider,
        )
        self.report_service = LLMReportService(
            llm_provider=self.llm_provider,
        )

    def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        current_df, reference_df = self.feature_service.build_dataframe(
            transactions=payload.result.transactions,
            historical_transactions=payload.historical_transactions,
        )

        use_llm = payload.use_llm and settings.LLM_ENABLED
        llm_available = self.llm_provider.is_available() if use_llm else False

        categorization = self.categorization_service.categorize(
            dataframe=current_df,
            use_llm=use_llm,
        )

        profile = self.profile_service.build_profile(
            dataframe=current_df,
            categorization=categorization,
        )

        anomalies = self.anomaly_service.detect(
            current_dataframe=current_df,
            reference_dataframe=reference_df,
            categorization=categorization,
            use_llm=use_llm,
        )

        currency = (
            payload.result.summary.primary_currency
            or (
                str(current_df["currency"].mode().iloc[0])
                if not current_df.empty
                else "TRY"
            )
        )

        forecast = self.forecast_service.forecast(
            reference_dataframe=reference_df,
            currency=currency,
        )

        installment = self.forecast_service.recommend(
            forecast=forecast,
            scenario=payload.purchase_scenario,
            use_llm=use_llm,
        )

        assistant = self.report_service.answer_question(
            question=payload.question,
            categorization=categorization,
            profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment=installment,
            use_llm=use_llm,
        )

        executive_summary = self.report_service.build_executive_summary(
            categorization=categorization,
            profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment=installment,
            use_llm=use_llm,
        )

        quality = self._build_quality(payload=payload, dataframe=current_df)

        warnings = list(payload.warnings)

        if quality.low_confidence_transaction_count:
            warnings.append("analysis_contains_low_confidence_transactions")

        if quality.invalid_transaction_count:
            warnings.append("analysis_contains_invalid_transactions")

        warnings.extend(installment.warnings)

        if use_llm and not llm_available:
            warnings.append("llm_unavailable_deterministic_fallback_used")

        result = AiAnalysisResult(
            categorization=categorization,
            spending_profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment_recommendation=installment,
            assistant=assistant,
            executive_summary=executive_summary,
        )

        status = "completed"

        if quality.invalid_transaction_count or quality.analysis_confidence < 0.55:
            status = "partial"

        return AnalyzeResponse(
            input_id=payload.input_id,
            status=status,
            result=result,
            quality=quality,
            engine=AiEngineMetadata(
                analysis_version=settings.AI_ANALYSIS_VERSION,
                llm_enabled=use_llm,
                llm_available=llm_available,
                llm_model=settings.LLM_MODEL if use_llm else None,
                embedding_enabled=settings.EMBEDDING_ENABLED,
                embedding_model=(
                    settings.EMBEDDING_MODEL_NAME
                    if settings.EMBEDDING_ENABLED
                    else None
                ),
                anomaly_method=anomalies.method,
                forecast_method=forecast.method,
            ),
            warnings=sorted(set(warnings)),
        )

    def _build_quality(self, payload: AnalyzeRequest, dataframe) -> AiAnalysisQuality:
        low_confidence_count = int((dataframe["confidence"] < 0.70).sum())
        invalid_count = int((dataframe["validation_status"] == "invalid").sum())

        source_overall_confidence = None

        source_score_summary = payload.scores.get("summary") if payload.scores else None

        if isinstance(source_score_summary, dict):
            source_overall_confidence = source_score_summary.get("overall_confidence")

        if source_overall_confidence is None:
            source_overall_confidence = round(float(dataframe["confidence"].mean()), 4)

        penalty = 0.0

        if len(dataframe):
            penalty += (low_confidence_count / len(dataframe)) * 0.20
            penalty += (invalid_count / len(dataframe)) * 0.35

        analysis_confidence = round(
            max(0.0, min(1.0, float(source_overall_confidence) - penalty)),
            4,
        )

        warnings = []

        if low_confidence_count:
            warnings.append("low_confidence_source_rows_present")

        if invalid_count:
            warnings.append("invalid_source_rows_present")

        return AiAnalysisQuality(
            source_overall_confidence=source_overall_confidence,
            usable_transaction_count=len(dataframe) - invalid_count,
            low_confidence_transaction_count=low_confidence_count,
            invalid_transaction_count=invalid_count,
            analysis_confidence=analysis_confidence,
            warnings=warnings,
        )