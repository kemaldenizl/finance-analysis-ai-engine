from fastapi import APIRouter, HTTPException

from app.schemas.analyze import (
    AiAnalysisQuality,
    AiAnalysisResult,
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.services.ai.anomaly_detection import AnomalyDetectionService
from app.services.ai.categorization import CategorizationService
from app.services.ai.feature_engineering import FeatureEngineeringService
from app.services.ai.forecast_installment import InstallmentRecommendationService
from app.services.ai.llm_report import LLMReportService
from app.services.ai.profiling import SpendingProfileService


router = APIRouter(prefix="/v1/ai", tags=["ai-analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_normalized_financial_data(
    payload: AnalyzeRequest,
) -> AnalyzeResponse:
    if not payload.result.transactions:
        raise HTTPException(
            status_code=422,
            detail="Analysis requires at least one normalized transaction.",
        )

    feature_service = FeatureEngineeringService()
    categorization_service = CategorizationService()
    profile_service = SpendingProfileService()
    anomaly_service = AnomalyDetectionService()
    installment_service = InstallmentRecommendationService()
    report_service = LLMReportService()

    dataframe = feature_service.build_dataframe(payload.result.transactions)

    categorization = categorization_service.categorize(dataframe)

    profile = profile_service.build_profile(
        dataframe=dataframe,
        categorization=categorization,
    )

    anomalies = anomaly_service.detect(
        dataframe=dataframe,
        categorization=categorization,
    )

    installment_recommendation = installment_service.recommend(
        dataframe=dataframe,
        scenario=payload.purchase_scenario,
    )

    assistant = report_service.answer_question(
        question=payload.question,
        categorization=categorization,
        profile=profile,
        anomalies=anomalies,
        installment=installment_recommendation,
    )

    executive_summary = report_service.build_executive_summary(
        categorization=categorization,
        profile=profile,
        anomalies=anomalies,
        installment=installment_recommendation,
    )

    quality = _build_quality(payload=payload, dataframe=dataframe)

    warnings = list(payload.warnings)

    if quality.low_confidence_transaction_count:
        warnings.append("analysis_contains_low_confidence_transactions")

    if quality.invalid_transaction_count:
        warnings.append("analysis_contains_invalid_transactions")

    if installment_recommendation.warnings:
        warnings.extend(installment_recommendation.warnings)

    result = AiAnalysisResult(
        categorization=categorization,
        spending_profile=profile,
        anomalies=anomalies,
        installment_recommendation=installment_recommendation,
        assistant=assistant,
        executive_summary=executive_summary,
    )

    response_status = "completed"

    if quality.invalid_transaction_count or quality.analysis_confidence < 0.55:
        response_status = "partial"

    return AnalyzeResponse(
        input_id=payload.input_id,
        status=response_status,
        result=result,
        quality=quality,
        warnings=sorted(set(warnings)),
    )


@router.get("/health")
def ai_analysis_health():
    return {
        "status": "healthy",
        "services": [
            "feature_engineering",
            "categorization",
            "profiling",
            "anomaly_detection",
            "installment_recommendation",
            "assistant_report",
        ],
        "llm_provider": "not_enabled_deterministic_template_v1",
    }


def _build_quality(payload: AnalyzeRequest, dataframe) -> AiAnalysisQuality:
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