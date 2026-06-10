from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.ai_analysis_record import AIAnalysisRecord
from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
)
from app.services.ai.analysis_service import AIAnalysisService
from app.services.ai.embedding_classifier import EmbeddingCategoryClassifier
from app.services.ai.llm_report import LLMReportService
from app.services.ai.providers.ollama_provider import OllamaProvider


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

    service = AIAnalysisService()

    return service.analyze(payload)


@router.post("/analyze-and-save", response_model=AnalyzeResponse)
def analyze_and_save(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    if not payload.result.transactions:
        raise HTTPException(
            status_code=422,
            detail="Analysis requires at least one normalized transaction.",
        )

    service = AIAnalysisService()
    response = service.analyze(payload)

    record = AIAnalysisRecord(
        input_id=payload.input_id,
        status=response.status,
        analysis_version=response.engine.analysis_version,
        llm_model=response.engine.llm_model,
        llm_available=str(response.engine.llm_available).lower(),
        analysis_confidence=response.quality.analysis_confidence,
        request_json=payload.model_dump(),
        result_json=response.model_dump(),
        warnings_json=response.warnings,
    )

    db.add(record)
    db.commit()

    response.analysis_id = record.id

    return response


@router.get("/analyses/{input_id}/latest")
def get_latest_ai_analysis(
    input_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(AIAnalysisRecord)
        .filter(AIAnalysisRecord.input_id == input_id)
        .order_by(AIAnalysisRecord.created_at.desc())
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="AI analysis result not found")

    return {
        "id": record.id,
        "input_id": record.input_id,
        "status": record.status,
        "analysis_version": record.analysis_version,
        "llm_model": record.llm_model,
        "llm_available": record.llm_available == "true",
        "analysis_confidence": record.analysis_confidence,
        "result": record.result_json,
        "warnings": record.warnings_json,
        "created_at": record.created_at,
    }


@router.post("/chat", response_model=ChatResponse)
def chat_about_analysis(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    analysis = payload.analysis

    if analysis is None and payload.analysis_id:
        record = (
            db.query(AIAnalysisRecord)
            .filter(AIAnalysisRecord.id == payload.analysis_id)
            .first()
        )

        if record is None:
            raise HTTPException(status_code=404, detail="AI analysis result not found")

        analysis = AnalyzeResponse.model_validate(record.result_json)

    if analysis is None:
        raise HTTPException(
            status_code=422,
            detail="Provide analysis or analysis_id for chat context.",
        )

    provider = OllamaProvider()
    report_service = LLMReportService(llm_provider=provider)

    assistant = report_service.answer_question(
        question=payload.question,
        categorization=analysis.result.categorization,
        profile=analysis.result.spending_profile,
        anomalies=analysis.result.anomalies,
        forecast=analysis.result.forecast,
        installment=analysis.result.installment_recommendation,
        use_llm=True,
    )

    return ChatResponse(
        answer=assistant.answer or "Cevap üretilemedi.",
        intent=assistant.intent or "general_statement_question",
        generation_method=assistant.generation_method,
    )


@router.get("/health")
def ai_analysis_health():
    provider = OllamaProvider()
    embedding_classifier = EmbeddingCategoryClassifier()

    return {
        "status": "healthy",
        "analysis_version": settings.AI_ANALYSIS_VERSION,
        "services": {
            "feature_engineering": "enabled",
            "categorization": "enabled",
            "embedding_classifier": (
                "available"
                if embedding_classifier.is_available()
                else "unavailable"
            ),
            "profiling": "enabled",
            "anomaly_detection": "enabled",
            "forecast_transformer": "enabled",
            "installment_recommendation": "enabled",
            "chat": "enabled",
        },
        "llm": {
            "enabled": settings.LLM_ENABLED,
            "provider": "ollama",
            "model": settings.LLM_MODEL,
            "available": provider.is_available(),
        },
    }