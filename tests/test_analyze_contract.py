import pytest

from app.core.config import settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.ai.analysis_service import AIAnalysisService


@pytest.fixture(autouse=True)
def disable_external_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "LLM_ENABLED", False)
    monkeypatch.setattr(settings, "EMBEDDING_ENABLED", False)
    yield


def _build_request() -> AnalyzeRequest:
    transactions = [
        {
            "date": "2025-01-05",
            "description": "MIGROS ALISVERIS",
            "amount": 450.0,
            "currency": "TRY",
            "direction": "debit",
            "confidence": 0.95,
            "validation_status": "valid",
        },
        {
            "date": "2025-01-09",
            "description": "SHELL AKARYAKIT",
            "amount": 1200.0,
            "currency": "TRY",
            "direction": "debit",
            "confidence": 0.93,
            "validation_status": "valid",
        },
        {
            "date": "2025-01-15",
            "description": "TURKCELL FATURA",
            "amount": 320.0,
            "currency": "TRY",
            "direction": "debit",
            "confidence": 0.9,
            "validation_status": "valid",
        },
        {
            "date": "2025-01-20",
            "description": "MAAS ODEMESI",
            "amount": 30000.0,
            "currency": "TRY",
            "direction": "credit",
            "confidence": 0.99,
            "validation_status": "valid",
        },
    ]

    return AnalyzeRequest.model_validate(
        {
            "input_id": "test-input-1",
            "result": {
                "transactions": transactions,
                "summary": {"primary_currency": "TRY"},
            },
            "use_llm": False,
        }
    )


def test_analyze_returns_valid_contract():
    response = AIAnalysisService().analyze(_build_request())

    assert isinstance(response, AnalyzeResponse)

    dumped = response.model_dump()
    revalidated = AnalyzeResponse.model_validate(dumped)

    assert revalidated.model_dump() == dumped


def test_analyze_core_fields_present():
    response = AIAnalysisService().analyze(_build_request())

    assert response.input_id == "test-input-1"
    assert response.status in {"completed", "partial"}
    assert response.result.categorization.transactions
    assert response.quality.usable_transaction_count == 4
    assert response.engine.llm_enabled is False


def test_taxonomy_rules_categorize_known_merchants():
    response = AIAnalysisService().analyze(_build_request())

    categories = {
        item.merchant: item.category
        for item in response.result.categorization.transactions
    }

    assert categories.get("MIGROS ALISVERIS") == "groceries"
    assert categories.get("SHELL AKARYAKIT") == "fuel"
    assert categories.get("TURKCELL FATURA") == "telecom"
