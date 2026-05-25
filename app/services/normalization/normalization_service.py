from typing import Any

from app.schemas.normalization import (
    NormalizationScores,
    NormalizedDebug,
    NormalizedResult,
    Stage4Response,
)
from app.services.normalization.scoring_service import ScoringService
from app.services.normalization.transaction_normalizer import TransactionNormalizer


class NormalizationService:
    def __init__(self):
        self.transaction_normalizer = TransactionNormalizer()
        self.scoring_service = ScoringService()

    def normalize_extraction_result(
        self,
        input_id: str,
        extraction_result: dict[str, Any],
        extraction_method: str,
    ) -> Stage4Response:
        raw_transactions = extraction_result.get("transactions") or []

        normalized_transactions, duplicate_removed_count = (
            self.transaction_normalizer.normalize_many(raw_transactions)
        )

        row_scores = []

        for transaction in normalized_transactions:
            row_score = self.scoring_service.score_transaction(
                transaction=transaction,
                extraction_confidence=transaction.confidence,
            )

            transaction.confidence = row_score.score
            row_scores.append(row_score)

        score_summary = self.scoring_service.summarize(row_scores)

        summary = self.transaction_normalizer.build_summary(
            transactions=normalized_transactions,
            duplicate_removed_count=duplicate_removed_count,
            average_confidence=score_summary.overall_confidence,
            low_confidence_count=score_summary.low_confidence_count,
        )

        raw_debug = extraction_result.get("debug") or {}

        debug = NormalizedDebug(
            input_kind=raw_debug.get("input_kind"),
            document_currency=raw_debug.get("document_currency"),
            total_lines=raw_debug.get("total_lines"),
            transaction_count=len(normalized_transactions),
            low_confidence_count=score_summary.low_confidence_count,
            extraction_method=extraction_method,
            normalization_version="normalization-v1",
        )

        warnings = list(extraction_result.get("warnings") or [])

        if duplicate_removed_count:
            warnings.append(f"duplicates_removed:{duplicate_removed_count}")

        result = NormalizedResult(
            transactions=normalized_transactions,
            summary=summary,
            debug=debug,
        )

        scores = NormalizationScores(
            summary=score_summary,
            rows=row_scores,
        )

        return Stage4Response(
            input_id=input_id,
            status="completed",
            result=result,
            scores=scores,
            warnings=warnings,
        )