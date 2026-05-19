from app.schemas.normalization import NormalizedTransaction, RowScore, ScoreSummary


class ScoringService:
    def __init__(self, low_confidence_threshold: float = 0.70):
        self.low_confidence_threshold = low_confidence_threshold

    def score_transaction(
        self,
        transaction: NormalizedTransaction,
        extraction_confidence: float,
    ) -> RowScore:
        field_confidence = self._field_confidence(transaction)
        completeness_score = self._completeness_score(transaction)
        validation_score = self._validation_score(transaction)

        score = (
            extraction_confidence * 0.40
            + field_confidence * 0.25
            + completeness_score * 0.20
            + validation_score * 0.15
        )

        flags = []

        if score < self.low_confidence_threshold:
            flags.append("low_confidence")

        if transaction.validation_status == "invalid":
            flags.append("validation_failed")

        if transaction.validation_status == "warning":
            flags.append("validation_warning")

        return RowScore(
            transaction_id=transaction.transaction_id,
            score=round(score, 4),
            extraction_confidence=round(extraction_confidence, 4),
            field_confidence=round(field_confidence, 4),
            completeness_score=round(completeness_score, 4),
            validation_score=round(validation_score, 4),
            flags=flags,
        )

    def summarize(self, row_scores: list[RowScore]) -> ScoreSummary:
        if not row_scores:
            return ScoreSummary(
                overall_confidence=None,
                min_confidence=None,
                max_confidence=None,
                low_confidence_threshold=self.low_confidence_threshold,
                low_confidence_count=0,
                invalid_count=0,
                warning_count=0,
                validation_passed=False,
            )

        scores = [row.score for row in row_scores]

        invalid_count = sum(
            1 for row in row_scores if "validation_failed" in row.flags
        )
        warning_count = sum(
            1 for row in row_scores if "validation_warning" in row.flags
        )
        low_confidence_count = sum(
            1 for row in row_scores if row.score < self.low_confidence_threshold
        )

        return ScoreSummary(
            overall_confidence=round(sum(scores) / len(scores), 4),
            min_confidence=round(min(scores), 4),
            max_confidence=round(max(scores), 4),
            low_confidence_threshold=self.low_confidence_threshold,
            low_confidence_count=low_confidence_count,
            invalid_count=invalid_count,
            warning_count=warning_count,
            validation_passed=invalid_count == 0,
        )

    def _field_confidence(self, transaction: NormalizedTransaction) -> float:
        score = 0.0

        if transaction.date:
            score += 0.25

        if transaction.amount is not None:
            score += 0.25

        if transaction.currency:
            score += 0.15

        if transaction.description:
            score += 0.15

        if transaction.merchant and transaction.merchant.confidence:
            score += min(transaction.merchant.confidence, 1.0) * 0.20

        return min(score, 1.0)

    def _completeness_score(self, transaction: NormalizedTransaction) -> float:
        required = [
            transaction.date,
            transaction.description,
            transaction.amount,
            transaction.currency,
            transaction.direction,
        ]

        present_count = sum(1 for item in required if item not in [None, ""])

        return present_count / len(required)

    def _validation_score(self, transaction: NormalizedTransaction) -> float:
        if transaction.validation_status == "valid":
            return 1.0

        if transaction.validation_status == "warning":
            return 0.65

        return 0.25