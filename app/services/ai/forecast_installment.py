import pandas as pd

from app.schemas.analyze import (
    InstallmentOption,
    InstallmentRecommendationResult,
    PurchaseScenarioInput,
)


class InstallmentRecommendationService:
    SAFE_BURDEN_THRESHOLD = 0.15
    MEDIUM_BURDEN_THRESHOLD = 0.30

    def recommend(
        self,
        dataframe: pd.DataFrame,
        scenario: PurchaseScenarioInput | None,
    ) -> InstallmentRecommendationResult:
        if scenario is None:
            return InstallmentRecommendationResult(
                status="not_requested",
                explanation=(
                    "Taksit değerlendirmesi için analiz isteğinde "
                    "purchase_scenario gönderilmelidir."
                ),
            )

        debit_rows = dataframe[dataframe["direction"] == "debit"].copy()

        if debit_rows.empty:
            return InstallmentRecommendationResult(
                status="insufficient_data",
                requested_amount=scenario.amount,
                currency=scenario.currency,
                explanation="Mevcut harcama verisi bulunmadığı için taksit değerlendirmesi yapılamadı.",
                warnings=["no_debit_transactions_for_installment_analysis"],
            )

        same_currency = debit_rows[debit_rows["currency"] == scenario.currency.upper()].copy()

        if same_currency.empty:
            return InstallmentRecommendationResult(
                status="insufficient_data",
                requested_amount=scenario.amount,
                currency=scenario.currency.upper(),
                explanation=(
                    "Satın alma para birimiyle aynı para biriminde "
                    "karşılaştırılabilir harcama bulunamadı."
                ),
                warnings=["currency_mismatch_for_installment_analysis"],
            )

        monthly_spend = (
            same_currency.groupby("month")["amount"]
            .sum()
            .mean()
        )

        baseline_monthly_spend = round(float(monthly_spend), 2)

        options = []

        for months in range(1, scenario.max_installment_months + 1):
            monthly_amount = round(scenario.amount / months, 2)

            burden_ratio = (
                monthly_amount / baseline_monthly_spend
                if baseline_monthly_spend > 0
                else None
            )

            risk_level = self._risk_level(burden_ratio)

            options.append(
                InstallmentOption(
                    months=months,
                    monthly_amount=monthly_amount,
                    monthly_burden_ratio=round(burden_ratio, 4)
                    if burden_ratio is not None
                    else None,
                    risk_level=risk_level,
                )
            )

        recommended = next(
            (
                option
                for option in options
                if option.risk_level == "low"
            ),
            options[-1],
        )

        warnings = []

        if same_currency["month"].nunique() < 2:
            warnings.append("recommendation_based_on_single_statement_period")

        explanation = (
            f"{scenario.amount:.2f} {scenario.currency.upper()} tutarındaki alışveriş için "
            f"mevcut aylık harcama seviyesine göre {recommended.months} ay seçeneği "
            f"{recommended.risk_level} baskı seviyesinde değerlendirildi."
        )

        return InstallmentRecommendationResult(
            status="completed",
            requested_amount=round(scenario.amount, 2),
            currency=scenario.currency.upper(),
            baseline_monthly_spend=baseline_monthly_spend,
            recommended_months=recommended.months,
            options=options,
            explanation=explanation,
            warnings=warnings,
        )

    def _risk_level(self, burden_ratio: float | None) -> str:
        if burden_ratio is None:
            return "high"

        if burden_ratio <= self.SAFE_BURDEN_THRESHOLD:
            return "low"

        if burden_ratio <= self.MEDIUM_BURDEN_THRESHOLD:
            return "medium"

        return "high"