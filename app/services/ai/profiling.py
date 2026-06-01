import pandas as pd

from app.schemas.analyze import CategorizationResult, SpendingProfileResult


class SpendingProfileService:
    def build_profile(
        self,
        dataframe: pd.DataFrame,
        categorization: CategorizationResult,
    ) -> SpendingProfileResult:
        debit_rows = dataframe[dataframe["direction"] == "debit"].copy()

        if debit_rows.empty:
            return SpendingProfileResult(
                labels=["insufficient_spending_data"],
                observations=["Analiz için harcama işlemi bulunamadı."],
            )

        category_summary = categorization.summary

        primary_category = category_summary[0].category if category_summary else None
        primary_share = (
            category_summary[0].share_of_spend
            if category_summary
            else None
        )

        transaction_count = len(debit_rows)

        installment_ratio = round(
            float(debit_rows["has_installment"].sum()) / transaction_count,
            4,
        )

        foreign_currency_ratio = round(
            float(debit_rows["is_foreign_currency"].sum()) / transaction_count,
            4,
        )

        average_amount = round(float(debit_rows["amount"].mean()), 2)
        largest_amount = round(float(debit_rows["amount"].max()), 2)

        labels = []
        observations = []

        if primary_category and primary_share is not None and primary_share >= 0.40:
            labels.append(f"{primary_category}_heavy_spender")
            observations.append(
                f"Harcama tutarının %{round(primary_share * 100, 1)} bölümü "
                f"{primary_category} kategorisinde yoğunlaşıyor."
            )

        if installment_ratio >= 0.35:
            labels.append("installment_heavy_spender")
            observations.append(
                f"İşlemlerin %{round(installment_ratio * 100, 1)} bölümü taksitli."
            )

        if foreign_currency_ratio >= 0.20:
            labels.append("international_spender")
            observations.append(
                f"İşlemlerin %{round(foreign_currency_ratio * 100, 1)} bölümü "
                "yabancı para işlemi içeriyor."
            )

        if largest_amount >= average_amount * 3 and transaction_count >= 3:
            labels.append("large_purchase_sensitive")
            observations.append(
                "En yüksek harcama, ortalama işlem tutarının belirgin üzerinde."
            )

        if not labels:
            labels.append("balanced_spender")
            observations.append(
                "Belirgin tek bir harcama davranışı yoğunluğu tespit edilmedi."
            )

        return SpendingProfileResult(
            labels=labels,
            primary_category=primary_category,
            primary_category_share=primary_share,
            installment_transaction_ratio=installment_ratio,
            foreign_currency_transaction_ratio=foreign_currency_ratio,
            average_transaction_amount=average_amount,
            largest_transaction_amount=largest_amount,
            observations=observations,
        )