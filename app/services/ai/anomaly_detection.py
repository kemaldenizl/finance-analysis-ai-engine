import numpy as np
import pandas as pd

from app.schemas.analyze import AnomalyItem, AnomalyResult, CategorizationResult


class AnomalyDetectionService:
    def detect(
        self,
        dataframe: pd.DataFrame,
        categorization: CategorizationResult,
    ) -> AnomalyResult:
        debit_rows = dataframe[dataframe["direction"] == "debit"].copy()

        if debit_rows.empty:
            return AnomalyResult(
                anomaly_count=0,
                method="robust_statistical_v1",
                items=[],
                observations=["Anomali analizi için harcama işlemi bulunamadı."],
            )

        category_map = {
            item.transaction_id: item.category
            for item in categorization.transactions
        }

        debit_rows["category"] = debit_rows["transaction_id"].map(category_map)

        median_amount = float(debit_rows["amount"].median())
        mad = float(np.median(np.abs(debit_rows["amount"] - median_amount)))

        anomaly_items = []

        for _, row in debit_rows.iterrows():
            flags = []
            score = 0.0

            amount = float(row["amount"])
            transaction_id = str(row["transaction_id"])
            merchant = str(row["merchant"])
            currency = str(row["currency"])

            if mad > 0:
                robust_z = abs(amount - median_amount) / (1.4826 * mad)

                if robust_z >= 3.5:
                    flags.append("unusually_high_amount")
                    score += min(0.70, robust_z / 10)
            elif len(debit_rows) >= 3 and amount > median_amount * 3:
                flags.append("unusually_high_amount")
                score += 0.55

            if bool(row["is_foreign_currency"]):
                flags.append("foreign_currency_transaction")
                score += 0.22

            if bool(row["has_installment"]) and amount >= median_amount * 2:
                flags.append("large_installment_transaction")
                score += 0.25

            if float(row["confidence"]) < 0.70:
                flags.append("low_source_confidence")
                score += 0.25

            if str(row["validation_status"]) != "valid":
                flags.append("source_validation_warning")
                score += 0.20

            score = round(min(score, 1.0), 4)

            if score < 0.22:
                continue

            severity = self._severity(score)
            message = self._message(flags=flags, merchant=merchant, amount=amount, currency=currency)

            anomaly_items.append(
                AnomalyItem(
                    transaction_id=transaction_id,
                    anomaly_type=",".join(flags),
                    severity=severity,
                    score=score,
                    message=message,
                    amount=round(amount, 2),
                    currency=currency,
                    merchant=merchant,
                )
            )

        anomaly_items.sort(key=lambda item: item.score, reverse=True)

        observations = []

        if not anomaly_items:
            observations.append("Belirgin bir anomali sinyali tespit edilmedi.")
        else:
            observations.append(
                f"{len(anomaly_items)} işlem için incelenmesi faydalı olabilecek sinyal bulundu."
            )

        if len(debit_rows) < 8:
            observations.append(
                "Tek ekstre veya az sayıda işlem ile anomali değerlendirmesi sınırlıdır."
            )

        return AnomalyResult(
            anomaly_count=len(anomaly_items),
            method="robust_statistical_v1",
            items=anomaly_items,
            observations=observations,
        )

    def _severity(self, score: float) -> str:
        if score >= 0.65:
            return "high"

        if score >= 0.40:
            return "medium"

        return "low"

    def _message(
        self,
        flags: list[str],
        merchant: str,
        amount: float,
        currency: str,
    ) -> str:
        readable = {
            "unusually_high_amount": "alışılmışın üzerinde tutar",
            "foreign_currency_transaction": "yabancı para işlemi",
            "large_installment_transaction": "yüksek tutarlı taksitli işlem",
            "low_source_confidence": "düşük extraction güveni",
            "source_validation_warning": "doğrulama uyarısı",
        }

        reasons = [
            readable[flag]
            for flag in flags
            if flag in readable
        ]

        reason_text = ", ".join(reasons)

        return f"{merchant} işleminde {reason_text} sinyali var: {amount:.2f} {currency}."