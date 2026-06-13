import json

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.preprocessing import MinMaxScaler

from app.core.config import settings
from app.schemas.analyze import (
    AnomalyItem,
    AnomalyResult,
    CategorizationResult,
    LlmNarrativeResponse,
)
from app.services.ai.providers.base import LLMProvider


class AnomalyDetectionService:
    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm_provider = llm_provider

    def detect(
        self,
        current_dataframe: pd.DataFrame,
        reference_dataframe: pd.DataFrame,
        categorization: CategorizationResult,
        use_llm: bool = True,
    ) -> AnomalyResult:
        current_debit = current_dataframe[current_dataframe["direction"] == "debit"].copy()
        reference_debit = reference_dataframe[reference_dataframe["direction"] == "debit"].copy()

        if current_debit.empty:
            return AnomalyResult(
                anomaly_count=0,
                method="no_debit_data",
                items=[],
                observations=["Anomali analizi için harcama işlemi bulunamadı."],
            )

        category_map = {
            item.transaction_id: item.category
            for item in categorization.transactions
        }

        current_debit["category"] = current_debit["transaction_id"].map(category_map)

        if len(reference_debit) >= settings.ANOMALY_MIN_ROWS_FOR_PYOD:
            items = self._detect_with_pyod(
                current_debit=current_debit,
                reference_debit=reference_debit,
            )
            method = "pyod_ecod_v1"
        else:
            items = self._detect_with_robust_statistics(current_debit)
            method = "robust_statistical_fallback_v2"

        observations = []

        if items:
            observations.append(
                f"{len(items)} işlem inceleme gerektirebilecek anomali sinyali taşıyor."
            )
        else:
            observations.append("Belirgin bir anomali sinyali tespit edilmedi.")

        if len(reference_debit) < settings.ANOMALY_MIN_ROWS_FOR_PYOD:
            observations.append(
                "PyOD modeli için yeterli geçmiş işlem olmadığı için istatistiksel fallback kullanıldı."
            )

        llm_explanation = None
        explanation_method = "deterministic_template_v2"

        if items and use_llm and self.llm_provider and self.llm_provider.is_available():
            llm_explanation = self._build_llm_explanation(items)

            if llm_explanation:
                explanation_method = "qwen_llm_explanation_v1"

        return AnomalyResult(
            anomaly_count=len(items),
            method=method,
            items=items,
            observations=observations,
            llm_explanation=llm_explanation,
            explanation_method=explanation_method,
        )

    def _detect_with_pyod(
        self,
        current_debit: pd.DataFrame,
        reference_debit: pd.DataFrame,
    ) -> list[AnomalyItem]:
        reference_features = self._feature_matrix(reference_debit)
        current_features = self._feature_matrix(current_debit)

        scaler = MinMaxScaler()
        reference_scaled = scaler.fit_transform(reference_features)
        current_scaled = scaler.transform(current_features)

        detector = ECOD(
            contamination=min(
                max(settings.ANOMALY_CONTAMINATION, 0.01),
                0.45,
            )
        )
        detector.fit(reference_scaled)

        raw_scores = detector.decision_function(current_scaled)
        normalized_scores = self._normalize_scores(raw_scores)

        items = []

        for (_, row), score in zip(current_debit.iterrows(), normalized_scores):
            business_flags = self._business_flags(row)

            adjusted_score = min(score + self._flag_bonus(business_flags), 1.0)

            if adjusted_score < settings.ANOMALY_PYOD_SCORE_CUTOFF:
                continue

            items.append(
                self._build_item(
                    row=row,
                    score=round(float(adjusted_score), 4),
                    flags=business_flags or ["pyod_outlier_score"],
                )
            )

        return sorted(items, key=lambda item: item.score, reverse=True)

    def _detect_with_robust_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> list[AnomalyItem]:
        median_amount = float(dataframe["amount"].median())
        mad = float(np.median(np.abs(dataframe["amount"] - median_amount)))

        items = []

        for _, row in dataframe.iterrows():
            flags = self._business_flags(row)
            score = self._flag_bonus(flags)

            if mad > 0:
                robust_z = abs(float(row["amount"]) - median_amount) / (1.4826 * mad)

                if robust_z >= 3.5:
                    flags.append("unusually_high_amount")
                    score += min(0.70, robust_z / 10)

            elif len(dataframe) >= 3 and float(row["amount"]) > median_amount * 3:
                flags.append("unusually_high_amount")
                score += 0.55

            score = min(score, 1.0)

            if score < settings.ANOMALY_ROBUST_SCORE_CUTOFF:
                continue

            items.append(
                self._build_item(
                    row=row,
                    score=round(score, 4),
                    flags=flags,
                )
            )

        return sorted(items, key=lambda item: item.score, reverse=True)

    def _feature_matrix(self, dataframe: pd.DataFrame) -> np.ndarray:
        return dataframe[
            [
                "log_amount",
                "has_installment",
                "is_foreign_currency",
                "is_low_confidence",
                "is_weekend",
            ]
        ].astype(float).to_numpy()

    def _business_flags(self, row) -> list[str]:
        flags = []

        if bool(row["is_foreign_currency"]):
            flags.append("foreign_currency_transaction")

        if bool(row["has_installment"]):
            flags.append("installment_transaction")

        if float(row["confidence"]) < 0.70:
            flags.append("low_source_confidence")

        if str(row["validation_status"]) != "valid":
            flags.append("source_validation_warning")

        return flags

    def _flag_bonus(self, flags: list[str]) -> float:
        bonus_map = {
            "foreign_currency_transaction": 0.12,
            "installment_transaction": 0.05,
            "low_source_confidence": 0.18,
            "source_validation_warning": 0.18,
        }

        return sum(bonus_map.get(flag, 0.0) for flag in flags)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores

        minimum = float(scores.min())
        maximum = float(scores.max())

        if maximum == minimum:
            return np.zeros_like(scores, dtype=float)

        return (scores - minimum) / (maximum - minimum)

    def _build_item(self, row, score: float, flags: list[str]) -> AnomalyItem:
        return AnomalyItem(
            transaction_id=str(row["transaction_id"]),
            anomaly_type=",".join(sorted(set(flags))),
            severity=self._severity(score),
            score=score,
            message=self._deterministic_message(
                flags=flags,
                merchant=str(row["merchant"]),
                amount=float(row["amount"]),
                currency=str(row["currency"]),
            ),
            amount=round(float(row["amount"]), 2),
            currency=str(row["currency"]),
            merchant=str(row["merchant"]),
        )

    def _severity(self, score: float) -> str:
        if score >= 0.70:
            return "high"

        if score >= 0.45:
            return "medium"

        return "low"

    def _deterministic_message(
        self,
        flags: list[str],
        merchant: str,
        amount: float,
        currency: str,
    ) -> str:
        readable = {
            "unusually_high_amount": "Alışılmışın üzerinde tutar",
            "foreign_currency_transaction": "Yabancı para işlemi",
            "installment_transaction": "Taksitli işlem",
            "low_source_confidence": "Düşük veri güveni",
            "source_validation_warning": "Doğrulama uyarısı",
            "pyod_outlier_score": "İstatistiksel sapma",
        }

        reasons = [
            readable[flag]
            for flag in flags
            if flag in readable
        ]

        reason_text = ", ".join(reasons) if reasons else "Olağan dışı işlem"

        return (
            f"{merchant} ({amount:,.2f} {currency}) işlemi dikkat çekiyor: "
            f"{reason_text}."
        )

    def _build_llm_explanation(self, items: list[AnomalyItem]) -> str | None:
        payload = [
            {
                "merchant": item.merchant,
                "amount": item.amount,
                "currency": item.currency,
                "severity": item.severity,
                "flags": item.anomaly_type,
            }
            for item in items[:5]
        ]

        system_prompt = (
            "Sen bir kişisel finans analiz asistanısın. "
            "Verilen anomali skorlarını değiştirme, yeni anomali icat etme. "
            "Yalnızca kullanıcı dostu kısa bir Türkçe açıklama üret."
        )

        user_prompt = (
            "Aşağıdaki sistem tarafından tespit edilmiş işlem sinyallerini "
            "en fazla üç cümleyle açıkla:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        response = self.llm_provider.generate_structured(
            response_model=LlmNarrativeResponse,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return response.text if response else None