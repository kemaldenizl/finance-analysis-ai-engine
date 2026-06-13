import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.core.config import settings
from app.schemas.analyze import (
    InstallmentOption,
    InstallmentRecommendationResult,
    LlmNarrativeResponse,
    PurchaseScenarioInput,
    SpendingForecastResult,
)
from app.services.ai.providers.base import LLMProvider


class MonthlySpendTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(x)
        encoded = self.encoder(encoded)
        last_token = encoded[:, -1, :]

        return self.output_head(last_token)


class ForecastInstallmentService:
    SAFE_BURDEN_THRESHOLD = 0.15
    MEDIUM_BURDEN_THRESHOLD = 0.30

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm_provider = llm_provider

    def forecast(
        self,
        reference_dataframe: pd.DataFrame,
        currency: str,
    ) -> SpendingForecastResult:
        monthly = self._monthly_spend_series(
            dataframe=reference_dataframe,
            currency=currency,
        )

        month_count = len(monthly)

        if month_count == 0:
            return SpendingForecastResult(
                status="insufficient_data",
                method="no_data",
                historical_month_count=0,
                currency=currency,
                observations=["Forecast için harcama verisi bulunamadı."],
            )

        weighted_average = self._weighted_moving_average(monthly)

        if month_count < settings.FORECAST_MIN_MONTHS_TRANSFORMER:
            predicted = self._clamp_to_history(weighted_average, monthly)

            return SpendingForecastResult(
                status="completed",
                method="weighted_moving_average_fallback_v2",
                historical_month_count=month_count,
                predicted_next_month_spend=round(predicted, 2),
                currency=currency,
                confidence=round(min(0.35 + month_count * 0.06, 0.65), 4),
                observations=[
                    "Yeterli dönem olmadığından ağırlıklı hareketli ortalama kullanıldı."
                ],
            )

        transformer_prediction = self._transformer_predict(monthly)

        blended = 0.6 * transformer_prediction + 0.4 * weighted_average
        prediction = self._clamp_to_history(blended, monthly)

        return SpendingForecastResult(
            status="completed",
            method="transformer_moving_average_ensemble_v1",
            historical_month_count=month_count,
            predicted_next_month_spend=round(prediction, 2),
            currency=currency,
            confidence=round(min(0.65 + month_count * 0.025, 0.90), 4),
            observations=[
                "Tahmin, Transformer Encoder ile ağırlıklı hareketli ortalamanın "
                "harmanlanmasıyla üretildi."
            ],
        )

    def recommend(
        self,
        forecast: SpendingForecastResult,
        scenario: PurchaseScenarioInput | None,
        use_llm: bool = True,
    ) -> InstallmentRecommendationResult:
        if scenario is None:
            return InstallmentRecommendationResult(
                status="not_requested",
                explanation=(
                    "Taksit değerlendirmesi için purchase_scenario gönderilmelidir."
                ),
            )

        if (
            forecast.status != "completed"
            or forecast.predicted_next_month_spend is None
            or forecast.currency != scenario.currency.upper()
        ):
            return InstallmentRecommendationResult(
                status="insufficient_data",
                requested_amount=scenario.amount,
                currency=scenario.currency.upper(),
                forecast_method=forecast.method,
                explanation=(
                    "Satın alma senaryosu için karşılaştırılabilir harcama tahmini bulunamadı."
                ),
                warnings=["insufficient_forecast_for_installment_recommendation"],
            )

        baseline = forecast.predicted_next_month_spend

        options = []

        for months in range(1, scenario.max_installment_months + 1):
            monthly_amount = round(scenario.amount / months, 2)
            burden_ratio = monthly_amount / baseline if baseline > 0 else None

            options.append(
                InstallmentOption(
                    months=months,
                    monthly_amount=monthly_amount,
                    monthly_burden_ratio=round(burden_ratio, 4)
                    if burden_ratio is not None
                    else None,
                    risk_level=self._risk_level(burden_ratio),
                )
            )

        recommended = next(
            (option for option in options if option.risk_level == "low"),
            options[-1],
        )

        risk_text = {
            "low": "rahatça karşılayabileceğiniz",
            "medium": "orta düzeyde zorlayabilecek",
            "high": "bütçenizi yükleyebilecek",
        }.get(recommended.risk_level, recommended.risk_level)

        explanation = (
            f"{scenario.amount:,.2f} {scenario.currency.upper()} tutarındaki alışveriş için "
            f"{recommended.months} ay taksit, aylık {recommended.monthly_amount:,.2f} "
            f"{scenario.currency.upper()} ödemeyle tahmini harcama seviyenize göre "
            f"{risk_text} bir seçenek görünüyor."
        )

        explanation_method = "deterministic_template_v2"

        if use_llm and self.llm_provider and self.llm_provider.is_available():
            llm_explanation = self._llm_explain(
                scenario=scenario,
                forecast=forecast,
                recommended=recommended,
            )

            if llm_explanation:
                explanation = llm_explanation
                explanation_method = "qwen_llm_explanation_v1"

        return InstallmentRecommendationResult(
            status="completed",
            requested_amount=round(scenario.amount, 2),
            currency=scenario.currency.upper(),
            baseline_monthly_spend=baseline,
            forecast_monthly_spend=baseline,
            forecast_method=forecast.method,
            recommended_months=recommended.months,
            options=options,
            explanation=explanation,
            explanation_method=explanation_method,
            warnings=[
                "recommendation_is_spending_burden_estimate_not_credit_advice"
            ],
        )

    def _weighted_moving_average(self, monthly: pd.Series) -> float:
        window = min(settings.FORECAST_LOOKBACK_MONTHS + 3, len(monthly))
        recent = monthly.tail(window).to_numpy(dtype=float)

        if len(recent) == 0:
            return 0.0

        weights = np.arange(1, len(recent) + 1, dtype=float)

        return float(np.average(recent, weights=weights))

    def _clamp_to_history(self, value: float, monthly: pd.Series) -> float:
        values = monthly.to_numpy(dtype=float)

        if len(values) == 0:
            return max(value, 0.0)

        lower = max(0.0, float(values.min()) * 0.5)
        upper = float(values.max()) * 1.5

        if upper <= 0.0:
            return 0.0

        return float(min(max(value, lower), upper))

    def _monthly_spend_series(
        self,
        dataframe: pd.DataFrame,
        currency: str,
    ) -> pd.Series:
        if dataframe.empty:
            return pd.Series(dtype=float)

        debit_rows = dataframe[
            (dataframe["direction"] == "debit")
            & (dataframe["currency"] == currency.upper())
            & (~dataframe["is_invalid"])
        ].copy()

        if debit_rows.empty:
            return pd.Series(dtype=float)

        monthly = (
            debit_rows.groupby("month_dt")["amount"]
            .sum()
            .sort_index()
        )

        full_index = pd.date_range(
            start=monthly.index.min(),
            end=monthly.index.max(),
            freq="MS",
        )

        return monthly.reindex(full_index, fill_value=0.0)

    def _transformer_predict(self, monthly: pd.Series) -> float:
        self._set_seed(settings.FORECAST_RANDOM_SEED)

        values = monthly.to_numpy(dtype=np.float32)

        mean = float(values.mean())
        std = float(values.std()) or 1.0

        normalized = (values - mean) / std

        sequence_length = min(
            settings.FORECAST_LOOKBACK_MONTHS,
            len(normalized) - 1,
        )

        x_values = []
        y_values = []

        for index in range(len(normalized) - sequence_length):
            x_values.append(normalized[index : index + sequence_length])
            y_values.append(normalized[index + sequence_length])

        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(-1)

        model = MonthlySpendTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_function = nn.MSELoss()

        model.train()

        for _ in range(settings.FORECAST_TRAIN_EPOCHS):
            optimizer.zero_grad()

            prediction = model(x_tensor)
            loss = loss_function(prediction, y_tensor)

            loss.backward()
            optimizer.step()

        model.eval()

        last_sequence = torch.tensor(
            normalized[-sequence_length:],
            dtype=torch.float32,
        ).view(1, sequence_length, 1)

        with torch.no_grad():
            predicted_normalized = float(model(last_sequence).item())

        predicted = predicted_normalized * std + mean

        return round(max(predicted, 0.0), 2)

    def _risk_level(self, burden_ratio: float | None) -> str:
        if burden_ratio is None:
            return "high"

        if burden_ratio <= self.SAFE_BURDEN_THRESHOLD:
            return "low"

        if burden_ratio <= self.MEDIUM_BURDEN_THRESHOLD:
            return "medium"

        return "high"

    def _llm_explain(
        self,
        scenario: PurchaseScenarioInput,
        forecast: SpendingForecastResult,
        recommended: InstallmentOption,
    ) -> str | None:
        system_prompt = (
            "Sen kişisel finans açıklama asistanısın. "
            "Verilen hesaplanmış sayıları değiştirme. "
            "Yatırım, kredi veya kesin finansal tavsiye verme. "
            "Sadece kullanıcı dostu kısa açıklama üret."
        )

        user_prompt = (
            f"Ürün tutarı: {scenario.amount:.2f} {scenario.currency.upper()}\n"
            f"Tahmini aylık harcama: {forecast.predicted_next_month_spend:.2f} {forecast.currency}\n"
            f"Önerilen taksit: {recommended.months} ay\n"
            f"Aylık ödeme: {recommended.monthly_amount:.2f} {scenario.currency.upper()}\n"
            f"Risk seviyesi: {recommended.risk_level}"
        )

        response = self.llm_provider.generate_structured(
            response_model=LlmNarrativeResponse,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return response.text if response else None

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)