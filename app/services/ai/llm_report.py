import json

from app.core.config import settings
from app.schemas.analyze import (
    AnomalyResult,
    AssistantAnswer,
    CategorizationResult,
    InstallmentRecommendationResult,
    LlmNarrativeResponse,
    SpendingForecastResult,
    SpendingProfileResult,
)
from app.services.ai.providers.base import LLMProvider


class LLMReportService:
    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm_provider = llm_provider

    def build_executive_summary(
        self,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        forecast: SpendingForecastResult,
        installment: InstallmentRecommendationResult,
        use_llm: bool = True,
    ) -> list[str]:
        deterministic = self._deterministic_summary(
            categorization=categorization,
            profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment=installment,
        )

        if not use_llm or not self.llm_provider or not self.llm_provider.is_available():
            return deterministic

        system_prompt = (
            "Sen bir banka ekstresi analiz raporu üreten asistansın. "
            "Sana verilen metrikleri değiştirme, yeni tutar veya bulgu icat etme. "
            "Yalnızca kısa ve kullanıcı dostu Türkçe özet üret."
        )

        user_prompt = (
            "Aşağıdaki doğrulanmış analiz özetini en fazla dört kısa maddelik "
            "bir finansal gözlem metnine dönüştür:\n"
            f"{json.dumps(deterministic, ensure_ascii=False)}"
        )

        response = self.llm_provider.generate_structured(
            response_model=LlmNarrativeResponse,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if response is None:
            return deterministic

        return [response.text]

    def answer_question(
        self,
        question: str | None,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        forecast: SpendingForecastResult,
        installment: InstallmentRecommendationResult,
        use_llm: bool = True,
    ) -> AssistantAnswer:
        if not question:
            return AssistantAnswer(
                question=None,
                answer=None,
                intent=None,
            )

        intent = self._detect_intent(question)

        deterministic_answer = self._deterministic_answer(
            question=question,
            intent=intent,
            categorization=categorization,
            profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment=installment,
        )

        if not use_llm or not self.llm_provider or not self.llm_provider.is_available():
            return AssistantAnswer(
                question=question,
                answer=deterministic_answer,
                intent=intent,
                generation_method="deterministic_template_v2",
            )

        compact_context = {
            "profile": profile.model_dump(),
            "anomaly_count": anomalies.anomaly_count,
            "top_anomalies": [
                item.model_dump()
                for item in anomalies.items[:3]
            ],
            "forecast": forecast.model_dump(),
            "installment": installment.model_dump(),
            "category_summary": [
                item.model_dump()
                for item in categorization.summary[:5]
            ],
        }

        system_prompt = (
            "Sen Bonus finans sohbet asistanısın. "
            "Sadece verilen analiz bağlamındaki bilgileri kullan. "
            "Yeni tutar, işlem veya anomali uydurma. "
            "Finansal karar tavsiyesi yerine açıklayıcı bilgi ver. "
            "Cevabı Türkçe ve kısa yaz."
        )

        user_prompt = (
            f"Kullanıcı sorusu: {question}\n"
            f"Intent: {intent}\n"
            f"Analiz bağlamı: {json.dumps(compact_context, ensure_ascii=False)}"
        )

        answer_text = self.llm_provider.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_predict=settings.LLM_CHAT_NUM_PREDICT,
        )

        answer_text = answer_text.strip() if answer_text else None

        return AssistantAnswer(
            question=question,
            answer=answer_text or deterministic_answer,
            intent=intent,
            generation_method=(
                "qwen_llm_context_answer_v1"
                if answer_text
                else "deterministic_template_v2"
            ),
        )

    def _deterministic_summary(
        self,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        forecast: SpendingForecastResult,
        installment: InstallmentRecommendationResult,
    ) -> list[str]:
        output = []

        if profile.primary_category and profile.primary_category_share is not None:
            output.append(
                f"En yoğun harcama kategorisi {profile.primary_category}; "
                f"payı %{round(profile.primary_category_share * 100, 1)}."
            )

        if profile.labels:
            output.append(
                "Tespit edilen profil: "
                + ", ".join(profile.labels)
                + "."
            )

        output.append(
            f"Anomali değerlendirmesinde {anomalies.anomaly_count} işlem işaretlendi."
        )

        if forecast.predicted_next_month_spend is not None:
            output.append(
                f"Bir sonraki dönem harcama tahmini "
                f"{forecast.predicted_next_month_spend:.2f} {forecast.currency}."
            )

        if installment.recommended_months:
            output.append(
                f"Satın alma senaryosu için {installment.recommended_months} ay "
                "taksit seçeneği değerlendirildi."
            )

        return output

    def _detect_intent(self, question: str) -> str:
        normalized = question.lower()

        if "kategori" in normalized or "en çok" in normalized:
            return "category_question"

        if "anomali" in normalized or "şüpheli" in normalized:
            return "anomaly_question"

        if "taksit" in normalized or "kaç ay" in normalized:
            return "installment_question"

        if "tahmin" in normalized or "gelecek ay" in normalized:
            return "forecast_question"

        return "general_statement_question"

    def _deterministic_answer(
        self,
        question: str,
        intent: str,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        forecast: SpendingForecastResult,
        installment: InstallmentRecommendationResult,
    ) -> str:
        if intent == "category_question" and profile.primary_category:
            return (
                f"En yoğun harcama kategoriniz {profile.primary_category}; "
                f"toplam harcama payı yaklaşık "
                f"%{round((profile.primary_category_share or 0) * 100, 1)}."
            )

        if intent == "anomaly_question":
            if anomalies.anomaly_count:
                return (
                    f"{anomalies.anomaly_count} işlem için inceleme sinyali bulundu. "
                    f"En yüksek sinyal: {anomalies.items[0].message}"
                )

            return "Belirgin bir anomali sinyali tespit edilmedi."

        if intent == "installment_question":
            if installment.recommended_months:
                return (
                    f"Verilen satın alma senaryosu için "
                    f"{installment.recommended_months} ay seçeneği değerlendirildi."
                )

            return "Taksit analizi için purchase_scenario bilgisi gönderilmelidir."

        if intent == "forecast_question":
            if forecast.predicted_next_month_spend is not None:
                return (
                    f"Bir sonraki dönem için tahmini harcama "
                    f"{forecast.predicted_next_month_spend:.2f} {forecast.currency}."
                )

            return "Tahmin üretmek için yeterli dönem verisi bulunamadı."

        return (
            "Ekstreniz için kategori dağılımı, profil, anomali, "
            "tahmin ve taksit değerlendirmesi hakkında soru sorabilirsiniz."
        )