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
            "Teknik terimler yerine günlük Türkçe kullan. "
            "Yalnızca kısa, akıcı ve kullanıcı dostu bir özet üret."
        )

        facts = "\n".join(f"- {line}" for line in deterministic)

        user_prompt = (
            "Aşağıdaki doğrulanmış bulguları, kullanıcının kolayca anlayacağı "
            "2-4 cümlelik akıcı bir finansal özet paragrafına dönüştür. "
            "Sadece bu bulgulara sadık kal:\n"
            f"{facts}"
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

        context_facts = self._build_chat_context(
            categorization=categorization,
            profile=profile,
            anomalies=anomalies,
            forecast=forecast,
            installment=installment,
        )

        system_prompt = (
            "Sen Bonus finans sohbet asistanısın. "
            "Sadece verilen analiz bağlamındaki bilgileri kullan. "
            "Yeni tutar, işlem veya anomali uydurma. "
            "Finansal karar tavsiyesi yerine açıklayıcı bilgi ver. "
            "Cevabı Türkçe, kısa ve anlaşılır yaz."
        )

        user_prompt = (
            f"Kullanıcı sorusu: {question}\n"
            f"Konu: {intent}\n"
            "Analiz bağlamı (yalnızca bu gerçekleri kullan):\n"
            f"{context_facts}"
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

    def _build_chat_context(
        self,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        forecast: SpendingForecastResult,
        installment: InstallmentRecommendationResult,
    ) -> str:
        lines = []

        if profile.primary_category and profile.primary_category_share is not None:
            lines.append(
                f"- En yoğun harcama kategorisi: {profile.primary_category} "
                f"(toplam harcamanın %{round(profile.primary_category_share * 100, 1)}'i)."
            )

        if profile.labels:
            lines.append(f"- Harcama profili: {', '.join(profile.labels)}.")

        top_categories = [
            f"{item.category} ({round(item.share_of_spend * 100, 1)}%)"
            for item in categorization.summary[:3]
        ]

        if top_categories:
            lines.append(f"- En çok harcanan kategoriler: {', '.join(top_categories)}.")

        lines.append(f"- İşaretlenen anomali sayısı: {anomalies.anomaly_count}.")

        for item in anomalies.items[:3]:
            lines.append(
                f"  - {item.merchant}: {item.amount:.2f} {item.currency} "
                f"({item.severity} önem)."
            )

        if forecast.predicted_next_month_spend is not None:
            lines.append(
                f"- Gelecek dönem harcama tahmini: "
                f"{forecast.predicted_next_month_spend:.2f} {forecast.currency}."
            )

        if installment.recommended_months:
            lines.append(
                f"- Önerilen taksit sayısı: {installment.recommended_months} ay "
                f"(risk: {installment.options[installment.recommended_months - 1].risk_level})."
                if installment.options
                and len(installment.options) >= installment.recommended_months
                else f"- Önerilen taksit sayısı: {installment.recommended_months} ay."
            )

        return "\n".join(lines)

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
                f"Harcamalarınızın en büyük bölümü {profile.primary_category} "
                f"kategorisinde (%{round(profile.primary_category_share * 100, 1)})."
            )

        if profile.labels:
            output.append(
                "Harcama profiliniz: "
                + ", ".join(profile.labels)
                + "."
            )

        if anomalies.anomaly_count:
            output.append(
                f"İncelemeye değer {anomalies.anomaly_count} işlem dikkat çekiyor."
            )
        else:
            output.append("Dikkat çeken olağan dışı bir işlem görünmüyor.")

        if forecast.predicted_next_month_spend is not None:
            output.append(
                f"Gelecek dönem için tahmini harcamanız yaklaşık "
                f"{forecast.predicted_next_month_spend:,.2f} {forecast.currency}."
            )

        if installment.recommended_months:
            output.append(
                f"Planladığınız alışveriş için {installment.recommended_months} ay "
                "taksit uygun bir seçenek olarak öne çıkıyor."
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
                f"En çok harcama yaptığınız kategori {profile.primary_category}; "
                f"bu kategori toplam harcamanızın yaklaşık "
                f"%{round((profile.primary_category_share or 0) * 100, 1)}'ini oluşturuyor."
            )

        if intent == "anomaly_question":
            if anomalies.anomaly_count:
                return (
                    f"İncelemenizde fayda olan {anomalies.anomaly_count} işlem var. "
                    f"En dikkat çekeni: {anomalies.items[0].message}"
                )

            return "Olağan dışı görünen bir işlem tespit edilmedi."

        if intent == "installment_question":
            if installment.recommended_months:
                return (
                    f"Planladığınız alışveriş için "
                    f"{installment.recommended_months} ay taksit uygun görünüyor."
                )

            return (
                "Taksit önerisi sunabilmem için alışveriş tutarını "
                "(purchase_scenario) paylaşmanız gerekiyor."
            )

        if intent == "forecast_question":
            if forecast.predicted_next_month_spend is not None:
                return (
                    f"Gelecek dönem için tahmini harcamanız yaklaşık "
                    f"{forecast.predicted_next_month_spend:,.2f} {forecast.currency}."
                )

            return "Tahmin üretebilmem için yeterli geçmiş dönem verisi yok."

        return (
            "Size kategori dağılımınız, harcama profiliniz, dikkat çeken işlemler, "
            "harcama tahmininiz ve taksit seçenekleri hakkında yardımcı olabilirim."
        )