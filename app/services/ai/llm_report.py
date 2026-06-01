from app.schemas.analyze import (
    AnomalyResult,
    AssistantAnswer,
    CategorizationResult,
    InstallmentRecommendationResult,
    SpendingProfileResult,
)


class LLMReportService:
    """
    V1:
        Deterministic template output üretir.

    Future:
        Buraya Qwen / Ollama / vLLM provider enjekte edilebilir.
        Sayısal analiz kararlarını LLM değil servisler üretmeye devam eder.
        LLM yalnızca açıklama ve sohbet cevabı oluşturur.
    """

    def build_executive_summary(
        self,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        installment: InstallmentRecommendationResult,
    ) -> list[str]:
        output = []

        if profile.primary_category and profile.primary_category_share is not None:
            output.append(
                f"En yoğun harcama kategorisi {profile.primary_category}; "
                f"toplam harcama payı %{round(profile.primary_category_share * 100, 1)}."
            )

        if profile.labels:
            output.append(
                "Tespit edilen harcama profilleri: "
                + ", ".join(profile.labels)
                + "."
            )

        if anomalies.anomaly_count:
            output.append(
                f"{anomalies.anomaly_count} işlem inceleme gerektirebilecek anomali sinyali taşıyor."
            )
        else:
            output.append("Belirgin bir anomali sinyali tespit edilmedi.")

        if installment.status == "completed" and installment.recommended_months:
            output.append(
                f"Verilen satın alma senaryosu için önerilen taksit değerlendirmesi: "
                f"{installment.recommended_months} ay."
            )

        if categorization.uncategorized_count:
            output.append(
                f"{categorization.uncategorized_count} işlem kategori eşleştirmesinde "
                "genel/diğer grubunda kaldı."
            )

        return output

    def answer_question(
        self,
        question: str | None,
        categorization: CategorizationResult,
        profile: SpendingProfileResult,
        anomalies: AnomalyResult,
        installment: InstallmentRecommendationResult,
    ) -> AssistantAnswer:
        if not question:
            return AssistantAnswer(
                question=None,
                answer=None,
                generation_method="deterministic_template_v1",
            )

        normalized_question = question.lower()

        if "en çok" in normalized_question and "harca" in normalized_question:
            if profile.primary_category:
                answer = (
                    f"En yoğun harcama kategoriniz {profile.primary_category}. "
                    f"Bu kategori toplam harcamanızın yaklaşık "
                    f"%{round((profile.primary_category_share or 0) * 100, 1)} bölümünü oluşturuyor."
                )
            else:
                answer = "Harcama kategorisi belirlemek için yeterli işlem bulunamadı."

        elif "anomali" in normalized_question or "şüpheli" in normalized_question:
            if anomalies.anomaly_count:
                answer = (
                    f"{anomalies.anomaly_count} işlem için inceleme sinyali tespit edildi. "
                    f"En yüksek sinyal: {anomalies.items[0].message}"
                )
            else:
                answer = "Belirgin bir anomali sinyali tespit edilmedi."

        elif "taksit" in normalized_question:
            if installment.status == "completed" and installment.recommended_months:
                answer = (
                    f"Verilen satın alma senaryosunda {installment.recommended_months} ay "
                    f"seçeneği {installment.options[installment.recommended_months - 1].risk_level} "
                    "baskı seviyesinde değerlendirildi."
                )
            else:
                answer = (
                    "Taksit değerlendirmesi yapabilmem için purchase_scenario içinde "
                    "ürün tutarı ve para birimi gönderilmelidir."
                )

        else:
            answer = (
                "Ekstreniz üzerinde kategori dağılımı, harcama profili, "
                "anomali sinyalleri ve taksit senaryoları hakkında soru sorabilirsiniz."
            )

        return AssistantAnswer(
            question=question,
            answer=answer,
            generation_method="deterministic_template_v1",
        )