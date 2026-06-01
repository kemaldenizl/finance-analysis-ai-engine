import json
import re
from dataclasses import dataclass

import pandas as pd

from app.schemas.analyze import (
    CategorizationResult,
    CategorizedTransaction,
    CategorySummary,
    LlmCategoryBatchResponse,
)
from app.services.ai.category_taxonomy import CategoryTaxonomy, load_taxonomy
from app.services.ai.embedding_classifier import EmbeddingCategoryClassifier
from app.services.ai.providers.base import LLMProvider


@dataclass
class CategoryPrediction:
    category: str
    subcategory: str | None
    confidence: float
    method: str


class CategorizationService:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        taxonomy: CategoryTaxonomy | None = None,
    ):
        self.taxonomy = taxonomy or load_taxonomy()
        self.embedding_classifier = EmbeddingCategoryClassifier(self.taxonomy)
        self.llm_provider = llm_provider

    def categorize(
        self,
        dataframe: pd.DataFrame,
        use_llm: bool = True,
    ) -> CategorizationResult:
        if dataframe.empty:
            return CategorizationResult(
                transactions=[],
                summary=[],
                uncategorized_count=0,
            )

        predictions: dict[str, CategoryPrediction] = {}
        unresolved: list[dict] = []

        for _, row in dataframe.iterrows():
            transaction_id = str(row["transaction_id"])

            if str(row["direction"]) == "credit":
                predictions[transaction_id] = CategoryPrediction(
                    category="income_or_refund",
                    subcategory="credit_transaction",
                    confidence=0.97,
                    method="direction_rule_v2",
                )
                continue

            merchant_text = f"{row['merchant']} {row['description']}"

            rule_prediction = self._predict_by_rule(merchant_text)

            if rule_prediction:
                predictions[transaction_id] = rule_prediction
                continue

            embedding_prediction = self.embedding_classifier.predict(merchant_text)

            if embedding_prediction:
                predictions[transaction_id] = CategoryPrediction(
                    category=embedding_prediction.category,
                    subcategory=embedding_prediction.subcategory,
                    confidence=embedding_prediction.confidence,
                    method="embedding_similarity_v1",
                )
                continue

            unresolved.append(
                {
                    "transaction_id": transaction_id,
                    "merchant": str(row["merchant"]),
                    "description": str(row["description"]),
                }
            )

        if unresolved and use_llm and self.llm_provider and self.llm_provider.is_available():
            llm_predictions = self._predict_by_llm(unresolved)

            predictions.update(llm_predictions)

        for item in unresolved:
            transaction_id = item["transaction_id"]

            if transaction_id not in predictions:
                predictions[transaction_id] = CategoryPrediction(
                    category="other",
                    subcategory="uncategorized",
                    confidence=0.35,
                    method="unclassified_fallback_v2",
                )

        categorized_transactions = []

        for _, row in dataframe.iterrows():
            transaction_id = str(row["transaction_id"])
            prediction = predictions[transaction_id]

            categorized_transactions.append(
                CategorizedTransaction(
                    transaction_id=transaction_id,
                    category=prediction.category,
                    subcategory=prediction.subcategory,
                    confidence=prediction.confidence,
                    method=prediction.method,
                    merchant=str(row["merchant"]),
                    amount=round(float(row["amount"]), 2),
                    currency=str(row["currency"]),
                )
            )

        summary = self._build_summary(
            dataframe=dataframe,
            categorized_transactions=categorized_transactions,
        )

        return CategorizationResult(
            transactions=categorized_transactions,
            summary=summary,
            uncategorized_count=sum(
                1 for item in categorized_transactions if item.category == "other"
            ),
            rule_assisted_count=sum(
                1 for item in categorized_transactions if "rule" in item.method
            ),
            embedding_assisted_count=sum(
                1 for item in categorized_transactions if "embedding" in item.method
            ),
            llm_assisted_count=sum(
                1 for item in categorized_transactions if "llm" in item.method
            ),
        )

    def _predict_by_rule(self, merchant_text: str) -> CategoryPrediction | None:
        for candidate in self.taxonomy.candidates:
            if candidate.category == "other":
                continue

            for pattern in candidate.patterns:
                if re.search(pattern, merchant_text, flags=re.IGNORECASE):
                    return CategoryPrediction(
                        category=candidate.category,
                        subcategory=candidate.subcategory,
                        confidence=0.94,
                        method="merchant_rule_v2",
                    )

        return None

    def _predict_by_llm(
        self,
        unresolved: list[dict],
    ) -> dict[str, CategoryPrediction]:
        allowed = sorted(self.taxonomy.allowed_categories)

        system_prompt = (
            "Sen bir banka ekstresi merchant kategori sınıflandırıcısısın. "
            "Yalnızca verilen merchant ve açıklamayı kullan. "
            "Kategori icat etme. Emin değilsen other seç."
        )

        user_prompt = (
            f"İzin verilen kategoriler: {allowed}\n"
            "Aşağıdaki işlemleri kategorize et:\n"
            f"{json.dumps(unresolved, ensure_ascii=False)}"
        )

        response = self.llm_provider.generate_structured(
            response_model=LlmCategoryBatchResponse,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if response is None:
            return {}

        by_merchant = {
            item["merchant"]: item["transaction_id"]
            for item in unresolved
        }

        output = {}

        for decision in response.decisions:
            transaction_id = by_merchant.get(decision.merchant)

            if transaction_id is None:
                continue

            category = (
                decision.category
                if decision.category in self.taxonomy.allowed_categories
                else "other"
            )

            output[transaction_id] = CategoryPrediction(
                category=category,
                subcategory=decision.subcategory,
                confidence=min(decision.confidence, 0.82),
                method="qwen_llm_fallback_v1",
            )

        return output

    def _build_summary(
        self,
        dataframe: pd.DataFrame,
        categorized_transactions: list[CategorizedTransaction],
    ) -> list[CategorySummary]:
        category_by_transaction = {
            item.transaction_id: item.category
            for item in categorized_transactions
        }

        debit_rows = dataframe[dataframe["direction"] == "debit"].copy()

        if debit_rows.empty:
            return []

        debit_rows["category"] = debit_rows["transaction_id"].map(category_by_transaction)

        total_spend = float(debit_rows["amount"].sum())

        grouped = (
            debit_rows.groupby("category", dropna=False)
            .agg(
                transaction_count=("transaction_id", "count"),
                total_amount=("amount", "sum"),
            )
            .reset_index()
            .sort_values("total_amount", ascending=False)
        )

        return [
            CategorySummary(
                category=str(row["category"]),
                transaction_count=int(row["transaction_count"]),
                total_amount=round(float(row["total_amount"]), 2),
                share_of_spend=round(float(row["total_amount"]) / total_spend, 4)
                if total_spend
                else 0.0,
            )
            for _, row in grouped.iterrows()
        ]