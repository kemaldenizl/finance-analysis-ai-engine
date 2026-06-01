import re
from dataclasses import dataclass

import pandas as pd

from app.schemas.analyze import (
    CategorizationResult,
    CategorizedTransaction,
    CategorySummary,
)


@dataclass(frozen=True)
class CategoryRule:
    category: str
    subcategory: str
    patterns: tuple[str, ...]


DEFAULT_CATEGORY_RULES = [
    CategoryRule(
        category="travel",
        subcategory="flight_transport",
        patterns=(
            r"\bpegasus\b",
            r"\bthy\b",
            r"\bturkish airlines\b",
            r"\bwizz\b",
            r"\bobilet\b",
            r"\bflight\b",
            r"\bairlines?\b",
        ),
    ),
    CategoryRule(
        category="food",
        subcategory="delivery_restaurant",
        patterns=(
            r"\btrendyol yemek\b",
            r"\byemeksepeti\b",
            r"\bgetir yemek\b",
            r"\brestaurant\b",
            r"\bcafe\b",
            r"\bkahve\b",
        ),
    ),
    CategoryRule(
        category="entertainment",
        subcategory="gaming_digital",
        patterns=(
            r"\bplaystation\b",
            r"\bsteam\b",
            r"\bnetflix\b",
            r"\bspotify\b",
            r"\bblutv\b",
            r"\bdisney\b",
            r"\bsinema\b",
        ),
    ),
    CategoryRule(
        category="shopping",
        subcategory="electronics",
        patterns=(
            r"\bmedia markt\b",
            r"\bmediamarkt\b",
            r"\bteknosa\b",
            r"\bapple\b",
            r"\belectronic\b",
        ),
    ),
    CategoryRule(
        category="shopping",
        subcategory="marketplace_retail",
        patterns=(
            r"\bamazon\b",
            r"\bhepsiburada\b",
            r"\btrendyol\b",
            r"\bn11\b",
            r"\bbkmkitap\b",
            r"\bkitap\b",
        ),
    ),
    CategoryRule(
        category="accommodation",
        subcategory="hotel_stay",
        patterns=(
            r"\bhotel\b",
            r"\botel\b",
            r"\bbooking\b",
            r"\bairbnb\b",
            r"\bkonaklama\b",
        ),
    ),
    CategoryRule(
        category="utilities_tax",
        subcategory="tax_public_payment",
        patterns=(
            r"\bvergi\b",
            r"\bdaire(?:si)?\b",
            r"\bsgk\b",
            r"\bfatura\b",
            r"\belektrik\b",
            r"\bsu\b",
        ),
    ),
    CategoryRule(
        category="payment",
        subcategory="card_payment",
        patterns=(
            r"\bkart ödeme\b",
            r"\bkart odeme\b",
            r"\bödemeniz için teşekkür\b",
            r"\bodemeniz icin tesekkur\b",
        ),
    ),
]


class CategorizationService:
    def __init__(self, rules: list[CategoryRule] | None = None):
        self.rules = rules or DEFAULT_CATEGORY_RULES

    def categorize(self, dataframe: pd.DataFrame) -> CategorizationResult:
        if dataframe.empty:
            return CategorizationResult(
                transactions=[],
                summary=[],
                uncategorized_count=0,
            )

        categorized_transactions: list[CategorizedTransaction] = []

        for _, row in dataframe.iterrows():
            category, subcategory, confidence, method = self._predict_row(
                merchant=str(row["merchant"]),
                description=str(row["description"]),
                direction=str(row["direction"]),
            )

            categorized_transactions.append(
                CategorizedTransaction(
                    transaction_id=str(row["transaction_id"]),
                    category=category,
                    subcategory=subcategory,
                    confidence=confidence,
                    method=method,
                    merchant=str(row["merchant"]),
                    amount=round(float(row["amount"]), 2),
                    currency=str(row["currency"]),
                )
            )

        summary = self._build_summary(
            dataframe=dataframe,
            categorized_transactions=categorized_transactions,
        )

        uncategorized_count = sum(
            1 for item in categorized_transactions if item.category == "other"
        )

        return CategorizationResult(
            transactions=categorized_transactions,
            summary=summary,
            uncategorized_count=uncategorized_count,
        )

    def _predict_row(
        self,
        merchant: str,
        description: str,
        direction: str,
    ) -> tuple[str, str | None, float, str]:
        value = f"{merchant} {description}".lower()

        if direction == "credit":
            return "income_or_refund", "credit_transaction", 0.95, "direction_rule_v1"

        for rule in self.rules:
            for pattern in rule.patterns:
                if re.search(pattern, value, flags=re.IGNORECASE):
                    return (
                        rule.category,
                        rule.subcategory,
                        0.92,
                        "merchant_rule_v1",
                    )

        return "other", None, 0.45, "unclassified_fallback_v1"

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

        output = []

        for _, row in grouped.iterrows():
            amount = round(float(row["total_amount"]), 2)

            output.append(
                CategorySummary(
                    category=str(row["category"]),
                    transaction_count=int(row["transaction_count"]),
                    total_amount=amount,
                    share_of_spend=round(amount / total_spend, 4) if total_spend else 0.0,
                )
            )

        return output