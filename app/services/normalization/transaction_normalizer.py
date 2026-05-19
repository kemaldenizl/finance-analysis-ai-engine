import hashlib
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pycountry

from app.schemas.normalization import (
    NormalizedInstallment,
    NormalizedSummary,
    NormalizedTransaction,
)
from app.services.normalization.merchant_normalizer import MerchantNormalizer


class TransactionNormalizer:
    VALID_CURRENCIES = {
        currency.alpha_3
        for currency in pycountry.currencies
        if hasattr(currency, "alpha_3")
    }

    def __init__(self):
        self.merchant_normalizer = MerchantNormalizer()

    def normalize_many(
        self,
        transactions: list[dict[str, Any]],
    ) -> tuple[list[NormalizedTransaction], int]:
        normalized = [
            self.normalize_one(transaction)
            for transaction in transactions
        ]

        deduplicated, duplicate_removed_count = self.remove_duplicates(normalized)

        return deduplicated, duplicate_removed_count

    def normalize_one(self, transaction: dict[str, Any]) -> NormalizedTransaction:
        date = self._normalize_date(transaction.get("date"))
        amount = self._normalize_amount(
            transaction.get("price")
            if transaction.get("price") is not None
            else transaction.get("amount")
        )

        currency = self._normalize_currency(transaction.get("currency"))

        original_amount = self._normalize_amount(
            transaction.get("original_price")
            if transaction.get("original_price") is not None
            else transaction.get("original_amount")
        )

        original_currency = self._normalize_currency(
            transaction.get("original_currency")
        )

        description = self._normalize_description(transaction.get("description") or "")

        merchant = self.merchant_normalizer.normalize(description)

        direction = transaction.get("direction") or "unknown"

        if direction not in {"debit", "credit", "unknown"}:
            direction = "unknown"

        installment_raw = transaction.get("installment") or {}

        installment = NormalizedInstallment(
            current=self._safe_int(installment_raw.get("current")),
            total=self._safe_int(installment_raw.get("total")),
            raw=installment_raw.get("raw"),
            unit_amount=self._normalize_amount(installment_raw.get("unit_amount")),
            total_amount=self._normalize_amount(installment_raw.get("total_amount")),
        )

        warnings = self.validate(
            date=date,
            amount=amount,
            currency=currency,
            description=description,
            installment=installment,
        )

        validation_status = self._validation_status(warnings)

        source = {
            "page": transaction.get("page"),
            "raw_confidence": transaction.get("confidence"),
        }

        transaction_id = self.build_transaction_id(
            date=date,
            merchant=merchant.normalized,
            amount=amount,
            currency=currency,
            direction=direction,
        )

        confidence = float(transaction.get("confidence") or 0.0)

        return NormalizedTransaction(
            transaction_id=transaction_id,
            date=date or "",
            description=description,
            merchant=merchant,
            amount=amount or 0.0,
            currency=currency or "UNKNOWN",
            original_amount=original_amount,
            original_currency=original_currency,
            direction=direction,
            installment=installment,
            source=source,
            confidence=confidence,
            validation_status=validation_status,
            warnings=warnings,
        )

    def remove_duplicates(
        self,
        transactions: list[NormalizedTransaction],
    ) -> tuple[list[NormalizedTransaction], int]:
        best_by_key: dict[str, NormalizedTransaction] = {}

        for transaction in transactions:
            key = self._duplicate_key(transaction)

            existing = best_by_key.get(key)

            if existing is None:
                best_by_key[key] = transaction
                continue

            if transaction.confidence > existing.confidence:
                best_by_key[key] = transaction

        deduplicated = list(best_by_key.values())
        deduplicated.sort(key=lambda item: (item.date, item.description, item.amount))

        return deduplicated, len(transactions) - len(deduplicated)

    def build_summary(
        self,
        transactions: list[NormalizedTransaction],
        duplicate_removed_count: int,
        average_confidence: float | None,
        low_confidence_count: int,
    ) -> NormalizedSummary:
        total_debit = round(
            sum(item.amount for item in transactions if item.direction == "debit"),
            2,
        )

        total_credit = round(
            sum(item.amount for item in transactions if item.direction == "credit"),
            2,
        )

        currencies = sorted(
            {
                item.currency
                for item in transactions
                if item.currency and item.currency != "UNKNOWN"
            }
        )

        primary_currency = self._primary_currency(transactions)

        invalid_count = sum(
            1 for item in transactions if item.validation_status == "invalid"
        )
        warning_count = sum(
            1 for item in transactions if item.validation_status == "warning"
        )

        return NormalizedSummary(
            transaction_count=len(transactions),
            duplicate_removed_count=duplicate_removed_count,
            total_debit=total_debit,
            total_credit=total_credit,
            net_amount=round(total_debit - total_credit, 2),
            currencies=currencies,
            primary_currency=primary_currency,
            low_confidence_count=low_confidence_count,
            invalid_count=invalid_count,
            warning_count=warning_count,
            average_confidence=average_confidence,
        )

    def validate(
        self,
        date: str | None,
        amount: float | None,
        currency: str | None,
        description: str,
        installment: NormalizedInstallment,
    ) -> list[str]:
        warnings = []

        if not date:
            warnings.append("missing_or_invalid_date")

        if amount is None:
            warnings.append("missing_amount")
        elif amount <= 0:
            warnings.append("non_positive_amount")

        if not currency:
            warnings.append("missing_currency")
        elif currency not in self.VALID_CURRENCIES and currency != "UNKNOWN":
            warnings.append("invalid_currency")

        if not description or len(description) < 2:
            warnings.append("missing_or_short_description")

        if installment.current and installment.total:
            if installment.current > installment.total:
                warnings.append("invalid_installment_current_gt_total")

        return warnings

    def _validation_status(self, warnings: list[str]) -> str:
        hard_failures = {
            "missing_or_invalid_date",
            "missing_amount",
            "non_positive_amount",
        }

        if any(warning in hard_failures for warning in warnings):
            return "invalid"

        if warnings:
            return "warning"

        return "valid"

    def _normalize_date(self, value: Any) -> str | None:
        if not value:
            return None

        value = str(value).strip()

        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            pass

        for pattern in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, pattern).date().isoformat()
            except Exception:
                continue

        return None

    def _normalize_amount(self, value: Any) -> float | None:
        if value is None:
            return None

        try:
            decimal = Decimal(str(value)).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )

            return float(decimal)
        except Exception:
            return None

    def _normalize_currency(self, value: Any) -> str | None:
        if not value:
            return None

        normalized = str(value).upper().strip()

        if normalized == "TL":
            return "TRY"

        return normalized

    def _normalize_description(self, value: str) -> str:
        value = value.replace("\u00a0", " ")
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" -–—|:,;")

        return value

    def _safe_int(self, value: Any) -> int | None:
        if value in [None, ""]:
            return None

        try:
            return int(value)
        except Exception:
            return None

    def _duplicate_key(self, transaction: NormalizedTransaction) -> str:
        return "|".join(
            [
                transaction.date,
                transaction.merchant.normalized,
                f"{transaction.amount:.2f}",
                transaction.currency,
                transaction.direction,
            ]
        )

    def build_transaction_id(
        self,
        date: str | None,
        merchant: str,
        amount: float | None,
        currency: str | None,
        direction: str,
    ) -> str:
        raw = "|".join(
            [
                date or "",
                merchant or "",
                f"{amount or 0:.2f}",
                currency or "",
                direction or "",
            ]
        )

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

        return f"txn_{digest}"

    def _primary_currency(
        self,
        transactions: list[NormalizedTransaction],
    ) -> str | None:
        counts: dict[str, int] = {}

        for transaction in transactions:
            if transaction.currency and transaction.currency != "UNKNOWN":
                counts[transaction.currency] = counts.get(transaction.currency, 0) + 1

        if not counts:
            return None

        return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]