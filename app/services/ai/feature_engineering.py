import hashlib

import numpy as np
import pandas as pd

from app.schemas.analyze import NormalizedTransactionInput


class FeatureEngineeringService:
    def build_dataframe(
        self,
        transactions: list[NormalizedTransactionInput],
        historical_transactions: list[NormalizedTransactionInput] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        current_df = self._build_single_dataframe(
            transactions=transactions,
            dataset_role="current",
        )

        history_df = self._build_single_dataframe(
            transactions=historical_transactions or [],
            dataset_role="history",
        )

        combined_df = pd.concat(
            [history_df, current_df],
            ignore_index=True,
        )

        if not combined_df.empty:
            combined_df = combined_df.drop_duplicates(
                subset=["transaction_id"],
                keep="last",
            ).reset_index(drop=True)

        return current_df, combined_df

    def _build_single_dataframe(
        self,
        transactions: list[NormalizedTransactionInput],
        dataset_role: str,
    ) -> pd.DataFrame:
        rows = []

        for index, transaction in enumerate(transactions):
            transaction_id = transaction.transaction_id or self._fallback_transaction_id(
                transaction=transaction,
                index=index,
                dataset_role=dataset_role,
            )

            merchant_name = self._merchant_name(transaction)

            rows.append(
                {
                    "transaction_id": transaction_id,
                    "dataset_role": dataset_role,
                    "date": transaction.date,
                    "description": transaction.description,
                    "merchant": merchant_name,
                    "amount": float(transaction.amount),
                    "currency": transaction.currency.upper(),
                    "original_amount": transaction.original_amount,
                    "original_currency": (
                        transaction.original_currency.upper()
                        if transaction.original_currency
                        else None
                    ),
                    "direction": transaction.direction,
                    "confidence": float(transaction.confidence),
                    "validation_status": transaction.validation_status,
                    "installment_current": transaction.installment.current,
                    "installment_total": transaction.installment.total,
                    "has_installment": transaction.installment.total is not None,
                }
            )

        if not rows:
            return self._empty_dataframe()

        dataframe = pd.DataFrame(rows)

        dataframe["date_dt"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

        dataframe["month_dt"] = dataframe["date_dt"].dt.to_period("M").dt.to_timestamp()
        dataframe["month"] = dataframe["month_dt"].dt.to_period("M").astype(str)
        dataframe["weekday"] = dataframe["date_dt"].dt.dayofweek
        dataframe["is_weekend"] = dataframe["weekday"].isin([5, 6])

        dataframe["spend_amount"] = np.where(
            dataframe["direction"] == "debit",
            dataframe["amount"],
            0.0,
        )

        dataframe["credit_amount"] = np.where(
            dataframe["direction"] == "credit",
            dataframe["amount"],
            0.0,
        )

        dataframe["signed_amount"] = np.where(
            dataframe["direction"] == "credit",
            -dataframe["amount"],
            dataframe["amount"],
        )

        dataframe["log_amount"] = np.log1p(dataframe["amount"])

        dataframe["is_foreign_currency"] = (
            dataframe["original_currency"].notna()
            & (dataframe["original_currency"] != dataframe["currency"])
        )

        dataframe["is_low_confidence"] = dataframe["confidence"] < 0.70
        dataframe["is_invalid"] = dataframe["validation_status"] == "invalid"

        return dataframe

    def debit_transactions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe.copy()

        return dataframe[dataframe["direction"] == "debit"].copy()

    def _merchant_name(self, transaction: NormalizedTransactionInput) -> str:
        if transaction.merchant:
            return (
                transaction.merchant.normalized
                or transaction.merchant.display_name
                or transaction.merchant.raw
                or transaction.description
            )

        return transaction.description

    def _fallback_transaction_id(
        self,
        transaction: NormalizedTransactionInput,
        index: int,
        dataset_role: str,
    ) -> str:
        raw = (
            f"{dataset_role}|{index}|{transaction.date}|{transaction.description}|"
            f"{transaction.amount}|{transaction.currency}|{transaction.direction}"
        )

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

        return f"txn_{digest}"

    def _empty_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "dataset_role",
                "date",
                "description",
                "merchant",
                "amount",
                "currency",
                "original_amount",
                "original_currency",
                "direction",
                "confidence",
                "validation_status",
                "installment_current",
                "installment_total",
                "has_installment",
                "date_dt",
                "month_dt",
                "month",
                "weekday",
                "is_weekend",
                "spend_amount",
                "credit_amount",
                "signed_amount",
                "log_amount",
                "is_foreign_currency",
                "is_low_confidence",
                "is_invalid",
            ]
        )