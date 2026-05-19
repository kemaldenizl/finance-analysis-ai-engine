from collections import Counter, defaultdict
from pathlib import Path

from app.schemas.extraction import (
    OcrExtractionDebug,
    OcrVariantDebug,
    PdfExtractionResult,
    PdfExtractionSummary,
)
from app.services.extraction.ocr_transaction_parser import OcrTransactionParser
from app.services.extraction.ocr_variant_selector import (
    OcrVariantInput,
    OcrVariantSelector,
)
from app.services.extraction.ocr_text_extractor import OcrLine


class ImageExtractionService:
    def __init__(self):
        self.variant_selector = OcrVariantSelector()
        self.transaction_parser = OcrTransactionParser()

    def extract(
        self,
        preprocessing_outputs: list[dict],
        include_debug: bool = False,
    ) -> PdfExtractionResult:
        page_groups = self._group_outputs_by_page(preprocessing_outputs)

        all_lines: list[OcrLine] = []
        variant_debug: list[OcrVariantDebug] = []

        for page_no, variants in page_groups.items():
            best, all_results = self.variant_selector.select_best_for_page(variants)

            all_lines.extend(best.lines)

            if include_debug:
                variant_debug.append(
                    OcrVariantDebug(
                        page=page_no,
                        selected_variant=best.variant,
                        selected_image=Path(best.image_path).name,
                        selected_psm=best.psm,
                        selected_score=best.final_score,
                        all_variants=[
                            self.variant_selector.debug_result(result)
                            for result in all_results
                        ],
                    )
                )

        document_currency = self.infer_document_currency(all_lines)

        transactions = []
        parsed_source_lines = []
        rejected_candidate_lines = []

        for line in all_lines:
            transaction = self.transaction_parser.parse_ocr_line_as_transaction(
                line=line,
                document_currency=document_currency,
            )

            if transaction:
                transactions.append(transaction)

                if include_debug:
                    parsed_source_lines.append(
                        {
                            "page": line.page,
                            "line": line.text,
                            "ocr_confidence": line.ocr_confidence,
                            "source_image": line.source_image,
                            "source_variant": line.source_variant,
                            "psm": line.psm,
                        }
                    )

                continue

            if include_debug:
                if (
                    self.transaction_parser.find_date(line.text)
                    or self.transaction_parser.find_money_values(line.text)
                ):
                    rejected_candidate_lines.append(
                        {
                            "page": line.page,
                            "source_image": line.source_image,
                            "source_variant": line.source_variant,
                            "text": line.text,
                            "ocr_confidence": line.ocr_confidence,
                        }
                    )

        warnings = []

        if not preprocessing_outputs:
            warnings.append("no_preprocessing_outputs_found")

        if not all_lines:
            warnings.append("no_ocr_lines_extracted")

        if not transactions:
            warnings.append("no_transactions_detected")

        average_confidence = self._average(
            [transaction.confidence for transaction in transactions]
        )

        total_debit = round(
            sum(
                transaction.price
                for transaction in transactions
                if transaction.direction == "debit"
            ),
            2,
        )

        total_credit = round(
            sum(
                transaction.price
                for transaction in transactions
                if transaction.direction == "credit"
            ),
            2,
        )

        low_confidence_count = sum(
            1 for transaction in transactions if transaction.confidence < 0.70
        )

        summary = PdfExtractionSummary(
            transaction_count=len(transactions),
            low_confidence_count=low_confidence_count,
            average_confidence=average_confidence,
            total_debit=total_debit,
            total_credit=total_credit,
            document_currency=document_currency,
        )

        debug = None

        if include_debug:
            debug = OcrExtractionDebug(
                input_kind="preprocessed_image_variants",
                document_currency=document_currency,
                total_lines=len(all_lines),
                candidate_line_count=sum(
                    1
                    for line in all_lines
                    if self.transaction_parser.find_date(line.text)
                    or self.transaction_parser.find_money_values(line.text)
                ),
                transaction_count=len(transactions),
                low_confidence_count=low_confidence_count,
                variant_selection=variant_debug,
                rejected_candidate_lines=rejected_candidate_lines[:150],
                parsed_source_lines=parsed_source_lines,
            )

        return PdfExtractionResult(
            transactions=transactions,
            summary=summary,
            debug=debug,
            warnings=warnings,
        )

    def _group_outputs_by_page(
        self,
        preprocessing_outputs: list[dict],
    ) -> dict[int, list[OcrVariantInput]]:
        groups: dict[int, list[OcrVariantInput]] = defaultdict(list)

        for output in preprocessing_outputs:
            storage_url = output.get("storage_url")

            if not storage_url:
                continue

            page = int(output.get("page_number") or 1)
            variant = output.get("variant") or "unknown"

            groups[page].append(
                OcrVariantInput(
                    page=page,
                    variant=variant,
                    storage_url=storage_url,
                    is_preferred=bool(output.get("is_preferred")),
                    purpose=output.get("purpose"),
                )
            )

        return dict(sorted(groups.items(), key=lambda item: item[0]))

    def infer_document_currency(self, lines: list[OcrLine]) -> str:
        currencies = []

        for line in lines:
            for money in self.transaction_parser.find_money_values(line.text):
                if money.get("currency"):
                    currencies.append(money["currency"])

        if not currencies:
            return "TRY"

        return Counter(currencies).most_common(1)[0][0]

    def _average(self, values: list[float]) -> float | None:
        if not values:
            return None

        return round(sum(values) / len(values), 4)