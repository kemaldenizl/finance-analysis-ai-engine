from app.schemas.extraction import (
    PdfExtractionDebug,
    PdfExtractionResult,
    PdfExtractionSummary,
)
from app.services.extraction.pdf_text_extractor import PdfTextExtractor
from app.services.extraction.pdf_transaction_parser import PdfTransactionParser


class PdfExtractionService:
    def __init__(self):
        self.text_extractor = PdfTextExtractor()
        self.transaction_parser = PdfTransactionParser()

    def extract(
        self,
        pdf_path: str,
        include_debug: bool = False,
    ) -> PdfExtractionResult:
        lines = self.text_extractor.extract_lines(pdf_path)
        candidate_lines = self.transaction_parser.build_candidate_lines(lines)

        document_currency = self.transaction_parser.infer_document_currency(lines)

        transactions = []
        parsed_source_lines = []
        rejected_candidate_lines = []

        for line in candidate_lines:
            transaction = self.transaction_parser.parse_line_as_transaction(
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
                            "source": line.source,
                        }
                    )

                continue

            if include_debug:
                if (
                    self.transaction_parser.find_date(line.text)
                    or self.transaction_parser.find_money_values(line.text)
                ):
                    rejected_candidate_lines.append(line.text)

        warnings = []

        if not lines:
            warnings.append("no_text_lines_extracted_from_pdf")

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

        summary = PdfExtractionSummary(
            transaction_count=len(transactions),
            low_confidence_count=sum(
                1 for transaction in transactions if transaction.confidence < 0.70
            ),
            average_confidence=average_confidence,
            total_debit=total_debit,
            total_credit=total_credit,
            document_currency=document_currency,
        )

        debug = None

        if include_debug:
            debug = PdfExtractionDebug(
                input_kind="native_pdf",
                document_currency=document_currency,
                total_lines=len(lines),
                candidate_line_count=len(candidate_lines),
                transaction_count=len(transactions),
                low_confidence_count=summary.low_confidence_count,
                rejected_candidate_lines=rejected_candidate_lines[:100],
                parsed_source_lines=parsed_source_lines,
            )

        return PdfExtractionResult(
            transactions=transactions,
            summary=summary,
            debug=debug,
            warnings=warnings,
        )

    def _average(self, values: list[float]) -> float | None:
        if not values:
            return None

        return round(sum(values) / len(values), 4)