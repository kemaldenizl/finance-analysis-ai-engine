from app.schemas.extraction import ExtractedTransaction
from app.services.extraction.ocr_text_extractor import OcrLine
from app.services.extraction.pdf_transaction_parser import PdfTransactionParser


class OcrTransactionParser(PdfTransactionParser):
    def parse_ocr_line_as_transaction(
        self,
        line: OcrLine,
        document_currency: str,
    ) -> ExtractedTransaction | None:
        base_line = type(
            "LineAdapter",
            (),
            {
                "page": line.page,
                "text": line.text,
                "source": "tesseract_ocr",
            },
        )()

        transaction = self.parse_line_as_transaction(
            line=base_line,
            document_currency=document_currency,
        )

        if transaction is None:
            return None

        confidence = self.calculate_ocr_confidence(
            base_confidence=transaction.confidence,
            ocr_confidence=line.ocr_confidence,
            word_count=line.word_count,
        )

        transaction.confidence = confidence
        transaction.page = line.page

        return transaction

    def calculate_ocr_confidence(
        self,
        base_confidence: float,
        ocr_confidence: float,
        word_count: int,
    ) -> float:
        score = 0.0
        score += base_confidence * 0.78
        score += max(0.0, min(ocr_confidence, 1.0)) * 0.18

        if word_count >= 3:
            score += 0.04

        return round(min(score, 0.98), 2)