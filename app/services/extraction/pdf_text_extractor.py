from dataclasses import dataclass

import fitz
import pdfplumber

from app.services.extraction.pdf_normalizers import remove_pdf_artifacts


@dataclass
class PdfTextLine:
    page: int
    text: str
    source: str = "pdfplumber"


class PdfTextExtractor:
    def extract_lines(self, pdf_path: str) -> list[PdfTextLine]:
        lines = self._extract_with_pdfplumber(pdf_path)

        if lines:
            return lines

        return self._extract_with_pymupdf(pdf_path)

    def _extract_with_pdfplumber(self, pdf_path: str) -> list[PdfTextLine]:
        lines: list[PdfTextLine] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(
                    x_tolerance=1,
                    y_tolerance=3,
                    layout=False,
                ) or ""

                for raw_line in text.splitlines():
                    line = remove_pdf_artifacts(raw_line)

                    if not line:
                        continue

                    lines.append(
                        PdfTextLine(
                            page=page_no,
                            text=line,
                            source="pdfplumber",
                        )
                    )

        return lines

    def _extract_with_pymupdf(self, pdf_path: str) -> list[PdfTextLine]:
        lines: list[PdfTextLine] = []

        document = fitz.open(pdf_path)

        for page_no, page in enumerate(document, start=1):
            text = page.get_text("text") or ""

            for raw_line in text.splitlines():
                line = remove_pdf_artifacts(raw_line)

                if not line:
                    continue

                lines.append(
                    PdfTextLine(
                        page=page_no,
                        text=line,
                        source="pymupdf",
                    )
                )

        return lines

    def get_page_count(self, pdf_path: str) -> int:
        document = fitz.open(pdf_path)

        return len(document)