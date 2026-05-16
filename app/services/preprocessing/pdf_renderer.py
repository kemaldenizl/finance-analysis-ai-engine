import fitz
import numpy as np
from PIL import Image

from app.core.config import settings


class PDFRenderer:
    def open_document(self, pdf_path: str) -> fitz.Document:
        return fitz.open(pdf_path)

    def render_pages(self, pdf_path: str, dpi: int | None = None) -> list[Image.Image]:
        dpi = dpi or settings.PDF_RENDER_DPI

        doc = fitz.open(pdf_path)
        pages: list[Image.Image] = []

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )

            pages.append(image)

        return pages

    def render_page_to_numpy(
        self,
        pdf_path: str,
        page_index: int,
        dpi: int | None = None,
    ) -> np.ndarray:
        dpi = dpi or settings.PDF_RENDER_DPI

        doc = fitz.open(pdf_path)
        page = doc[page_index]

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return np.array(image)