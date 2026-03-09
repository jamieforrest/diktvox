"""PDF page rendering using pypdfium2."""

import base64
import pathlib

import pypdfium2 as pdfium


def render_pages(pdf_path: pathlib.Path, *, scale: float = 2.0) -> list[str]:
    """Render each page of a PDF to a base64-encoded PNG string.

    Args:
        pdf_path: Path to the PDF file.
        scale: Rendering scale factor (default 2.0 for good OCR quality).

    Returns:
        List of base64-encoded PNG strings, one per page.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages_b64: list[str] = []

    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        import io

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        pages_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return pages_b64
