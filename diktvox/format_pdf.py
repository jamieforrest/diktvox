"""PDF output generation from Markdown content."""


_CSS = """
@page {
    size: letter;
    margin: 2cm 2.5cm;
}
body {
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}
h1 {
    font-size: 20pt;
    margin-bottom: 0.2em;
}
h2 {
    font-size: 14pt;
    color: #555;
    margin-top: 0.2em;
}
h3 {
    font-size: 12pt;
    color: #333;
    margin-top: 1.2em;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 0.8em 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 0.5em;
    font-size: 10pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
}
th {
    background-color: #f5f5f5;
    font-weight: bold;
}
p strong {
    font-size: 10pt;
}
"""


def format_pdf(md_content: str) -> bytes:
    """Convert Markdown content to a styled PDF.

    Args:
        md_content: Markdown string (as produced by ``format_markdown``).

    Returns:
        PDF file content as bytes.

    Raises:
        ImportError: If *weasyprint* is not installed.
    """
    try:
        import markdown
        from weasyprint import HTML  # noqa: WPS433
    except ImportError:
        raise ImportError(
            "PDF output requires the 'weasyprint' and 'markdown' packages. "
            "Install them with:  pip install 'diktvox[pdf]'"
        )

    html_body = markdown.markdown(md_content, extensions=["tables"])
    html_doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        f"<style>{_CSS}</style>"
        f"</head><body>{html_body}</body></html>"
    )
    return HTML(string=html_doc).write_pdf()
