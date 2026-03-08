"""LLM vision-based text extraction from choral score pages."""

import hashlib
import json
import pathlib

import litellm

from diktvox.models import ScoreData, Section, VoicePart

_CACHE_DIR = pathlib.Path.home() / ".cache" / "diktvox"

_SYSTEM_PROMPT = """\
You are an expert choral score reader. Given images of a choral score, extract the sung text.

Instructions:
1. Identify the piece: title, composer, edition if visible. If this is a well-known work, \
cross-reference the text against the known source text rather than relying purely on visual OCR \
(choral score OCR is unreliable due to syllable breaks, hyphens, and interleaved notation).
2. Detect the language. If it is not German, set language to the detected language code.
3. Extract the sung text per voice part (Soprano, Alto, Tenor, Bass, or whatever parts are present), \
organized by rehearsal number or section. Reconstruct full words from syllable breaks \
(e.g., "Auf - er - steh'n" → "Aufersteh'n"). Preserve punctuation and apostrophes.
4. Where voice parts share identical text in a section, set all_parts_same to true and provide \
the text under a single voice part named "All". Where parts diverge, list each part separately.

Respond with valid JSON only, using this schema:
{
  "title": "string",
  "composer": "string",
  "edition": "string or empty",
  "language": "ISO 639-1 code",
  "sections": [
    {
      "name": "section name or rehearsal number",
      "all_parts_same": true/false,
      "voice_parts": [
        {"name": "Soprano|Alto|Tenor|Bass|All", "text": "full reconstructed text"}
      ]
    }
  ]
}
"""


def _cache_key(pdf_path: pathlib.Path) -> str:
    """Generate a cache key from the PDF file content hash."""
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return h


def _load_cache(pdf_path: pathlib.Path) -> ScoreData | None:
    """Load cached extraction result if available."""
    cache_file = _CACHE_DIR / f"{_cache_key(pdf_path)}.json"
    if not cache_file.exists():
        return None
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    return _parse_response(data)


def _save_cache(pdf_path: pathlib.Path, data: dict) -> None:
    """Save extraction result to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{_cache_key(pdf_path)}.json"
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_response(data: dict) -> ScoreData:
    """Parse the LLM JSON response into ScoreData."""
    sections = []
    for s in data.get("sections", []):
        parts = [VoicePart(name=vp["name"], text=vp["text"]) for vp in s.get("voice_parts", [])]
        sections.append(Section(
            name=s.get("name", ""),
            voice_parts=parts,
            all_parts_same=s.get("all_parts_same", False),
        ))
    return ScoreData(
        title=data.get("title", ""),
        composer=data.get("composer", ""),
        edition=data.get("edition", ""),
        language=data.get("language", "de"),
        sections=sections,
    )


def extract_text(
    pages_b64: list[str],
    *,
    model: str,
    language: str,
    use_cache: bool = True,
    pdf_path: pathlib.Path | None = None,
) -> ScoreData:
    """Extract sung text from rendered score pages using an LLM with vision.

    Args:
        pages_b64: Base64-encoded PNG images of each page.
        model: LiteLLM model identifier.
        language: Expected language code.
        use_cache: Whether to use cached results.
        pdf_path: Original PDF path (used for cache key).

    Returns:
        ScoreData with extracted text organized by section and voice part.

    Raises:
        click.ClickException: If the detected language doesn't match the expected language.
    """
    if use_cache and pdf_path is not None:
        cached = _load_cache(pdf_path)
        if cached is not None:
            import click
            click.echo("  Using cached extraction result.", err=True)
            return cached

    # Build the message content with all page images
    content: list[dict] = []
    for i, page_b64 in enumerate(pages_b64):
        content.append({
            "type": "text",
            "text": f"Page {i + 1}:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page_b64}",
            },
        })

    content.append({
        "type": "text",
        "text": f"Extract all sung text from this choral score. The expected language is {language}.",
    })

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content
    data = json.loads(raw_text)

    # Validate language
    detected = data.get("language", language)
    if detected != language:
        import click
        raise click.ClickException(
            f"Detected language '{detected}' does not match expected '{language}'. "
            f"v1 of diktvox only supports German."
        )

    # Cache the result
    if use_cache and pdf_path is not None:
        _save_cache(pdf_path, data)

    return _parse_response(data)
