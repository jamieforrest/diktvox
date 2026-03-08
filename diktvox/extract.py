"""LLM vision-based text extraction from choral score pages."""

import hashlib
import json
import pathlib

import click
import litellm

from diktvox.models import ScoreData, Section, VoicePart

_CACHE_DIR = pathlib.Path.home() / ".cache" / "diktvox"

_PAGE_SYSTEM_PROMPT = """\
You are an expert choral score reader. Given an image of one page from a choral score, extract any sung text visible on this page.

Instructions:
1. If this page contains title, composer, or edition information, include it.
2. Detect the language of the sung text.
3. Extract the sung text per voice part (Soprano, Alto, Tenor, Bass, or whatever parts are present), \
organized by rehearsal number or section. Reconstruct full words from syllable breaks \
(e.g., "Auf - er - steh'n" → "Aufersteh'n"). Preserve punctuation and apostrophes.
4. If this is a well-known work, cross-reference the text against the known source text rather than \
relying purely on visual OCR (choral score OCR is unreliable due to syllable breaks, hyphens, and \
interleaved notation).
5. Where voice parts share identical text in a section, set all_parts_same to true and provide \
the text under a single voice part named "All". Where parts diverge, list each part separately.
6. If a section continues from a previous page, name it with a "(continued)" suffix, \
e.g., "Verse 1 (continued)".
7. If the page has no sung text (e.g., purely instrumental), return an empty sections list.

Respond with valid JSON only, using this schema:
{
  "title": "string or empty",
  "composer": "string or empty",
  "edition": "string or empty",
  "language": "ISO 639-1 code or empty",
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


def _extract_page(page_b64: str, page_num: int, total_pages: int, model: str, language: str) -> dict:
    """Extract text from a single page image."""
    content = [
        {"type": "text", "text": f"Page {page_num} of {total_pages}:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
        {"type": "text", "text": f"Extract all sung text from this page. The expected language is {language}."},
    ]

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _PAGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=8192,
    )

    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise click.ClickException(
            f"LLM response truncated on page {page_num} (finish_reason={choice.finish_reason})."
        )

    try:
        return json.loads(choice.message.content)
    except json.JSONDecodeError as e:
        raise click.ClickException(
            f"LLM returned invalid JSON for page {page_num}: {e}. "
            f"Try re-running with --no-cache or a different model."
        )


def _merge_page_results(page_results: list[dict]) -> dict:
    """Merge per-page extraction results into a single ScoreData dict."""
    title = ""
    composer = ""
    edition = ""
    language = ""

    # Take metadata from the first page that provides it
    for pr in page_results:
        if not title and pr.get("title"):
            title = pr["title"]
        if not composer and pr.get("composer"):
            composer = pr["composer"]
        if not edition and pr.get("edition"):
            edition = pr["edition"]
        if not language and pr.get("language"):
            language = pr["language"]

    # Merge sections: combine "(continued)" sections with their parent
    merged_sections: list[dict] = []
    for pr in page_results:
        for section in pr.get("sections", []):
            name = section.get("name", "")
            continued = name.endswith("(continued)")
            base_name = name.removesuffix(" (continued)").removesuffix("(continued)").strip() if continued else name

            if continued and merged_sections:
                # Find the last section with a matching base name to append to
                target = None
                for ms in reversed(merged_sections):
                    if ms["name"] == base_name:
                        target = ms
                        break

                if target is not None:
                    # Append text to matching voice parts
                    for vp in section.get("voice_parts", []):
                        matched = False
                        for tvp in target["voice_parts"]:
                            if tvp["name"] == vp["name"]:
                                tvp["text"] = tvp["text"].rstrip() + " " + vp["text"].lstrip()
                                matched = True
                                break
                        if not matched:
                            target["voice_parts"].append(vp)
                    continue

            merged_sections.append({
                "name": base_name if continued else name,
                "all_parts_same": section.get("all_parts_same", False),
                "voice_parts": list(section.get("voice_parts", [])),
            })

    return {
        "title": title,
        "composer": composer,
        "edition": edition,
        "language": language or "de",
        "sections": merged_sections,
    }


def extract_text(
    pages_b64: list[str],
    *,
    model: str,
    language: str,
    use_cache: bool = True,
    pdf_path: pathlib.Path | None = None,
) -> ScoreData:
    """Extract sung text from rendered score pages using an LLM with vision.

    Processes each page individually and merges results to avoid output truncation.
    """
    if use_cache and pdf_path is not None:
        cached = _load_cache(pdf_path)
        if cached is not None:
            click.echo("  Using cached extraction result.", err=True)
            return cached

    total = len(pages_b64)
    page_results: list[dict] = []
    for i, page_b64 in enumerate(pages_b64):
        click.echo(f"  Processing page {i + 1}/{total}...", err=True)
        result = _extract_page(page_b64, i + 1, total, model, language)
        page_results.append(result)

    merged = _merge_page_results(page_results)

    # Validate language
    detected = merged.get("language", language)
    if detected != language:
        raise click.ClickException(
            f"Detected language '{detected}' does not match expected '{language}'. "
            f"v1 of diktvox only supports German."
        )

    # Cache the merged result
    if use_cache and pdf_path is not None:
        _save_cache(pdf_path, merged)

    return _parse_response(merged)
