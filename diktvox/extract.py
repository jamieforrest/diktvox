"""LLM vision-based text extraction from choral score pages."""

import hashlib
import json
import pathlib
import time

import re

import click
import litellm

from diktvox.models import ScoreData, Section, VoicePart

_CACHE_DIR = pathlib.Path.home() / ".cache" / "diktvox"

_PAGE_SYSTEM_PROMPT = """\
You are an expert choral score reader. Given an image of one page from a choral score, extract any sung text visible on this page.

Instructions:
1. If this page contains title, composer, or edition information, include it.
2. Detect the language of the sung text.
3. Extract the sung text per voice part (Soprano, Alto, Tenor, Bass, or whatever parts are present). \
Reconstruct full words from syllable breaks (e.g., "Auf - er - steh'n" → "Aufersteh'n"). \
Preserve punctuation and apostrophes.
4. If this is a well-known work, cross-reference the text against the known source text rather than \
relying purely on visual OCR (choral score OCR is unreliable due to syllable breaks, hyphens, and \
interleaved notation). Fill in any gaps or correct OCR errors using the canonical text.
5. The section "name" field is ONLY for rehearsal numbers and tempo/expression markings that are \
explicitly printed in the score. Use just the number for rehearsal marks (e.g., "44" not \
"Rehearsal 44"). Combine a rehearsal number with its tempo marking when both appear together \
(e.g., "31 Langsam (Slow). Misterioso."). If no rehearsal number or tempo marking is printed \
at the start of a passage, the name MUST be empty. \
NEVER invent descriptive labels — no "Section 1", "Continuing Text", "Opening Section", \
"Soloists", "Upper Voices", "Lower Voices", or any other name you made up. Only use text that \
is visually printed in the score as a rehearsal mark or tempo/expression indication.
6. Where voice parts share identical text in a section, set all_parts_same to true and provide \
the text under a single voice part named "All". Where parts diverge, group parts that share \
identical text together (e.g., "Soprano, Alto" as the name) and list differing groups separately. \
Distinguish between solo passages and tutti entrances — if only one voice sings while others rest, \
label it as that voice (e.g., "Alto") not "All".
7. If a section continues from a previous page, name it with a "(continued)" suffix \
appended to the original section name (e.g., "31 (continued)"). If the original section had \
an empty name, use just "(continued)".
8. Continuous phrases that span a system break on the same page should stay together in one section, \
not be split into separate sections.
9. If the page has no sung text (e.g., purely instrumental), return an empty sections list.
10. Flag any text you are uncertain about with [?] rather than guessing.
11. Ensure text is complete — do not truncate passages. If a passage continues beyond the visible \
page, include all text that IS visible.

Respond with valid JSON only, using this schema:
{
  "title": "string or empty",
  "composer": "string or empty",
  "edition": "string or empty",
  "language": "ISO 639-1 code or empty",
  "sections": [
    {
      "name": "rehearsal number and/or tempo marking, or empty string",
      "all_parts_same": true/false,
      "voice_parts": [
        {"name": "Soprano|Alto|Tenor|Bass|All|Soprano, Alto|etc.", "text": "full reconstructed text"}
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
        name = _normalize_section_name(s.get("name", ""))
        sections.append(Section(
            name=name,
            voice_parts=parts,
            all_parts_same=s.get("all_parts_same", False),
            page_number=s.get("page_number"),
        ))
    return ScoreData(
        title=data.get("title", ""),
        composer=data.get("composer", ""),
        edition=data.get("edition", ""),
        language=data.get("language", "de"),
        sections=sections,
    )


# Patterns that indicate the LLM fabricated a section name rather than
# reading a real rehearsal mark or tempo indication from the score.
_FABRICATED_NAME_RE = re.compile(
    r"^(?:"
    r"(?:Section|Part|Segment|Movement|Passage|Block)\s*\d*"
    r"|Continuing\s+Text"
    r"|Opening\s+(?:Section|Chorale|Chorus)"
    r"|Closing\s+(?:Section|Chorale|Chorus)"
    r"|Soloists?"
    r"|Upper\s+Voices"
    r"|Lower\s+Voices"
    r"|Intro(?:duction)?"
    r"|Outro"
    r"|Interlude"
    r"|Bridge"
    r")$",
    re.IGNORECASE,
)


def _normalize_section_name(name: str) -> str:
    """Clean up a section name returned by the LLM.

    - Strips the "Rehearsal" prefix (e.g., "Rehearsal 44" → "44")
    - Blanks out fabricated descriptive labels
    """
    name = name.strip()
    # "Rehearsal 44" → "44"
    name = re.sub(r"^Rehearsal\s+", "", name, flags=re.IGNORECASE)
    # Blank out fabricated names
    if _FABRICATED_NAME_RE.match(name):
        return ""
    return name


def _stamp_page_numbers(page_result: dict, page_num: int) -> None:
    """Force-set page_number on all sections from a given page.

    This is more reliable than relying on the LLM to echo the page number
    back, since we already know which page we extracted from.
    """
    for section in page_result.get("sections", []):
        section["page_number"] = page_num


def _extract_all_json_objects(raw: str) -> list[dict]:
    """Extract all top-level JSON objects from a string.

    Handles responses where the model returns multiple ```json blocks
    or concatenated JSON objects.
    """
    results: list[dict] = []
    # Remove all markdown fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw)
    decoder = json.JSONDecoder()
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "{":
            try:
                obj, end = decoder.raw_decode(cleaned, i)
                results.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return results


def _parse_json(raw: str, page_num: int) -> dict:
    """Parse JSON from an LLM response, handling markdown fences and preamble.

    Some models wrap their JSON in ```json ... ``` fences or add preamble
    text even when response_format=json_object is set.
    """
    last_error: json.JSONDecodeError | None = None

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        last_error = e

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw, count=1)
    stripped = re.sub(r"\n?```\s*$", "", stripped, count=1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        last_error = e

    # Try to find and merge multiple JSON objects (model sometimes returns
    # several ```json blocks for a single page)
    objects = _extract_all_json_objects(raw)
    if objects:
        merged = objects[0]
        for obj in objects[1:]:
            merged.setdefault("sections", []).extend(obj.get("sections", []))
        return merged

    head = raw[:200] + ("..." if len(raw) > 200 else "")
    tail = ("..." + raw[-200:]) if len(raw) > 400 else ""
    raise click.ClickException(
        f"LLM returned invalid JSON for page {page_num}. "
        f"Try re-running with --no-cache or a different model.\n"
        f"Parse error: {last_error}\n"
        f"Response ({len(raw)} chars) started with: {head}"
        + (f"\nResponse ended with: {tail}" if tail else "")
    )


_MAX_RETRIES = 4
_BACKOFF_DELAYS = [2, 4, 8, 16]


def _extract_page(page_b64: str, page_num: int, total_pages: int, model: str, language: str) -> dict:
    """Extract text from a single page image, with retry on rate limits."""
    content = [
        {"type": "text", "text": f"Page {page_num} of {total_pages}:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
        {"type": "text", "text": f"Extract all sung text from this page. The expected language is {language}."},
    ]

    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
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

            raw = (choice.message.content or "").strip()
            if not raw:
                raise click.ClickException(
                    f"LLM returned empty response for page {page_num}. "
                    f"Try re-running with --no-cache or a different model."
                )

            data = _parse_json(raw, page_num)

            # Stamp page numbers and normalize section names
            _stamp_page_numbers(data, page_num)
            for section in data.get("sections", []):
                section["name"] = _normalize_section_name(section.get("name", ""))

            return data
        except click.ClickException:
            raise
        except litellm.exceptions.RateLimitError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_DELAYS[attempt]
                click.echo(
                    f"  Rate limited on page {page_num}. "
                    f"Retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})...",
                    err=True,
                )
                time.sleep(delay)
            else:
                raise click.ClickException(
                    f"Rate limited on page {page_num} after {_MAX_RETRIES} retries. "
                    f"Try again later or use a different model.\n"
                    f"Original error: {e}"
                )
        except (litellm.exceptions.APIError, litellm.exceptions.APIConnectionError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_DELAYS[attempt]
                click.echo(
                    f"  API error on page {page_num}. "
                    f"Retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})...",
                    err=True,
                )
                time.sleep(delay)
            else:
                raise click.ClickException(
                    f"API error on page {page_num} after {_MAX_RETRIES} retries: {e}"
                )

    raise click.ClickException(f"Failed to extract page {page_num} after {_MAX_RETRIES} retries: {last_exc}")


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
                "page_number": section.get("page_number"),
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
