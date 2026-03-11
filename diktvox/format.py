"""Markdown output formatting and IPA glossary generation."""

from collections import defaultdict

from diktvox.models import TranscribedScore, TranscribedSection

# Common IPA symbols with plain-English approximations
_IPA_DESCRIPTIONS: dict[str, str] = {
    "ʃ": '"sh" in "ship"',
    "ç": '"h" in "huge"',
    "x": '"ch" in Scottish "loch"',
    "ʁ": "uvular fricative (French r)",
    "ʀ": "uvular trill (rolled uvular r)",
    "r": "alveolar trill (rolled r)",
    "ɾ": "alveolar tap (flipped r)",
    "ŋ": '"ng" in "sing"',
    "ɛ": '"e" in "bed"',
    "ɪ": '"i" in "bit"',
    "ʊ": '"oo" in "book"',
    "ɔ": '"aw" in "law"',
    "ə": '"a" in "about" (schwa)',
    "ɐ": "near-open central vowel (vocalized r)",
    "øː": "rounded front vowel (German ö, long)",
    "ø": "rounded front vowel (German ö, short)",
    "œ": "open rounded front vowel (German ö, short)",
    "yː": "rounded front vowel (German ü, long)",
    "ʏ": "rounded front vowel (German ü, short)",
    "aɪ̯": '"i" in "ride"',
    "aʊ̯": '"ow" in "cow"',
    "ɔʏ̯": '"oy" in "boy"',
    "ʔ": "glottal stop (brief pause)",
    "ː": "long vowel marker",
    "ˈ": "primary stress",
    "ˌ": "secondary stress",
    "\u032f": "non-syllabic marker",
    "θ": '"th" in "think"',
    "ð": '"th" in "this"',
    "ʒ": '"s" in "measure"',
    "ɡ": "hard g",
    "ts": '"ts" in "cats"',
    "pf": '"pf" in "Pferd"',
    "tʃ": '"ch" in "church"',
    "dʒ": '"j" in "judge"',
    "ɲ": '"gn" in Italian "gnocchi"',
}


def _collect_symbols(score: TranscribedScore) -> set[str]:
    """Collect all unique IPA symbols used in the transcription."""
    symbols: set[str] = set()
    all_ipa = " ".join(vp.ipa for s in score.sections for vp in s.voice_parts)

    # Check multi-char symbols first, then mask them so single-char components
    # aren't falsely detected (e.g., ʃ inside tʃ, or ʒ inside dʒ).
    multi_char = sorted(
        (sym for sym in _IPA_DESCRIPTIONS if len(sym) > 1),
        key=lambda s: -len(s),
    )
    masked_ipa = all_ipa
    for sym in multi_char:
        if sym in masked_ipa:
            symbols.add(sym)
            masked_ipa = masked_ipa.replace(sym, " " * len(sym))

    # Now check single-char symbols against the masked string
    for sym in _IPA_DESCRIPTIONS:
        if len(sym) == 1 and sym in masked_ipa:
            symbols.add(sym)

    # Also collect any non-ASCII characters that might not be in our descriptions
    for char in masked_ipa:
        if char not in " \t\n" and not char.isascii():
            symbols.add(char)

    return symbols


def _find_example(symbol: str, score: TranscribedScore, used_words: set[str]) -> str:
    """Find an example word from the score that contains the given IPA symbol.

    Prefers words not already used as examples for other symbols to ensure
    diverse, illustrative examples across the glossary.
    """
    # First pass: find a word not yet used
    for section in score.sections:
        for vp in section.voice_parts:
            ipa_words = vp.ipa.split()
            text_words = vp.text.split()
            for i, ipa_word in enumerate(ipa_words):
                if symbol in ipa_word:
                    orig = text_words[i] if i < len(text_words) else ""
                    if orig and orig.lower() not in used_words:
                        used_words.add(orig.lower())
                        return f"{orig} [{ipa_word}]"

    # Second pass: allow reuse if no unique word found
    for section in score.sections:
        for vp in section.voice_parts:
            ipa_words = vp.ipa.split()
            text_words = vp.text.split()
            for i, ipa_word in enumerate(ipa_words):
                if symbol in ipa_word:
                    orig = text_words[i] if i < len(text_words) else ""
                    if orig:
                        return f"{orig} [{ipa_word}]"
                    return f"[{ipa_word}]"
    return ""


def _build_glossary(score: TranscribedScore) -> str:
    """Build the IPA Symbol Reference table."""
    symbols = _collect_symbols(score)
    if not symbols:
        return ""

    lines = [
        "## IPA Symbol Reference",
        "",
        "| Symbol | Sounds like | Example |",
        "|--------|------------|---------|",
    ]

    # Sort symbols for consistent output (multi-char first, then single char)
    sorted_symbols = sorted(
        (s for s in symbols if s in _IPA_DESCRIPTIONS),
        key=lambda s: (-len(s), s),
    )

    used_words: set[str] = set()
    for sym in sorted_symbols:
        desc = _IPA_DESCRIPTIONS[sym]
        example = _find_example(sym, score, used_words)
        lines.append(f"| {sym} | {desc} | {example} |")

    # Report any symbols in the output that aren't in the descriptions
    undocumented = symbols - set(_IPA_DESCRIPTIONS.keys())
    # Filter out combining characters and stress marks already covered
    undocumented = {s for s in undocumented if len(s) == 1 and s not in "ːˈˌ\u032f"}
    if undocumented:
        for sym in sorted(undocumented):
            lines.append(f"| {sym} | *(undocumented)* | |")

    return "\n".join(lines)


def _format_section_annotation(name: str) -> str:
    """Format a section name into a Markdown annotation line.

    Separates a leading rehearsal number (in brackets) from any trailing
    tempo / expression marking so the output reads naturally:
      ``**[31] Langsam (Slow). Misterioso.**``

    If the name is purely a number: ``**[31]**``
    If it has no leading number: ``**Langsam. Misterioso.**``
    """
    if not name:
        return ""

    # Try to split a leading integer rehearsal number from the rest
    parts = name.split(None, 1)  # split on first whitespace
    if parts[0].isdigit():
        num = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if rest:
            return f"**[{num}] {rest}**"
        return f"**[{num}]**"

    # No leading number — show the name as-is (no brackets)
    return f"**{name}**"


def _group_voice_parts(section: TranscribedSection) -> list[tuple[str, str, str]]:
    """Group voice parts that share identical text and IPA.

    Returns a list of (group_label, text, ipa) tuples.
    """
    if section.all_parts_same and section.voice_parts:
        vp = section.voice_parts[0]
        return [("All parts", vp.text, vp.ipa)]

    # Group by (text, ipa) content
    groups: dict[tuple[str, str], list[str]] = {}
    for vp in section.voice_parts:
        key = (vp.text, vp.ipa)
        if key not in groups:
            groups[key] = []
        groups[key].append(vp.name)

    result = []
    for (text, ipa), names in groups.items():
        if len(names) == len(section.voice_parts):
            label = "All parts"
        else:
            label = ", ".join(names)
        result.append((label, text, ipa))
    return result


def format_markdown(score: TranscribedScore) -> str:
    """Format a TranscribedScore as a Markdown companion document.

    Args:
        score: Transcribed score with IPA data.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = []

    # Header
    title = score.title or "Choral Score"
    lines.append(f"# {title} — IPA Companion")
    lines.append("")

    if score.composer or score.edition:
        parts = []
        if score.composer:
            parts.append(score.composer)
        if score.edition:
            parts.append(score.edition)
        lines.append(f"## {' / '.join(parts)}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Group sections by page number
    current_page = None
    for section in score.sections:
        page = section.page_number
        if page is not None and page != current_page:
            current_page = page
            lines.append(f"### Page {page}")
            lines.append("")

        # Section annotation (rehearsal number / tempo marking)
        annotation = _format_section_annotation(section.name)
        if annotation:
            lines.append(annotation)
            lines.append("")

        # Voice parts — grouped by identical content
        grouped = _group_voice_parts(section)
        for label, text, ipa in grouped:
            lines.append(f"**{label}:**")
            lines.append(f"{text}")
            lines.append(f"[{ipa}]")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Glossary
    glossary = _build_glossary(score)
    if glossary:
        lines.append(glossary)
        lines.append("")

    return "\n".join(lines)
