"""Markdown output formatting and IPA glossary generation."""

from diktvox.models import TranscribedScore

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
    "œ": "rounded front vowel (German ö, short)",
    "yː": "rounded front vowel (German ü, long)",
    "ʏ": "rounded front vowel (German ü, short)",
    "aɪ̯": '"i" in "ride"',
    "aʊ̯": '"ow" in "cow"',
    "ɔʏ̯": '"oy" in "boy"',
    "ʔ": "glottal stop (brief pause)",
    "ː": "long vowel marker",
    "ˈ": "primary stress",
    "ˌ": "secondary stress",
    "̯": "non-syllabic marker",
    "θ": '"th" in "think"',
    "ð": '"th" in "this"',
    "ʒ": '"s" in "measure"',
    "ɡ": "hard g",
    "ts": '"ts" in "cats"',
    "pf": '"pf" in "Pferd"',
}


def _collect_symbols(score: TranscribedScore) -> set[str]:
    """Collect all unique IPA symbols used in the transcription."""
    symbols: set[str] = set()
    for section in score.sections:
        for vp in section.voice_parts:
            for char in vp.ipa:
                if char not in " \t\n" and not char.isascii():
                    symbols.add(char)
                # Also check for multi-char symbols
    # Check for known multi-char symbols
    all_ipa = " ".join(vp.ipa for s in score.sections for vp in s.voice_parts)
    for sym in _IPA_DESCRIPTIONS:
        if sym in all_ipa:
            symbols.add(sym)

    return symbols


def _build_glossary(score: TranscribedScore) -> str:
    """Build the IPA Symbol Reference table."""
    symbols = _collect_symbols(score)
    if not symbols:
        return ""

    # Find example words for each symbol
    all_ipa = " ".join(vp.ipa for s in score.sections for vp in s.voice_parts)
    all_text = " ".join(vp.text for s in score.sections for vp in s.voice_parts)

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

    for sym in sorted_symbols:
        desc = _IPA_DESCRIPTIONS[sym]
        # Find an example word containing this symbol
        example = _find_example(sym, score)
        lines.append(f"| {sym} | {desc} | {example} |")

    return "\n".join(lines)


def _find_example(symbol: str, score: TranscribedScore) -> str:
    """Find an example word from the score that contains the given IPA symbol."""
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

    # Sections
    for section in score.sections:
        lines.append(f"### Section: {section.name}")
        lines.append("")

        if section.all_parts_same and section.voice_parts:
            # All parts share the same text
            vp = section.voice_parts[0]
            lines.append("**All parts:**")
            lines.append(f"{vp.text}")
            lines.append(f"[{vp.ipa}]")
            lines.append("")
        else:
            for vp in section.voice_parts:
                lines.append(f"**{vp.name}:**")
                lines.append(f"{vp.text}")
                lines.append(f"[{vp.ipa}]")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Glossary
    glossary = _build_glossary(score)
    if glossary:
        lines.append(glossary)
        lines.append("")

    return "\n".join(lines)
