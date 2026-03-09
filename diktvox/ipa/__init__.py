"""IPA transcription backends and rules engine."""

import pathlib

from diktvox.ipa.rules import load_rules, apply_rules
from diktvox.models import ScoreData, TranscribedScore, TranscribedSection, TranscribedVoicePart

_DEFAULT_RULES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "rules"


def transcribe(
    score: ScoreData,
    *,
    backend: str = "espeak",
    language: str = "de",
    rules_path: pathlib.Path | None = None,
    model: str | None = None,
) -> TranscribedScore:
    """Transcribe all text in a ScoreData to IPA.

    Args:
        score: Extracted score data.
        backend: "espeak" or "llm".
        language: Language code.
        rules_path: Path to a custom YAML rules file. If None, uses the default.
        model: LiteLLM model string (required for llm backend).

    Returns:
        TranscribedScore with IPA transcriptions.
    """
    # Load rules
    if rules_path is None:
        default_file = _DEFAULT_RULES_DIR / f"{language}_standard.yaml"
        if default_file.exists():
            rules_path = default_file
    rules = load_rules(rules_path) if rules_path else None

    if backend == "espeak":
        return _transcribe_espeak(score, language, rules)
    elif backend == "llm":
        return _transcribe_llm(score, language, rules, model)
    else:
        raise ValueError(f"Unknown IPA backend: {backend}")


def _transcribe_espeak(score: ScoreData, language: str, rules) -> TranscribedScore:
    """Transcribe using espeak-ng, one call per voice part."""
    from diktvox.ipa.espeak import espeak_transcribe

    sections: list[TranscribedSection] = []
    for section in score.sections:
        t_parts: list[TranscribedVoicePart] = []
        for vp in section.voice_parts:
            raw_ipa = espeak_transcribe(vp.text, language)
            ipa = apply_rules(raw_ipa, vp.text, rules) if rules else raw_ipa
            t_parts.append(TranscribedVoicePart(name=vp.name, text=vp.text, ipa=ipa))
        sections.append(TranscribedSection(
            name=section.name,
            voice_parts=t_parts,
            all_parts_same=section.all_parts_same,
            page_number=section.page_number,
        ))

    return TranscribedScore(
        title=score.title,
        composer=score.composer,
        edition=score.edition,
        language=score.language,
        sections=sections,
    )


def _transcribe_llm(score: ScoreData, language: str, rules, model: str | None) -> TranscribedScore:
    """Transcribe using LLM, batching all texts into a single API call."""
    from diktvox.ipa.llm import llm_transcribe_batch

    resolved_model = model or "anthropic/claude-sonnet-4-20250514"

    # Collect all voice part texts into a flat list for batching
    all_texts: list[str] = []
    # Track (section_idx, vp_idx) for each entry so we can reassemble
    index_map: list[tuple[int, int]] = []
    for si, section in enumerate(score.sections):
        for vi, vp in enumerate(section.voice_parts):
            all_texts.append(vp.text)
            index_map.append((si, vi))

    # Single batched LLM call
    all_ipa = llm_transcribe_batch(all_texts, lang=language, model=resolved_model)

    # Apply rules and reassemble into sections
    # Build a lookup: (section_idx, vp_idx) -> ipa
    ipa_lookup: dict[tuple[int, int], str] = {}
    for idx, (si, vi) in enumerate(index_map):
        raw_ipa = all_ipa[idx]
        original_text = all_texts[idx]
        ipa = apply_rules(raw_ipa, original_text, rules) if rules else raw_ipa
        ipa_lookup[(si, vi)] = ipa

    sections: list[TranscribedSection] = []
    for si, section in enumerate(score.sections):
        t_parts: list[TranscribedVoicePart] = []
        for vi, vp in enumerate(section.voice_parts):
            t_parts.append(TranscribedVoicePart(
                name=vp.name, text=vp.text, ipa=ipa_lookup[(si, vi)],
            ))
        sections.append(TranscribedSection(
            name=section.name,
            voice_parts=t_parts,
            all_parts_same=section.all_parts_same,
            page_number=section.page_number,
        ))

    return TranscribedScore(
        title=score.title,
        composer=score.composer,
        edition=score.edition,
        language=score.language,
        sections=sections,
    )
