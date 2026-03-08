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

    # Select backend
    if backend == "espeak":
        from diktvox.ipa.espeak import espeak_transcribe
        transcribe_fn = espeak_transcribe
    elif backend == "llm":
        from diktvox.ipa.llm import llm_transcribe

        def transcribe_fn(text: str, lang: str) -> str:
            return llm_transcribe(text, lang=lang, model=model or "anthropic/claude-sonnet-4-20250514")
    else:
        raise ValueError(f"Unknown IPA backend: {backend}")

    sections: list[TranscribedSection] = []
    for section in score.sections:
        t_parts: list[TranscribedVoicePart] = []
        for vp in section.voice_parts:
            raw_ipa = transcribe_fn(vp.text, language)
            ipa = apply_rules(raw_ipa, vp.text, rules) if rules else raw_ipa
            t_parts.append(TranscribedVoicePart(name=vp.name, text=vp.text, ipa=ipa))
        sections.append(TranscribedSection(
            name=section.name,
            voice_parts=t_parts,
            all_parts_same=section.all_parts_same,
        ))

    return TranscribedScore(
        title=score.title,
        composer=score.composer,
        edition=score.edition,
        language=score.language,
        sections=sections,
    )
