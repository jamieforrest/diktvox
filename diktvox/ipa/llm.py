"""LLM-based IPA transcription backend (experimental)."""

import litellm

_SYSTEM_PROMPT = """\
You are an expert in IPA (International Phonetic Alphabet) transcription for sung text.

Given a text and a language, produce an accurate IPA transcription suitable for classical singers.
Apply standard singing diction conventions:
- For German: use standard sung German conventions (rolled r at word start, vocalized r in \
final -er, ich-laut after front vowels, ach-laut after back vowels, etc.)
- Preserve word boundaries with spaces.
- Output ONLY the IPA transcription, nothing else. No brackets, no explanations.
"""


def llm_transcribe(text: str, *, lang: str, model: str) -> str:
    """Transcribe text to IPA using an LLM.

    Args:
        text: Input text to transcribe.
        lang: Language code.
        model: LiteLLM model identifier.

    Returns:
        IPA transcription string.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Language: {lang}\nText: {text}"},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()
