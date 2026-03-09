"""LLM-based IPA transcription backend (experimental)."""

import time

import click
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

_MAX_RETRIES = 4
_BACKOFF_DELAYS = [2, 4, 8, 16]


def llm_transcribe(text: str, *, lang: str, model: str) -> str:
    """Transcribe text to IPA using an LLM.

    Args:
        text: Input text to transcribe.
        lang: Language code.
        model: LiteLLM model identifier.

    Returns:
        IPA transcription string.

    Raises:
        click.ClickException: On persistent rate limiting or other API errors.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Language: {lang}\nText: {text}"},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except litellm.exceptions.RateLimitError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_DELAYS[attempt]
                click.echo(
                    f"  Rate limited by {model}. Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})...",
                    err=True,
                )
                time.sleep(delay)
            else:
                raise click.ClickException(
                    f"Rate limited by {model} after {_MAX_RETRIES} retries. "
                    f"Try again later or use a different model/backend.\n"
                    f"Original error: {e}"
                )
        except (litellm.exceptions.APIError, litellm.exceptions.APIConnectionError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_DELAYS[attempt]
                click.echo(
                    f"  API error from {model}. Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})...",
                    err=True,
                )
                time.sleep(delay)
            else:
                raise click.ClickException(
                    f"API error from {model} after {_MAX_RETRIES} retries: {e}"
                )

    raise click.ClickException(f"Failed to transcribe after {_MAX_RETRIES} retries: {last_exc}")
