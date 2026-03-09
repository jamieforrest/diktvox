"""LLM-based IPA transcription backend."""

import json
import time

import click
import litellm

_SYSTEM_PROMPT = """\
You are an expert in IPA (International Phonetic Alphabet) transcription for sung text.

Given numbered text lines and a language, produce an accurate IPA transcription for each line.
Apply standard singing diction conventions:
- For German: use standard sung German conventions (rolled r at word start, vocalized r in \
final -er, ich-laut after front vowels, ach-laut after back vowels, etc.)
- Preserve word boundaries with spaces.

Respond with valid JSON only: an object with a "results" key containing an array of strings, \
one IPA transcription per input line, in the same order. Example:
{"results": ["ˈaʊfɛɐ̯ˌʃteːn", "ʃtaʊp"]}
"""

_MAX_RETRIES = 4
_BACKOFF_DELAYS = [2, 4, 8, 16]


def llm_transcribe_batch(texts: list[str], *, lang: str, model: str) -> list[str]:
    """Transcribe multiple texts to IPA in a single LLM call.

    Args:
        texts: List of input texts to transcribe.
        lang: Language code.
        model: LiteLLM model identifier.

    Returns:
        List of IPA transcription strings, one per input text.
    """
    if not texts:
        return []

    # Build numbered input
    numbered_lines = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
    user_content = f"Language: {lang}\n\n{numbered_lines}"

    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise click.ClickException(
                    f"LLM returned invalid JSON for IPA batch transcription: {e}"
                )
            results = data.get("results", [])
            if len(results) != len(texts):
                raise click.ClickException(
                    f"LLM returned {len(results)} transcriptions for {len(texts)} inputs. "
                    f"Try a different model or use --ipa-backend=espeak."
                )
            return [r.strip() for r in results]
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
