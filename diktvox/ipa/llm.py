"""LLM-based IPA transcription backend."""

import json
import re
import time

import click
import litellm

_SYSTEM_PROMPT = """\
You are an expert in IPA (International Phonetic Alphabet) transcription for sung text.

Given numbered text lines and a language, produce an accurate IPA transcription for each line.
Apply standard singing diction conventions:
- For German: use standard sung German conventions (rolled r at word start, vocalized r in \
final -er, ich-laut after front vowels, ach-laut after back vowels, etc.)
- Include stress marks: use [ˈ] for primary stress and [ˌ] for secondary stress before the \
stressed syllable.
- Preserve word boundaries with spaces.

Respond with valid JSON only, no markdown fencing or explanation. Use this exact format:
{"results": ["ipa line 1", "ipa line 2", ...]}
"""

_MAX_RETRIES = 4
_BACKOFF_DELAYS = [2, 4, 8, 16]

# Match JSON object in LLM output that may contain markdown fencing or preamble
_JSON_EXTRACT_RE = re.compile(r"\{[^{}]*\"results\"\s*:\s*\[.*?\]\s*\}", re.DOTALL)


def _parse_json_response(content: str, expected_count: int) -> list[str]:
    """Parse JSON results from LLM response, handling markdown fencing and preamble."""
    if not content:
        raise click.ClickException(
            "LLM returned empty response for IPA transcription. "
            "Try a different --ipa-model or use --ipa-backend=espeak."
        )

    # Try direct parse first
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks or surrounding text
        match = _JSON_EXTRACT_RE.search(content)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if data is None or "results" not in data:
        # Show a truncated preview of what we got
        preview = content[:200] + ("..." if len(content) > 200 else "")
        raise click.ClickException(
            f"LLM did not return valid JSON with a 'results' array.\n"
            f"Try a different --ipa-model or use --ipa-backend=espeak.\n"
            f"Response preview: {preview}"
        )

    results = data["results"]
    if len(results) != expected_count:
        raise click.ClickException(
            f"LLM returned {len(results)} transcriptions for {expected_count} inputs. "
            f"Try a different model or use --ipa-backend=espeak."
        )
    return [r.strip() for r in results]


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
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return _parse_json_response(content.strip(), len(texts))
        except click.ClickException:
            raise
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
