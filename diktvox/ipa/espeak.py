"""espeak-ng subprocess wrapper for IPA transcription.

Calls espeak-ng as a subprocess rather than linking against it,
keeping our MIT license clean (espeak-ng is GPL).
"""

import shutil
import subprocess


def _find_espeak() -> str:
    """Locate the espeak-ng binary."""
    path = shutil.which("espeak-ng")
    if path is None:
        raise RuntimeError(
            "espeak-ng is not installed or not found on PATH. "
            "Install it with: apt install espeak-ng (Linux) or brew install espeak-ng (macOS)"
        )
    return path


def espeak_transcribe(text: str, language: str = "de") -> str:
    """Transcribe text to IPA using espeak-ng.

    Args:
        text: Input text to transcribe.
        language: Language code (e.g., "de" for German).

    Returns:
        IPA transcription string.
    """
    espeak_bin = _find_espeak()

    result = subprocess.run(
        [espeak_bin, "--ipa", "-q", "-v", language, text],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {result.stderr.strip()}")

    # espeak-ng outputs IPA with spaces between words and newlines between lines.
    # Join lines and clean up extra whitespace.
    lines = result.stdout.strip().splitlines()
    return " ".join(line.strip() for line in lines if line.strip())
