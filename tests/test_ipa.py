"""Tests for IPA transcription backends."""

from unittest.mock import MagicMock, patch

import pytest

from diktvox.ipa import transcribe
from diktvox.models import ScoreData, Section, VoicePart


@pytest.fixture
def sample_score() -> ScoreData:
    return ScoreData(
        title="Test",
        composer="Test",
        language="de",
        sections=[
            Section(
                name="Section 1",
                all_parts_same=True,
                voice_parts=[
                    VoicePart(name="All", text="Aufersteh'n"),
                ],
            ),
        ],
    )


class TestEspeakBackend:
    @patch("diktvox.ipa.espeak.subprocess.run")
    @patch("diktvox.ipa.espeak.shutil.which", return_value="/usr/bin/espeak-ng")
    def test_espeak_transcribe(self, mock_which, mock_run, sample_score):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ˈaʊ̯fɛɐ̯ˌʃteːn\n",
            stderr="",
        )
        result = transcribe(sample_score, backend="espeak", language="de", rules_path=None)
        assert len(result.sections) == 1
        assert result.sections[0].voice_parts[0].ipa != ""

    @patch("diktvox.ipa.espeak.shutil.which", return_value=None)
    def test_espeak_not_found(self, mock_which, sample_score):
        with pytest.raises(RuntimeError, match="espeak-ng is not installed"):
            transcribe(sample_score, backend="espeak", language="de", rules_path=None)


class TestLLMBackend:
    @patch("diktvox.ipa.llm.litellm")
    def test_llm_transcribe(self, mock_litellm, sample_score):
        import json
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"results": ["ˈaʊ̯fɛɐ̯ˌʃteːn"]}
        )
        mock_litellm.completion.return_value = mock_response

        result = transcribe(
            sample_score, backend="llm", language="de",
            rules_path=None, model="test-model",
        )
        assert result.sections[0].voice_parts[0].ipa == "ˈaʊ̯fɛɐ̯ˌʃteːn"

    @patch("diktvox.ipa.llm.litellm")
    def test_llm_batches_all_parts(self, mock_litellm):
        """All voice parts should be sent in a single LLM call."""
        import json
        score = ScoreData(
            title="Test", language="de",
            sections=[
                Section(name="A", voice_parts=[
                    VoicePart(name="Soprano", text="Hallo"),
                    VoicePart(name="Bass", text="Welt"),
                ]),
                Section(name="B", voice_parts=[
                    VoicePart(name="All", text="Ja"),
                ]),
            ],
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"results": ["hˈaloː", "vˈɛlt", "jˈaː"]}
        )
        mock_litellm.completion.return_value = mock_response

        result = transcribe(score, backend="llm", language="de", rules_path=None, model="m")
        # Should have made exactly 1 LLM call for all 3 texts
        assert mock_litellm.completion.call_count == 1
        assert result.sections[0].voice_parts[0].ipa == "hˈaloː"
        assert result.sections[0].voice_parts[1].ipa == "vˈɛlt"
        assert result.sections[1].voice_parts[0].ipa == "jˈaː"


class TestInvalidBackend:
    def test_unknown_backend(self, sample_score):
        with pytest.raises(ValueError, match="Unknown IPA backend"):
            transcribe(sample_score, backend="invalid", language="de")
