"""Tests for IPA transcription backends."""

from unittest.mock import MagicMock, patch

import click
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
        # Default rules apply glottal stop before word-initial vowels (even after stress marks)
        assert result.sections[0].voice_parts[0].ipa != ""

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


class TestLatinEspeakBackend:
    @patch("diktvox.ipa.espeak.subprocess.run")
    @patch("diktvox.ipa.espeak.shutil.which", return_value="/usr/bin/espeak-ng")
    def test_espeak_transcribe_latin(self, mock_which, mock_run):
        score = ScoreData(
            title="Test", language="la",
            sections=[
                Section(
                    name="S1", all_parts_same=True,
                    voice_parts=[VoicePart(name="All", text="Gloria in excelsis Deo")],
                ),
            ],
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ɡlˈoria in ɛkstʃˈɛlsis dˈeo\n",
            stderr="",
        )
        result = transcribe(score, backend="espeak", language="la", rules_path=None)
        assert result.sections[0].voice_parts[0].ipa != ""
        # espeak should be called with -v la
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-v" in call_args
        assert "la" in call_args


class TestLatinLLMBackend:
    @patch("diktvox.ipa.llm.litellm")
    def test_llm_transcribe_latin(self, mock_litellm):
        import json
        score = ScoreData(
            title="Test", language="la",
            sections=[
                Section(
                    name="S1", all_parts_same=True,
                    voice_parts=[VoicePart(name="All", text="Sanctus")],
                ),
            ],
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"results": ["ˈsanktus"]}
        )
        mock_litellm.completion.return_value = mock_response
        result = transcribe(score, backend="llm", language="la", rules_path=None, model="test-model")
        assert result.sections[0].voice_parts[0].ipa != ""

    @patch("diktvox.ipa.llm.litellm")
    def test_llm_system_prompt_includes_latin(self, mock_litellm):
        """LLM system prompt should include Ecclesiastical Latin conventions."""
        import json
        score = ScoreData(
            title="T", language="la",
            sections=[
                Section(
                    name="S", all_parts_same=True,
                    voice_parts=[VoicePart(name="All", text="Deo")],
                ),
            ],
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"results": ["deo"]})
        mock_litellm.completion.return_value = mock_response
        transcribe(score, backend="llm", language="la", rules_path=None, model="m")
        call_args = mock_litellm.completion.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "Ecclesiastical" in system_msg or "Church Latin" in system_msg


class TestLLMJsonParsing:
    """Test that the JSON parser handles various LLM response formats."""

    def test_markdown_fenced_json(self):
        from diktvox.ipa.llm import _parse_json_response
        content = '```json\n{"results": ["hɛloː"]}\n```'
        assert _parse_json_response(content, 1) == ["hɛloː"]

    def test_json_with_preamble(self):
        from diktvox.ipa.llm import _parse_json_response
        content = 'Here is the transcription:\n{"results": ["hɛloː", "vɛlt"]}'
        assert _parse_json_response(content, 2) == ["hɛloː", "vɛlt"]

    def test_empty_response_raises(self):
        from diktvox.ipa.llm import _parse_json_response
        with pytest.raises(click.ClickException, match="empty response"):
            _parse_json_response("", 1)

    def test_non_json_response_raises(self):
        from diktvox.ipa.llm import _parse_json_response
        with pytest.raises(click.ClickException, match="did not return valid JSON"):
            _parse_json_response("hɛloː vɛlt", 1)

    def test_wrong_count_raises(self):
        from diktvox.ipa.llm import _parse_json_response
        with pytest.raises(click.ClickException, match="2 transcriptions for 3 inputs"):
            _parse_json_response('{"results": ["a", "b"]}', 3)


class TestInvalidBackend:
    def test_unknown_backend(self, sample_score):
        with pytest.raises(ValueError, match="Unknown IPA backend"):
            transcribe(sample_score, backend="invalid", language="de")
