"""Tests for text extraction (using mocked LLM calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from diktvox.extract import _parse_response, extract_text


class TestParseResponse:
    def test_basic_parse(self):
        data = {
            "title": "Auferstehung",
            "composer": "Gustav Mahler",
            "edition": "",
            "language": "de",
            "sections": [
                {
                    "name": "Rehearsal 31",
                    "all_parts_same": True,
                    "voice_parts": [
                        {"name": "All", "text": "Aufersteh'n, ja aufersteh'n wirst du"}
                    ],
                },
                {
                    "name": "Rehearsal 32",
                    "all_parts_same": False,
                    "voice_parts": [
                        {"name": "Soprano", "text": "O glaube, mein Herz, o glaube"},
                        {"name": "Alto", "text": "O glaube, mein Herz"},
                    ],
                },
            ],
        }
        score = _parse_response(data)
        assert score.title == "Auferstehung"
        assert score.composer == "Gustav Mahler"
        assert len(score.sections) == 2
        assert score.sections[0].all_parts_same is True
        assert score.sections[0].voice_parts[0].name == "All"
        assert len(score.sections[1].voice_parts) == 2

    def test_empty_sections(self):
        data = {"title": "Test", "sections": []}
        score = _parse_response(data)
        assert score.title == "Test"
        assert len(score.sections) == 0

    def test_missing_fields(self):
        data = {}
        score = _parse_response(data)
        assert score.title == ""
        assert score.language == "de"


class TestExtractText:
    @patch("diktvox.extract.litellm")
    def test_extract_with_mock_llm(self, mock_litellm):
        response_data = {
            "title": "Test Piece",
            "composer": "Test Composer",
            "edition": "",
            "language": "de",
            "sections": [
                {
                    "name": "Section 1",
                    "all_parts_same": True,
                    "voice_parts": [
                        {"name": "All", "text": "Test text"}
                    ],
                }
            ],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_data)
        mock_litellm.completion.return_value = mock_response

        result = extract_text(
            ["base64data"],
            model="test-model",
            language="de",
            use_cache=False,
        )

        assert result.title == "Test Piece"
        assert len(result.sections) == 1
        mock_litellm.completion.assert_called_once()

    @patch("diktvox.extract.litellm")
    def test_language_mismatch_raises(self, mock_litellm):
        response_data = {
            "title": "Test",
            "language": "it",
            "sections": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_data)
        mock_litellm.completion.return_value = mock_response

        import click

        with pytest.raises(click.ClickException, match="Detected language 'it'"):
            extract_text(
                ["base64data"],
                model="test-model",
                language="de",
                use_cache=False,
            )
