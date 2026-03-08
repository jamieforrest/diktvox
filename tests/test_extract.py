"""Tests for text extraction (using mocked LLM calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from diktvox.extract import _merge_page_results, _parse_response, extract_text


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


class TestMergePageResults:
    def test_metadata_from_first_page(self):
        pages = [
            {"title": "My Piece", "composer": "Bach", "edition": "Peters", "language": "de", "sections": []},
            {"title": "", "composer": "", "edition": "", "language": "", "sections": []},
        ]
        merged = _merge_page_results(pages)
        assert merged["title"] == "My Piece"
        assert merged["composer"] == "Bach"
        assert merged["edition"] == "Peters"

    def test_sections_concatenated(self):
        pages = [
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Verse 1", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "Hello"}]},
            ]},
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Verse 2", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "World"}]},
            ]},
        ]
        merged = _merge_page_results(pages)
        assert len(merged["sections"]) == 2
        assert merged["sections"][0]["name"] == "Verse 1"
        assert merged["sections"][1]["name"] == "Verse 2"

    def test_continued_section_merged(self):
        pages = [
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Verse 1", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "First part"}]},
            ]},
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Verse 1 (continued)", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "second part"}]},
            ]},
        ]
        merged = _merge_page_results(pages)
        assert len(merged["sections"]) == 1
        assert merged["sections"][0]["name"] == "Verse 1"
        assert merged["sections"][0]["voice_parts"][0]["text"] == "First part second part"

    def test_continued_adds_new_voice_part(self):
        pages = [
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Chorus", "all_parts_same": False, "voice_parts": [{"name": "Soprano", "text": "La la"}]},
            ]},
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Chorus (continued)", "all_parts_same": False, "voice_parts": [
                    {"name": "Soprano", "text": "la la"},
                    {"name": "Alto", "text": "dum dum"},
                ]},
            ]},
        ]
        merged = _merge_page_results(pages)
        assert len(merged["sections"]) == 1
        parts = {vp["name"]: vp["text"] for vp in merged["sections"][0]["voice_parts"]}
        assert parts["Soprano"] == "La la la la"
        assert parts["Alto"] == "dum dum"

    def test_empty_pages_skipped(self):
        pages = [
            {"title": "Piece", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Intro", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "Sing"}]},
            ]},
            {"title": "", "composer": "", "edition": "", "language": "", "sections": []},
            {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
                {"name": "Verse 1", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "More"}]},
            ]},
        ]
        merged = _merge_page_results(pages)
        assert len(merged["sections"]) == 2


def _make_mock_response(data: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(data)
    mock_response.choices[0].finish_reason = "stop"
    return mock_response


class TestExtractText:
    @patch("diktvox.extract.litellm")
    def test_extract_with_mock_llm(self, mock_litellm):
        page_data = {
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
        mock_litellm.completion.return_value = _make_mock_response(page_data)

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
    def test_extract_multiple_pages(self, mock_litellm):
        page1 = {"title": "Piece", "composer": "Composer", "edition": "", "language": "de", "sections": [
            {"name": "Verse 1", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "Page one text"}]},
        ]}
        page2 = {"title": "", "composer": "", "edition": "", "language": "de", "sections": [
            {"name": "Verse 2", "all_parts_same": True, "voice_parts": [{"name": "All", "text": "Page two text"}]},
        ]}
        mock_litellm.completion.side_effect = [
            _make_mock_response(page1),
            _make_mock_response(page2),
        ]

        result = extract_text(
            ["page1data", "page2data"],
            model="test-model",
            language="de",
            use_cache=False,
        )

        assert result.title == "Piece"
        assert len(result.sections) == 2
        assert mock_litellm.completion.call_count == 2

    @patch("diktvox.extract.litellm")
    def test_language_mismatch_raises(self, mock_litellm):
        response_data = {
            "title": "Test",
            "language": "it",
            "sections": [],
        }
        mock_litellm.completion.return_value = _make_mock_response(response_data)

        import click

        with pytest.raises(click.ClickException, match="Detected language 'it'"):
            extract_text(
                ["base64data"],
                model="test-model",
                language="de",
                use_cache=False,
            )
