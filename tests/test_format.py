"""Tests for Markdown output formatting."""

from diktvox.format import format_markdown
from diktvox.models import TranscribedScore, TranscribedSection, TranscribedVoicePart


class TestFormatMarkdown:
    def test_basic_output(self):
        score = TranscribedScore(
            title="Auferstehung",
            composer="Gustav Mahler",
            edition="Cunningham/Goodmusic",
            language="de",
            sections=[
                TranscribedSection(
                    name="Rehearsal 31",
                    all_parts_same=True,
                    voice_parts=[
                        TranscribedVoicePart(
                            name="All",
                            text="Aufersteh'n, ja aufersteh'n",
                            ipa="ˈaʊ̯fɛɐ̯ˌʃteːn jaː ˈaʊ̯fɛɐ̯ˌʃteːn",
                        )
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        assert "# Auferstehung — IPA Companion" in md
        assert "## Gustav Mahler / Cunningham/Goodmusic" in md
        assert "**All parts:**" in md
        assert "Aufersteh'n, ja aufersteh'n" in md
        assert "[ˈaʊ̯fɛɐ̯ˌʃteːn jaː ˈaʊ̯fɛɐ̯ˌʃteːn]" in md

    def test_separate_voice_parts(self):
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="Section 1",
                    all_parts_same=False,
                    voice_parts=[
                        TranscribedVoicePart(name="Soprano", text="Text A", ipa="ipa_a"),
                        TranscribedVoicePart(name="Bass", text="Text B", ipa="ipa_b"),
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        assert "**Soprano:**" in md
        assert "**Bass:**" in md
        assert "**All parts:**" not in md

    def test_ipa_glossary(self):
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="S1",
                    voice_parts=[
                        TranscribedVoicePart(
                            name="All",
                            text="Staub",
                            ipa="ʃtaʊ̯p",
                        )
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        assert "## IPA Symbol Reference" in md
        assert "| Symbol |" in md

    def test_empty_score(self):
        score = TranscribedScore()
        md = format_markdown(score)
        assert "# Choral Score — IPA Companion" in md
