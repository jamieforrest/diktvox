"""Tests for Markdown output formatting."""

from diktvox.format import format_markdown, _group_voice_parts
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

    def test_page_number_headings(self):
        """Page numbers should appear as ### Page N headings."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="31",
                    page_number=4,
                    all_parts_same=True,
                    voice_parts=[
                        TranscribedVoicePart(name="All", text="Hello", ipa="hɛloː"),
                    ],
                ),
                TranscribedSection(
                    name="32",
                    page_number=5,
                    all_parts_same=True,
                    voice_parts=[
                        TranscribedVoicePart(name="All", text="World", ipa="vɛlt"),
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        assert "### Page 4" in md
        assert "### Page 5" in md
        assert "**[31]**" in md
        assert "**[32]**" in md

    def test_same_page_no_duplicate_heading(self):
        """Multiple sections on the same page should share one page heading."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="A", page_number=3,
                    voice_parts=[TranscribedVoicePart(name="All", text="a", ipa="a")],
                ),
                TranscribedSection(
                    name="B", page_number=3,
                    voice_parts=[TranscribedVoicePart(name="All", text="b", ipa="b")],
                ),
            ],
        )
        md = format_markdown(score)
        assert md.count("### Page 3") == 1

    def test_no_page_number_skips_heading(self):
        """Sections without page_number should not generate page headings."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="X",
                    voice_parts=[TranscribedVoicePart(name="All", text="x", ipa="x")],
                ),
            ],
        )
        md = format_markdown(score)
        assert "### Page" not in md

    def test_empty_section_name_no_annotation(self):
        """Sections with empty name should not produce a bold annotation line."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="",
                    page_number=1,
                    voice_parts=[TranscribedVoicePart(name="All", text="x", ipa="x")],
                ),
            ],
        )
        md = format_markdown(score)
        assert "**[]**" not in md


class TestGroupVoiceParts:
    def test_identical_parts_grouped(self):
        section = TranscribedSection(
            name="Test",
            all_parts_same=False,
            voice_parts=[
                TranscribedVoicePart(name="Soprano", text="Leben!", ipa="lˈeːbən"),
                TranscribedVoicePart(name="Alto", text="Leben!", ipa="lˈeːbən"),
                TranscribedVoicePart(name="Tenor", text="Sterben!", ipa="ʃtˈɛɾbən"),
                TranscribedVoicePart(name="Bass", text="Sterben!", ipa="ʃtˈɛɾbən"),
            ],
        )
        groups = _group_voice_parts(section)
        assert len(groups) == 2
        labels = {g[0] for g in groups}
        assert "Soprano, Alto" in labels
        assert "Tenor, Bass" in labels

    def test_all_identical_becomes_all_parts(self):
        section = TranscribedSection(
            name="Test",
            all_parts_same=False,
            voice_parts=[
                TranscribedVoicePart(name="Soprano", text="Text", ipa="ipa"),
                TranscribedVoicePart(name="Alto", text="Text", ipa="ipa"),
                TranscribedVoicePart(name="Bass", text="Text", ipa="ipa"),
            ],
        )
        groups = _group_voice_parts(section)
        assert len(groups) == 1
        assert groups[0][0] == "All parts"

    def test_all_different_listed_separately(self):
        section = TranscribedSection(
            name="Test",
            all_parts_same=False,
            voice_parts=[
                TranscribedVoicePart(name="Soprano", text="A", ipa="a"),
                TranscribedVoicePart(name="Bass", text="B", ipa="b"),
            ],
        )
        groups = _group_voice_parts(section)
        assert len(groups) == 2


class TestGlossaryDiversity:
    def test_distinct_examples_per_symbol(self):
        """Each symbol should prefer a different example word."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="S1",
                    voice_parts=[
                        TranscribedVoicePart(
                            name="All",
                            text="Aufersteh'n Staub Ruh",
                            ipa="ˈaʊfɛɾʃtˌeːn ʃtaʊp ɾuː",
                        )
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        # The ʃ symbol should not use the same example as ɛ
        lines = [l for l in md.split("\n") if l.startswith("| ")]
        examples = {}
        for line in lines:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 4:
                sym = cols[1]
                ex = cols[3]
                if ex:
                    examples[sym] = ex
        # At least some symbols should have different examples
        example_words = [e.split("[")[0].strip() for e in examples.values() if "[" in e]
        if len(example_words) > 1:
            assert len(set(example_words)) > 1, "All symbols should not use the same example word"

    def test_short_o_umlaut_in_glossary(self):
        """The [ø] symbol should appear in the glossary, not as undocumented."""
        score = TranscribedScore(
            title="Test",
            sections=[
                TranscribedSection(
                    name="S1",
                    voice_parts=[
                        TranscribedVoicePart(
                            name="All",
                            text="Hör'",
                            ipa="høːɐ",
                        )
                    ],
                ),
            ],
        )
        md = format_markdown(score)
        assert "undocumented" not in md
