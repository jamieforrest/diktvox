"""Tests for PDF output generation."""

import pathlib

from click.testing import CliRunner

from diktvox.format import format_markdown
from diktvox.format_pdf import format_pdf
from diktvox.models import TranscribedScore, TranscribedSection, TranscribedVoicePart


def _sample_score() -> TranscribedScore:
    return TranscribedScore(
        title="Auferstehung",
        composer="Gustav Mahler",
        edition="Cunningham/Goodmusic",
        language="de",
        sections=[
            TranscribedSection(
                name="Rehearsal 31",
                page_number=1,
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


class TestFormatPdf:
    def test_returns_bytes(self):
        md = format_markdown(_sample_score())
        result = format_pdf(md)
        assert isinstance(result, bytes)

    def test_pdf_magic_bytes(self):
        md = format_markdown(_sample_score())
        result = format_pdf(md)
        assert result[:5] == b"%PDF-"

    def test_nonempty_output(self):
        md = format_markdown(_sample_score())
        result = format_pdf(md)
        assert len(result) > 100

    def test_empty_markdown(self):
        result = format_pdf("")
        assert result[:5] == b"%PDF-"


class TestCliPdfFormat:
    def test_pdf_format_requires_output(self):
        from diktvox.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--format", "pdf", "dummy.pdf"])
        # Should fail because either the file doesn't exist or because no -o given
        assert result.exit_code != 0

    def test_format_choices(self):
        from diktvox.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--format", "invalid", "dummy.pdf"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "Invalid value" in result.output

    def test_md_plus_pdf_accepted(self):
        """md+pdf should be a valid format choice."""
        from diktvox.__main__ import cli

        runner = CliRunner()
        # Will fail due to missing file, but the format option should be accepted
        result = runner.invoke(cli, ["--format", "md+pdf", "nonexistent.pdf"])
        # The error should be about the missing file, not the format choice
        assert "Invalid value for '--format'" not in (result.output or "")
