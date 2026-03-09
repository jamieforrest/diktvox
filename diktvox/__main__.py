"""CLI entry point for diktvox."""

import pathlib

import click
from dotenv import load_dotenv

load_dotenv()

from diktvox.extract import extract_text
from diktvox.format import format_markdown
from diktvox.format_pdf import format_pdf
from diktvox.ipa import transcribe
from diktvox.pdf import render_pages


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-o", "--output", type=click.Path(path_type=pathlib.Path), default=None, help="Output file path. Defaults to stdout.")
@click.option("--language", default="de", help="Language code (default: de). v1 supports German only.")
@click.option("--ipa-backend", type=click.Choice(["espeak", "llm"]), default="espeak", help="IPA transcription backend.")
@click.option("--rules", "rules_path", type=click.Path(exists=True, path_type=pathlib.Path), default=None, help="Custom YAML rules file for singing conventions.")
@click.option("--model", default="anthropic/claude-sonnet-4-20250514", help="LiteLLM model for text extraction (vision-capable).")
@click.option("--ipa-model", default=None, help="LiteLLM model for IPA transcription (only used with --ipa-backend=llm). Defaults to --model.")
@click.option("--format", "output_formats", type=click.Choice(["md", "pdf"]), multiple=True, default=("md",), help="Output format(s). May be repeated, e.g. --format md --format pdf.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable PDF extraction caching.")
def cli(
    pdf_path: pathlib.Path,
    output: pathlib.Path | None,
    language: str,
    ipa_backend: str,
    rules_path: pathlib.Path | None,
    model: str,
    ipa_model: str | None,
    output_formats: tuple[str, ...],
    no_cache: bool,
) -> None:
    """Generate an IPA companion document from a choral score PDF."""
    if language != "de":
        raise click.ClickException(f"Language '{language}' is not yet supported. v1 supports German only.")

    # Resolve IPA model: explicit --ipa-model, or fall back to --model
    resolved_ipa_model = ipa_model or model

    # Step 1: Render PDF pages to images
    click.echo(f"Rendering pages from {pdf_path}...", err=True)
    pages = render_pages(pdf_path)
    click.echo(f"  {len(pages)} page(s) rendered.", err=True)

    # Step 2: Extract text via LLM vision
    click.echo(f"Extracting text via LLM vision (model: {model})...", err=True)
    score_data = extract_text(pages, model=model, language=language, use_cache=not no_cache, pdf_path=pdf_path)
    click.echo(f"  Done — {len(score_data.sections)} section(s) extracted.", err=True)

    # Step 3: IPA transcription
    click.echo(f"Transcribing to IPA (backend: {ipa_backend})...", err=True)
    transcribed = transcribe(score_data, backend=ipa_backend, language=language, rules_path=rules_path, model=resolved_ipa_model)

    # Step 4: Format output
    click.echo("Formatting output...", err=True)
    md = format_markdown(transcribed)
    formats = set(output_formats)

    if "md" in formats:
        if output:
            md_path = output.with_suffix(".md")
            md_path.write_text(md, encoding="utf-8")
            click.echo(f"Written to {md_path}", err=True)
        else:
            click.echo(md)

    if "pdf" in formats:
        click.echo("Generating PDF...", err=True)
        pdf_bytes = format_pdf(md)
        if output:
            pdf_path = output.with_suffix(".pdf")
            pdf_path.write_bytes(pdf_bytes)
            click.echo(f"Written to {pdf_path}", err=True)
        else:
            raise click.ClickException("PDF output requires -o/--output (cannot write binary to stdout).")


if __name__ == "__main__":
    cli()
