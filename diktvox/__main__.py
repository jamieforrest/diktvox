"""CLI entry point for diktvox."""

import pathlib

import click
from dotenv import load_dotenv

load_dotenv()

from diktvox.extract import extract_text
from diktvox.format import format_markdown
from diktvox.ipa import transcribe
from diktvox.pdf import render_pages


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-o", "--output", type=click.Path(path_type=pathlib.Path), default=None, help="Output file path. Defaults to stdout.")
@click.option("--language", default="de", help="Language code (default: de). v1 supports German only.")
@click.option("--ipa-backend", type=click.Choice(["espeak", "llm"]), default="espeak", help="IPA transcription backend.")
@click.option("--rules", "rules_path", type=click.Path(exists=True, path_type=pathlib.Path), default=None, help="Custom YAML rules file for singing conventions.")
@click.option("--model", default="anthropic/claude-sonnet-4-20250514", help="LiteLLM model string for the vision LLM.")
@click.option("--format", "output_format", type=click.Choice(["md"]), default="md", help="Output format.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable PDF extraction caching.")
def cli(
    pdf_path: pathlib.Path,
    output: pathlib.Path | None,
    language: str,
    ipa_backend: str,
    rules_path: pathlib.Path | None,
    model: str,
    output_format: str,
    no_cache: bool,
) -> None:
    """Generate an IPA companion document from a choral score PDF."""
    if language != "de":
        raise click.ClickException(f"Language '{language}' is not yet supported. v1 supports German only.")

    # Step 1: Render PDF pages to images
    click.echo(f"Rendering pages from {pdf_path}...", err=True)
    pages = render_pages(pdf_path)
    click.echo(f"  {len(pages)} page(s) rendered.", err=True)

    # Step 2: Extract text via LLM vision
    click.echo(f"Extracting text via LLM vision (model: {model}, {len(pages)} page(s))...", err=True)
    click.echo("  This may take a minute for large scores.", err=True)
    score_data = extract_text(pages, model=model, language=language, use_cache=not no_cache, pdf_path=pdf_path)
    click.echo(f"  Extracted {len(score_data.sections)} section(s).", err=True)

    # Step 3: IPA transcription
    click.echo(f"Transcribing to IPA (backend: {ipa_backend})...", err=True)
    transcribed = transcribe(score_data, backend=ipa_backend, language=language, rules_path=rules_path, model=model)

    # Step 4: Format output
    click.echo("Formatting output...", err=True)
    md = format_markdown(transcribed)

    if output:
        output.write_text(md, encoding="utf-8")
        click.echo(f"Written to {output}", err=True)
    else:
        click.echo(md)


if __name__ == "__main__":
    cli()
