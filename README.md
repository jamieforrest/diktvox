# diktvox

IPA transcription companion for choral singers. Feed it a PDF score, get back a formatted guide with the original text and IPA phonemes organized by voice part. Supports configurable sung-language conventions (vocalized r, glottal onsets, etc). German first, more languages coming.

## Requirements

- Python 3.10+
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (only required for the default `espeak` IPA backend)
- An API key for a vision-capable LLM (Claude, GPT-4o, Gemini, etc.)

## Installation

```bash
# Clone the repo
git clone https://github.com/jamieforrest/diktvox.git
cd diktvox

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install diktvox in editable mode
pip install -e .

# (Optional) Install dev dependencies for running tests
pip install -e ".[dev]"
```

### Install espeak-ng

Required only when using `--ipa-backend espeak` (the default).

```bash
# Linux (Debian/Ubuntu)
sudo apt install espeak-ng

# macOS
brew install espeak-ng
```

### Set your LLM API key

diktvox uses [LiteLLM](https://github.com/BerriAI/litellm) so you can use any supported provider. Copy the sample env file and fill in your key:

```bash
cp .env.sample .env
# Edit .env and add your API key
```

Or export environment variables directly:

```bash
export ANTHROPIC_API_KEY=sk-...    # for Claude models
# or
export OPENAI_API_KEY=sk-...       # for GPT-4o
# or
export GEMINI_API_KEY=...          # for Gemini
```

The `.env` file is loaded automatically at startup and is already in `.gitignore`.

## Usage

```bash
# Basic usage — outputs to stdout
diktvox score.pdf

# Save to a file
diktvox score.pdf -o output.md

# Full options
diktvox score.pdf \
  --output output.md \
  --language de \
  --ipa-backend espeak \
  --rules rules/de_standard.yaml \
  --model gemini/gemini-2.0-flash \
  --format md

# Use the LLM backend for IPA (no espeak-ng install needed)
diktvox score.pdf --ipa-backend llm -o output.md

# Skip cache (re-run LLM extraction even if the PDF was processed before)
diktvox score.pdf --no-cache -o output.md

# Compare backends
diktvox score.pdf --ipa-backend espeak -o espeak_output.md
diktvox score.pdf --ipa-backend llm -o llm_output.md
diff espeak_output.md llm_output.md
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | stdout | Output file path |
| `--language` | `de` | Language code (v1: German only) |
| `--ipa-backend` | `espeak` | IPA engine: `espeak` or `llm` |
| `--rules` | `rules/de_standard.yaml` | Custom YAML rules file for singing conventions |
| `--model` | `anthropic/claude-sonnet-4-20250514` | LiteLLM model string for vision |
| `--format` | `md` | Output format |
| `--no-cache` | off | Disable PDF extraction caching |

## Singing Convention Rules

Pronunciation rules live in YAML config files. The default German rules are in `rules/de_standard.yaml`. Rules are processed in four tiers (later tiers override earlier ones):

1. **Substitutions** — simple phoneme swaps (e.g., uvular fricative → uvular trill)
2. **Contextual** — position-dependent rules (e.g., vocalized r in final -er)
3. **Insertions** — add phonemes (e.g., glottal stops before word-initial vowels)
4. **Overrides** — word-level overrides (highest priority)

Pass a custom rules file with `--rules my_rules.yaml`. Conductors can share rule files with their ensembles.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
