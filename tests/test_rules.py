"""Tests for the IPA rules engine."""

import pathlib
import tempfile

import pytest

from diktvox.ipa.rules import Rules, ContextualRule, InsertionRule, apply_rules, load_rules


@pytest.fixture
def basic_rules() -> Rules:
    return Rules(
        language="de",
        name="test",
        processing_order=["substitutions", "contextual", "insertions", "overrides"],
        substitutions={"ʁ": "ʀ"},
        contextual=[
            ContextualRule(match="ʀ", position="word_final", after="vowel", replace="ɐ"),
            ContextualRule(match="ʀ", position="word_initial", replace="r", after=None),
        ],
        insertions=[
            InsertionRule(insert="ʔ", position="before_word_initial_vowel"),
        ],
        overrides={"Herr": "hɛr"},
    )


class TestSubstitutions:
    def test_simple_substitution(self, basic_rules: Rules):
        result = apply_rules("ʁuːə", "Ruhe", basic_rules)
        # ʁ → ʀ via substitution, then contextual rules may apply
        assert "ʁ" not in result or result == "hɛr"  # overrides don't apply here

    def test_no_match(self, basic_rules: Rules):
        result = apply_rules("haʊ̯s", "Haus", basic_rules)
        assert "haʊ̯s" in result or result.startswith("h")


class TestContextualRules:
    def test_word_final_after_vowel(self):
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ʀ", position="word_final", after="vowel", replace="ɐ"),
            ],
        )
        result = apply_rules("vɪʀ", "wir", rules)
        assert result == "vɪɐ"

    def test_word_initial(self):
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ʀ", position="word_initial", replace="r", after=None),
            ],
        )
        result = apply_rules("ʀuːə", "Ruhe", rules)
        assert result == "ruːə"

    def test_disabled_rule(self):
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ʀ", position="word_initial", replace="r", enabled=False),
            ],
        )
        result = apply_rules("ʀuːə", "Ruhe", rules)
        assert result == "ʀuːə"


class TestInsertions:
    def test_glottal_stop_before_initial_vowel(self):
        rules = Rules(
            processing_order=["insertions"],
            insertions=[
                InsertionRule(insert="ʔ", position="before_word_initial_vowel"),
            ],
        )
        result = apply_rules("aʊ̯f", "auf", rules)
        assert result == "ʔaʊ̯f"

    def test_no_glottal_before_consonant(self):
        rules = Rules(
            processing_order=["insertions"],
            insertions=[
                InsertionRule(insert="ʔ", position="before_word_initial_vowel"),
            ],
        )
        result = apply_rules("ʃtaʊ̯p", "Staub", rules)
        assert result == "ʃtaʊ̯p"


class TestOverrides:
    def test_word_override(self):
        rules = Rules(
            processing_order=["overrides"],
            overrides={"Herr": "hɛr"},
        )
        result = apply_rules("hɛʁ deːɐ̯", "Herr der", rules)
        assert result.startswith("hɛr")

    def test_override_with_punctuation(self):
        rules = Rules(
            processing_order=["overrides"],
            overrides={"Herr": "hɛr"},
        )
        result = apply_rules("hɛʁ", "Herr,", rules)
        assert result == "hɛr"


class TestProcessingOrder:
    def test_overrides_win(self):
        """Overrides should take precedence since they're processed last."""
        rules = Rules(
            processing_order=["substitutions", "overrides"],
            substitutions={"ʁ": "ʀ"},
            overrides={"Kirche": "kɪʁçə"},
        )
        # The override should replace the word entirely, even after substitution
        result = apply_rules("kɪʁçə", "Kirche", rules)
        assert result == "kɪʁçə"


class TestLoadRules:
    def test_load_default_german(self):
        rules_path = pathlib.Path(__file__).resolve().parent.parent / "rules" / "de_standard.yaml"
        rules = load_rules(rules_path)
        assert rules.language == "de"
        assert rules.name == "Standard sung German"
        assert len(rules.substitutions) > 0
        assert len(rules.contextual) > 0
        assert len(rules.insertions) > 0
        assert len(rules.overrides) > 0

    def test_load_from_temp_file(self, tmp_path: pathlib.Path):
        yaml_content = """
language: de
name: test
processing_order: [substitutions]
substitutions:
  "ʁ": "r"
"""
        f = tmp_path / "test.yaml"
        f.write_text(yaml_content)
        rules = load_rules(f)
        assert rules.substitutions == {"ʁ": "r"}
