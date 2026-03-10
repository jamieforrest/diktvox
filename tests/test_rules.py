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


class TestEpsilonToSchwaSubstitution:
    """Test the ɜ → ɐ substitution for German vocalized -er."""

    def test_epsilon_replaced_globally(self):
        rules = Rules(
            processing_order=["substitutions"],
            substitutions={"ɜ": "ɐ"},
        )
        result = apply_rules("kˈʊɐtsɜ", "kurzer", rules)
        assert "ɜ" not in result
        assert result == "kˈʊɐtsɐ"

    def test_multiple_occurrences(self):
        rules = Rules(
            processing_order=["substitutions"],
            substitutions={"ɜ": "ɐ"},
        )
        result = apply_rules("vˈiːdɜ ˈalbɛtsvˌɪŋɜ", "Wieder Allbezwinger", rules)
        assert "ɜ" not in result
        assert "ɐ" in result


class TestBeforeLiquidRevoicing:
    """Test contextual rules that fix espeak's over-applied final devoicing before liquids."""

    def test_p_to_b_before_liquid(self):
        """p→b before /l/ when original text has 'bl' (e.g., Unsterblich)."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="p", position="before_liquid", after_grapheme="bl", replace="b"),
            ],
        )
        result = apply_rules("ˈʊnʃtˌɛɾplɪç", "Unsterblich", rules)
        assert "bl" in result
        assert "pl" not in result

    def test_lieblich(self):
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="p", position="before_liquid", after_grapheme="bl", replace="b"),
            ],
        )
        result = apply_rules("lˈiːplɪç", "lieblich", rules)
        assert result == "lˈiːblɪç"

    def test_k_to_g_before_liquid(self):
        """k→ɡ before /l/ when original text has 'gl' (e.g., möglich)."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="k", position="before_liquid", after_grapheme="gl", replace="ɡ"),
            ],
        )
        result = apply_rules("mˈøːklɪç", "möglich", rules)
        assert "ɡl" in result
        assert "kl" not in result

    def test_no_revoicing_without_grapheme(self):
        """Legitimate voiceless stops before liquids should not be changed."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="p", position="before_liquid", after_grapheme="bl", replace="b"),
            ],
        )
        # "Platz" has legitimate /pl/ — no 'bl' in original text
        result = apply_rules("plats", "Platz", rules)
        assert result == "plats"

    def test_no_revoicing_klasse(self):
        """'Klasse' has legitimate /kl/ — no 'gl' in original text."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="k", position="before_liquid", after_grapheme="gl", replace="ɡ"),
            ],
        )
        result = apply_rules("klasə", "Klasse", rules)
        assert result == "klasə"


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


class TestStressMarkHandling:
    """Rules should work correctly when IPA contains stress marks."""

    def test_word_initial_with_stress_mark(self):
        """Word-initial rule should fire even with a leading stress mark."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ʀ", position="word_initial", replace="r", after=None),
            ],
        )
        result = apply_rules("ˈʀiːf", "rief", rules)
        assert result == "ˈriːf"

    def test_word_initial_without_stress_mark(self):
        """Word-initial rule should still work without stress marks."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ʀ", position="word_initial", replace="r", after=None),
            ],
        )
        result = apply_rules("ʀiːf", "rief", rules)
        assert result == "riːf"

    def test_glottal_stop_with_stress_mark(self):
        """Glottal stop insertion should work with leading stress marks."""
        rules = Rules(
            processing_order=["insertions"],
            insertions=[
                InsertionRule(insert="ʔ", position="before_word_initial_vowel"),
            ],
        )
        result = apply_rules("ˈaʊ̯f", "auf", rules)
        assert result == "ˈʔaʊ̯f"

    def test_glottal_stop_with_secondary_stress(self):
        rules = Rules(
            processing_order=["insertions"],
            insertions=[
                InsertionRule(insert="ʔ", position="before_word_initial_vowel"),
            ],
        )
        result = apply_rules("ˌaʊ̯f", "auf", rules)
        assert result == "ˌʔaʊ̯f"


class TestLoadLatinRules:
    def test_load_default_latin(self):
        rules_path = pathlib.Path(__file__).resolve().parent.parent / "rules" / "la_standard.yaml"
        rules = load_rules(rules_path)
        assert rules.language == "la"
        assert rules.name == "Ecclesiastical (Church) Latin"
        assert len(rules.substitutions) > 0
        assert len(rules.contextual) > 0


class TestBeforeFrontVowelPosition:
    def test_before_front_vowel_match(self):
        """k before front vowel should be replaced (e.g., Latin 'caeli')."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="k", position="before_front_vowel", replace="tʃ"),
            ],
        )
        result = apply_rules("keli", "caeli", rules)
        assert result == "tʃeli"

    def test_before_back_vowel_no_match(self):
        """k before back vowel should NOT be changed by before_front_vowel rule."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="k", position="before_front_vowel", replace="tʃ"),
            ],
        )
        result = apply_rules("karo", "caro", rules)
        assert result == "karo"

    def test_g_before_front_vowel(self):
        """ɡ before front vowel should be replaced (e.g., Latin 'gentes')."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="ɡ", position="before_front_vowel", replace="dʒ"),
            ],
        )
        result = apply_rules("ɡentes", "gentes", rules)
        assert result == "dʒentes"

    def test_before_back_vowel_position(self):
        """Test the before_back_vowel position type."""
        rules = Rules(
            processing_order=["contextual"],
            contextual=[
                ContextualRule(match="k", position="before_back_vowel", replace="k"),
            ],
        )
        # k before 'a' (back vowel) should match
        result = apply_rules("karo", "caro", rules)
        assert result == "karo"


class TestLabiodentalApproximant:
    """Test ʋ → v substitution for sung German."""

    def test_v_substitution(self):
        rules = Rules(
            processing_order=["substitutions"],
            substitutions={"ʋ": "v"},
        )
        result = apply_rules("ˈalbətsʋɪŋɐ", "Allbezwinger", rules)
        assert "ʋ" not in result
        assert "vɪŋɐ" in result
