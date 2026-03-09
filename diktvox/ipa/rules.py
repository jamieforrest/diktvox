"""YAML rules loading and IPA post-processing engine.

Rules are organized into four tiers with explicit processing order:
1. substitutions — simple phoneme-to-phoneme replacements
2. contextual — position-dependent rules
3. insertions — add phonemes (e.g., glottal stops)
4. overrides — word-level overrides (highest priority)
"""

import pathlib
import re
from dataclasses import dataclass, field

import yaml

# Vowel sets for position detection
_FRONT_VOWELS = set("ieæɛøœyʏɪεeː")
_BACK_VOWELS = set("aɑɒoɔuʊ")
_ALL_VOWELS = _FRONT_VOWELS | _BACK_VOWELS | set("ə")
_CONSONANTS_PATTERN = re.compile(r"[bcdfghjklmnpqrstvwxzðŋɡʃʒθçʁʀɾɲɫ]", re.IGNORECASE)
_LIQUIDS = set("lɾrɫ")
_STRESS_MARKS = set("ˈˌ")


@dataclass
class ContextualRule:
    match: str
    position: str
    replace: str
    after: str | None = None
    after_grapheme: str | None = None
    enabled: bool = True


@dataclass
class InsertionRule:
    insert: str
    position: str
    enabled: bool = True


@dataclass
class Rules:
    language: str = ""
    name: str = ""
    description: str = ""
    processing_order: list[str] = field(default_factory=lambda: [
        "substitutions", "contextual", "insertions", "overrides"
    ])
    substitutions: dict[str, str] = field(default_factory=dict)
    contextual: list[ContextualRule] = field(default_factory=list)
    insertions: list[InsertionRule] = field(default_factory=list)
    overrides: dict[str, str] = field(default_factory=dict)


def load_rules(path: pathlib.Path) -> Rules:
    """Load rules from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    contextual = []
    for r in data.get("contextual", []):
        contextual.append(ContextualRule(
            match=r["match"],
            position=r["position"],
            replace=r["replace"],
            after=r.get("after"),
            after_grapheme=r.get("after_grapheme"),
            enabled=r.get("enabled", True),
        ))

    insertions = []
    for r in data.get("insertions", []):
        insertions.append(InsertionRule(
            insert=r["insert"],
            position=r["position"],
            enabled=r.get("enabled", True),
        ))

    return Rules(
        language=data.get("language", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        processing_order=data.get("processing_order", [
            "substitutions", "contextual", "insertions", "overrides"
        ]),
        substitutions=data.get("substitutions", {}),
        contextual=contextual,
        insertions=insertions,
        overrides=data.get("overrides", {}),
    )


def _is_vowel(char: str) -> bool:
    return char in _ALL_VOWELS


def _is_consonant(char: str) -> bool:
    return bool(_CONSONANTS_PATTERN.match(char))


def _apply_substitutions(ipa: str, rules: Rules) -> str:
    """Apply simple phoneme-to-phoneme substitutions."""
    for old, new in rules.substitutions.items():
        ipa = ipa.replace(old, new)
    return ipa


def _check_position(ipa_word: str, match_start: int, match_end: int, position: str, rule: ContextualRule) -> bool:
    """Check if a match at the given position satisfies the position constraint."""
    if position == "word_initial":
        # Skip leading stress marks (ˈ, ˌ) when checking word-initial position
        return match_start == 0 or all(c in _STRESS_MARKS for c in ipa_word[:match_start])
    elif position == "word_final":
        return match_end == len(ipa_word)
    elif position == "word_medial":
        return match_start > 0 and match_end < len(ipa_word)
    elif position == "intervocalic":
        if match_start == 0 or match_end >= len(ipa_word):
            return False
        return _is_vowel(ipa_word[match_start - 1]) and _is_vowel(ipa_word[match_end])
    elif position == "before_consonant":
        return match_end < len(ipa_word) and _is_consonant(ipa_word[match_end])
    elif position == "after_consonant":
        return match_start > 0 and _is_consonant(ipa_word[match_start - 1])
    elif position == "after_front_vowel":
        return match_start > 0 and ipa_word[match_start - 1] in _FRONT_VOWELS
    elif position == "after_back_vowel":
        return match_start > 0 and ipa_word[match_start - 1] in _BACK_VOWELS
    elif position == "before_liquid":
        return match_end < len(ipa_word) and ipa_word[match_end] in _LIQUIDS
    elif position == "syllable_final":
        # Approximate: treat as word-final or before a consonant cluster
        return match_end == len(ipa_word) or (
            match_end < len(ipa_word) and _is_consonant(ipa_word[match_end])
        )
    return False


def _apply_contextual(ipa: str, original_text: str, rules: Rules) -> str:
    """Apply position-dependent rules word by word."""
    ipa_words = ipa.split()
    orig_words = original_text.split()

    result_words = []
    for i, ipa_word in enumerate(ipa_words):
        orig_word = orig_words[i] if i < len(orig_words) else ""

        for rule in rules.contextual:
            if not rule.enabled:
                continue

            # Check grapheme constraint
            if rule.after_grapheme and rule.after_grapheme not in orig_word:
                continue

            # Check 'after' constraint (e.g., after: vowel)
            idx = 0
            new_word = ""
            while idx < len(ipa_word):
                match_start = ipa_word.find(rule.match, idx)
                if match_start == -1:
                    new_word += ipa_word[idx:]
                    break

                new_word += ipa_word[idx:match_start]
                match_end = match_start + len(rule.match)

                if _check_position(ipa_word, match_start, match_end, rule.position, rule):
                    after_ok = True
                    if rule.after == "vowel" and match_start > 0:
                        after_ok = _is_vowel(ipa_word[match_start - 1])
                    elif rule.after == "consonant" and match_start > 0:
                        after_ok = _is_consonant(ipa_word[match_start - 1])
                    elif rule.after and match_start == 0:
                        after_ok = False

                    if after_ok:
                        new_word += rule.replace
                    else:
                        new_word += rule.match
                else:
                    new_word += rule.match
                idx = match_end

            ipa_word = new_word

        result_words.append(ipa_word)

    return " ".join(result_words)


def _apply_insertions(ipa: str, rules: Rules) -> str:
    """Apply insertion rules (e.g., glottal stops before word-initial vowels)."""
    ipa_words = ipa.split()
    result_words = []

    for word in ipa_words:
        for rule in rules.insertions:
            if not rule.enabled:
                continue
            if rule.position == "before_word_initial_vowel":
                # Find first non-stress-mark character
                first_phoneme = 0
                while first_phoneme < len(word) and word[first_phoneme] in _STRESS_MARKS:
                    first_phoneme += 1
                if first_phoneme < len(word) and _is_vowel(word[first_phoneme]):
                    # Insert glottal stop after any leading stress marks
                    word = word[:first_phoneme] + rule.insert + word[first_phoneme:]

        result_words.append(word)

    return " ".join(result_words)


def _apply_overrides(ipa: str, original_text: str, rules: Rules) -> str:
    """Apply word-level overrides (highest priority)."""
    if not rules.overrides:
        return ipa

    ipa_words = ipa.split()
    orig_words = original_text.split()

    for i, orig_word in enumerate(orig_words):
        # Strip punctuation for matching
        clean = orig_word.strip(".,;:!?\"'()[]")
        if clean in rules.overrides and i < len(ipa_words):
            ipa_words[i] = rules.overrides[clean]

    return " ".join(ipa_words)


def apply_rules(ipa: str, original_text: str, rules: Rules | None) -> str:
    """Apply all rule tiers in the configured processing order.

    Args:
        ipa: Raw IPA transcription from the backend.
        original_text: Original text (needed for grapheme-dependent rules and overrides).
        rules: Loaded rules, or None to skip post-processing.

    Returns:
        Post-processed IPA string.
    """
    if rules is None:
        return ipa

    processors = {
        "substitutions": lambda ipa_text: _apply_substitutions(ipa_text, rules),
        "contextual": lambda ipa_text: _apply_contextual(ipa_text, original_text, rules),
        "insertions": lambda ipa_text: _apply_insertions(ipa_text, rules),
        "overrides": lambda ipa_text: _apply_overrides(ipa_text, original_text, rules),
    }

    for step in rules.processing_order:
        if step in processors:
            ipa = processors[step](ipa)

    return ipa
