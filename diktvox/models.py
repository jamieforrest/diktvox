"""Data models for diktvox."""

from dataclasses import dataclass, field


@dataclass
class VoicePart:
    """Text for a single voice part in a section."""

    name: str  # e.g. "Soprano", "Alto", "Tenor", "Bass"
    text: str  # Reconstructed sung text (full words, no syllable breaks)


@dataclass
class Section:
    """A section of a choral score (e.g., a rehearsal number)."""

    name: str  # e.g. "Rehearsal 31", "Opening Chorale"
    voice_parts: list[VoicePart] = field(default_factory=list)
    all_parts_same: bool = False  # If True, all parts share the same text


@dataclass
class ScoreData:
    """Extracted text data from a choral score."""

    title: str = ""
    composer: str = ""
    edition: str = ""
    language: str = "de"
    sections: list[Section] = field(default_factory=list)


@dataclass
class TranscribedVoicePart:
    """A voice part with both original text and IPA transcription."""

    name: str
    text: str
    ipa: str


@dataclass
class TranscribedSection:
    """A section with IPA transcriptions for each voice part."""

    name: str
    voice_parts: list[TranscribedVoicePart] = field(default_factory=list)
    all_parts_same: bool = False


@dataclass
class TranscribedScore:
    """Full score with IPA transcriptions."""

    title: str = ""
    composer: str = ""
    edition: str = ""
    language: str = "de"
    sections: list[TranscribedSection] = field(default_factory=list)
