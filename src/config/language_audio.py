from __future__ import annotations

from dataclasses import dataclass

from livekit.plugins import google
from livekit.plugins import elevenlabs

@dataclass(frozen=True)
class LanguageProfile:
    label: str
    code: str


DEFAULT_LANGUAGE_PROFILE = LanguageProfile(label="English", code="en-US")

_LANGUAGE_ALIASES: dict[str, LanguageProfile] = {
    "english": LanguageProfile(label="English", code="en-US"),
    "en": LanguageProfile(label="English", code="en-US"),
    "en-us": LanguageProfile(label="English", code="en-US"),
    "arabic": LanguageProfile(label="Arabic", code="ar"),
    "ar": LanguageProfile(label="Arabic", code="ar"),
}


def resolve_language_profile(preferred_language: str | None) -> LanguageProfile:
    if not preferred_language:
        return DEFAULT_LANGUAGE_PROFILE

    normalized = preferred_language.strip().lower()
    if not normalized:
        return DEFAULT_LANGUAGE_PROFILE

    return _LANGUAGE_ALIASES.get(
        normalized,
        LanguageProfile(label=preferred_language.strip(), code=DEFAULT_LANGUAGE_PROFILE.code),
    )


def build_stt(profile: LanguageProfile):
    languages = [profile.code]
    if profile.code != DEFAULT_LANGUAGE_PROFILE.code:
        languages.append(DEFAULT_LANGUAGE_PROFILE.code)

    return google.STT(
        model="latest_long",
        languages=languages,
        detect_language=True,
        spoken_punctuation=False,
        punctuate=True,
        interim_results=True,
        location="global",
    )

def build_tts(profile: LanguageProfile):
    return google.TTS(
        model_name="gemini-2.5-flash-lite-preview-tts",
        gender="female",
        voice_name="Achernar",
        language=profile.code,
        use_streaming=True,
    )

