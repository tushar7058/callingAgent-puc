from dataclasses import dataclass

from livekit.agents import stt, tts
from livekit.plugins import deepgram, elevenlabs, google


@dataclass(frozen=True)
class LanguageProfile:
    label: str
    code: str


DEFAULT_LANGUAGE_PROFILE = LanguageProfile(label="English", code="en-US")
ELEVENLABS_TTS_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
GOOGLE_TTS_VOICE_NAME = "Sulafat"

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
        LanguageProfile(
            label=preferred_language.strip(), code=DEFAULT_LANGUAGE_PROFILE.code
        ),
    )


def _is_english(profile: LanguageProfile) -> bool:
    return profile.code.lower().startswith("en")


def build_stt(profile: LanguageProfile):
    languages = [profile.code]
    if profile.code != DEFAULT_LANGUAGE_PROFILE.code:
        languages.append(DEFAULT_LANGUAGE_PROFILE.code)

    return stt.FallbackAdapter(
        [
            google.STT(
                model="latest_long",
                languages=languages,
                detect_language=True,
                spoken_punctuation=False,
                punctuate=True,
                interim_results=True,
                location="global",
            ),
            deepgram.STT(
                model="nova-3",
                language=profile.code,
                detect_language=True,
                interim_results=True,
                punctuate=True,
            ),
            elevenlabs.STT(
                model_id="scribe_v2_realtime",
                language_code=profile.code,
                use_realtime=True,
            ),
        ],
        attempt_timeout=10.0,
        max_retry_per_stt=1,
    )


def build_tts(profile: LanguageProfile):
    fallback_tts = [
        google.TTS(
            model_name="gemini-2.5-flash-lite-preview-tts",
            gender="female",
            voice_name=GOOGLE_TTS_VOICE_NAME,
            language=profile.code,
            use_streaming=True,
        ),
        elevenlabs.TTS(
            model="eleven_turbo_v2_5",
            voice_id=ELEVENLABS_TTS_VOICE_ID,
            language=profile.code,
        ),
    ]

    if _is_english(profile):
        fallback_tts.append(deepgram.TTS(model="aura-2-asteria-en"))

    return tts.FallbackAdapter(
        fallback_tts,
        max_retry_per_tts=2,
    )
