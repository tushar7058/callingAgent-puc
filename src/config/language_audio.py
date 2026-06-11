import os
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import stt, tts
from livekit.plugins import cartesia, deepgram, elevenlabs, google, openai

load_dotenv()


@dataclass(frozen=True)
class LanguageProfile:
    label: str
    code: str


DEFAULT_LANGUAGE_PROFILE = LanguageProfile(label="English", code="en-US")
ELEVENLABS_TTS_VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID") or os.getenv(
    "ELEVELABS_TTS_VOICE_ID"
)
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME")


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


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_stt(profile: LanguageProfile):
    languages = [profile.code]
    if profile.code != DEFAULT_LANGUAGE_PROFILE.code:
        languages.append(DEFAULT_LANGUAGE_PROFILE.code)

    return stt.FallbackAdapter(
        [
            deepgram.STT(
                model=os.getenv("DEEPGRAM_STT_MODEL"),
                language=profile.code,
                interim_results=True,
                punctuate=True,
            ),
            google.STT(
                model=os.getenv("GOOGLE_STT_MODEL"),
                languages=languages,
                detect_language=True,
                spoken_punctuation=False,
                punctuate=True,
                interim_results=True,
                location="global",
            ),
        ],
        attempt_timeout=10.0,
        max_retry_per_stt=1,
    )


def build_tts(profile: LanguageProfile):
    fallback_tts = [
        elevenlabs.TTS(
            model=os.getenv("ELEVENLABS_TTS_MODEL"),
            voice_id=ELEVENLABS_TTS_VOICE_ID,
            language=profile.code,
        ),
        google.TTS(
            model_name=os.getenv("GOOGLE_TTS_MODEL"),
            gender="female",
            voice_name=GOOGLE_TTS_VOICE_NAME,
            language=profile.code,
            use_streaming=True,
        ),
        openai.TTS(
            model=os.getenv("OPENAI_TTS_MODEL"),
            voice=os.getenv("OPENAI_TTS_VOICE"),
        ),
    ]
    if _env_flag_enabled("ENABLE_CARTESIA_TTS"):
        fallback_tts.append(
            cartesia.TTS(
                model=os.getenv("CARTESIA_TTS_MODEL"),
                language=profile.code,
                voice=os.getenv("CARTESIA_TTS_VOICE"),
                speed=1.0,
                emotion="Excited",
            )
        )
    if _is_english(profile):
        fallback_tts.append(deepgram.TTS(model=os.getenv("DEEPGRAM_TTS_MODEL")))

    return tts.FallbackAdapter(
        fallback_tts,
        max_retry_per_tts=2,
    )
