import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import stt, tts
from livekit.plugins import cartesia, deepgram, elevenlabs, google, openai

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageProfile:
    label: str
    code: str


DEFAULT_LANGUAGE_PROFILE = LanguageProfile(label="English", code="en-US")
ELEVENLABS_TTS_VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID") or os.getenv(
    "ELEVELABS_TTS_VOICE_ID"
)
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME")
GOOGLE_STT_MISSING_CREDENTIALS_ERROR = (
    "Application default credentials must be available"
)


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

    providers = [
        openai.STT(
            model=os.getenv("OPENAI_STT_MODEL"),
            language=profile.code,
            detect_language=True,
            use_realtime=True,
        ),
        deepgram.STT(
            model=os.getenv("DEEPGRAM_STT_MODEL"),
            language=profile.code,
            interim_results=True,
            punctuate=True,
        ),
    ]
    try:
        providers.append(
            google.STT(
                model=os.getenv("GOOGLE_STT_MODEL"),
                languages=languages,
                detect_language=True,
                spoken_punctuation=False,
                punctuate=True,
                interim_results=True,
                location="global",
            )
        )
    except ValueError as exc:
        if GOOGLE_STT_MISSING_CREDENTIALS_ERROR not in str(exc):
            raise
        logger.warning("Google STT fallback disabled: missing application credentials")

    return stt.FallbackAdapter(
        providers,
        attempt_timeout=10.0,
        max_retry_per_stt=1,
    )


def build_tts(profile: LanguageProfile):
    fallback_tts = [
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
        cartesia.TTS(
            model=os.getenv("CARTESIA_TTS_MODEL"),
            language=profile.code,
            voice=os.getenv("CARTESIA_TTS_VOICE"),
            speed=0.8,
            emotion="Excited",
        ),
        elevenlabs.TTS(
            model=os.getenv("ELEVENLABS_TTS_MODEL"),
            voice_id=ELEVENLABS_TTS_VOICE_ID,
            language=profile.code,
            auto_mode=True,
        ),
    ]

    return tts.FallbackAdapter(
        fallback_tts,
        max_retry_per_tts=2,
    )
