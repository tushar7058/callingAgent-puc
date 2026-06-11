from __future__ import annotations

from config import language_audio
from config.language_audio import LanguageProfile, build_stt, build_tts


class _FallbackSpy:
    def __init__(self, instances, **kwargs):
        self.instances = instances
        self.kwargs = kwargs


def test_build_stt_uses_openai_primary_with_deepgram_google_and_elevenlabs_fallbacks(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    monkeypatch.setenv("DEEPGRAM_STT_MODEL", "nova-3")
    monkeypatch.setenv("GOOGLE_STT_MODEL", "telephony")
    monkeypatch.setenv("ELEVENLABS_STT_MODEL", "scribe_v2_realtime")
    monkeypatch.setattr(language_audio.openai, "STT", recorder("openai"))
    monkeypatch.setattr(language_audio.google, "STT", recorder("google"))
    monkeypatch.setattr(language_audio.deepgram, "STT", recorder("deepgram"))
    monkeypatch.setattr(language_audio.elevenlabs, "STT", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.stt, "FallbackAdapter", _FallbackSpy)

    fallback = build_stt(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["openai", "deepgram", "google", "elevenlabs"]
    assert fallback.kwargs == {"attempt_timeout": 10.0, "max_retry_per_stt": 1}
    assert created[0] == (
        "openai",
        {
            "model": "gpt-4o-mini-transcribe",
            "language": "ar",
            "use_realtime": True,
        },
    )
    assert created[1] == (
        "deepgram",
        {
            "model": "nova-3",
            "language": "ar",
            "interim_results": True,
            "punctuate": True,
        },
    )
    assert created[2] == (
        "google",
        {
            "model": "telephony",
            "languages": ["ar", "en-US"],
            "detect_language": True,
            "spoken_punctuation": False,
            "punctuate": True,
            "interim_results": True,
            "location": "global",
        },
    )
    assert created[3] == (
        "elevenlabs",
        {
            "model_id": "scribe_v2_realtime",
            "language_code": "ar",
            "use_realtime": True,
        },
    )


def test_build_tts_uses_cartesia_primary_with_provider_fallbacks_for_english(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("CARTESIA_TTS_MODEL", "sonic")
    monkeypatch.setenv("CARTESIA_TTS_VOICE", "cartesia-voice")
    monkeypatch.setenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    monkeypatch.setenv("GOOGLE_TTS_MODEL", "gemini-2.5-flash-lite-preview-tts")
    monkeypatch.setenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    monkeypatch.setenv("OPENAI_TTS_VOICE", "alloy")
    monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-2-asteria-en")
    monkeypatch.setattr(language_audio, "ELEVENLABS_TTS_VOICE_ID", "eleven-voice")
    monkeypatch.setattr(language_audio, "GOOGLE_TTS_VOICE_NAME", "Sulafat")
    monkeypatch.setattr(language_audio.cartesia, "TTS", recorder("cartesia"))
    monkeypatch.setattr(language_audio.google, "TTS", recorder("google"))
    monkeypatch.setattr(language_audio.elevenlabs, "TTS", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.deepgram, "TTS", recorder("deepgram"))
    monkeypatch.setattr(language_audio.openai, "TTS", recorder("openai"))
    monkeypatch.setattr(language_audio.tts, "FallbackAdapter", _FallbackSpy)

    fallback = build_tts(LanguageProfile(label="English", code="en-US"))

    assert fallback.instances == [
        "cartesia",
        "elevenlabs",
        "google",
        "openai",
        "deepgram",
    ]
    assert fallback.kwargs == {"max_retry_per_tts": 2}
    assert created[0] == (
        "cartesia",
        {
            "model": "sonic",
            "language": "en-US",
            "voice": "cartesia-voice",
            "speed": 1.0,
            "emotion": "Excited",
        },
    )
    assert created[1] == (
        "elevenlabs",
        {
            "model": "eleven_multilingual_v2",
            "voice_id": "eleven-voice",
            "language": "en-US",
        },
    )
    assert created[2] == (
        "google",
        {
            "model_name": "gemini-2.5-flash-lite-preview-tts",
            "gender": "female",
            "voice_name": "Sulafat",
            "language": "en-US",
            "use_streaming": True,
        },
    )
    assert created[3] == (
        "openai",
        {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
        },
    )
    assert created[4] == ("deepgram", {"model": "aura-2-asteria-en"})


def test_build_tts_skips_deepgram_tts_for_arabic(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("CARTESIA_TTS_MODEL", "sonic")
    monkeypatch.setenv("CARTESIA_TTS_VOICE", "cartesia-voice")
    monkeypatch.setenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    monkeypatch.setenv("GOOGLE_TTS_MODEL", "gemini-2.5-flash-lite-preview-tts")
    monkeypatch.setenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    monkeypatch.setenv("OPENAI_TTS_VOICE", "alloy")
    monkeypatch.setattr(language_audio, "ELEVENLABS_TTS_VOICE_ID", "eleven-voice")
    monkeypatch.setattr(language_audio, "GOOGLE_TTS_VOICE_NAME", "Sulafat")
    monkeypatch.setattr(language_audio.cartesia, "TTS", recorder("cartesia"))
    monkeypatch.setattr(language_audio.google, "TTS", recorder("google"))
    monkeypatch.setattr(language_audio.elevenlabs, "TTS", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.deepgram, "TTS", recorder("deepgram"))
    monkeypatch.setattr(language_audio.openai, "TTS", recorder("openai"))
    monkeypatch.setattr(language_audio.tts, "FallbackAdapter", _FallbackSpy)

    fallback = build_tts(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["cartesia", "elevenlabs", "google", "openai"]
    assert [provider for provider, _ in created] == [
        "cartesia",
        "elevenlabs",
        "google",
        "openai",
    ]
