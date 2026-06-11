from __future__ import annotations

from config import language_audio
from config.language_audio import LanguageProfile, build_stt, build_tts


class _FallbackSpy:
    def __init__(self, instances, **kwargs):
        self.instances = instances
        self.kwargs = kwargs


def test_build_stt_uses_deepgram_primary_with_google_fallback(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("DEEPGRAM_STT_MODEL", "nova-3")
    monkeypatch.setenv("GOOGLE_STT_MODEL", "telephony")
    monkeypatch.setattr(language_audio.google, "STT", recorder("google"))
    monkeypatch.setattr(language_audio.deepgram, "STT", recorder("deepgram"))
    monkeypatch.setattr(language_audio.stt, "FallbackAdapter", _FallbackSpy)

    fallback = build_stt(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["deepgram", "google"]
    assert fallback.kwargs == {"attempt_timeout": 10.0, "max_retry_per_stt": 1}
    assert created[0] == (
        "deepgram",
        {
            "model": "nova-3",
            "language": "ar",
            "interim_results": True,
            "punctuate": True,
        },
    )
    assert created[1] == (
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


def test_build_tts_uses_fast_provider_fallbacks_for_english_by_default(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

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
        "elevenlabs",
        "google",
        "openai",
        "deepgram",
    ]
    assert fallback.kwargs == {"max_retry_per_tts": 2}
    assert created[0] == (
        "elevenlabs",
        {
            "model": "eleven_multilingual_v2",
            "voice_id": "eleven-voice",
            "language": "en-US",
        },
    )
    assert created[1] == (
        "google",
        {
            "model_name": "gemini-2.5-flash-lite-preview-tts",
            "gender": "female",
            "voice_name": "Sulafat",
            "language": "en-US",
            "use_streaming": True,
        },
    )
    assert created[2] == (
        "openai",
        {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
        },
    )
    assert created[3] == ("deepgram", {"model": "aura-2-asteria-en"})


def test_build_tts_includes_cartesia_only_when_enabled(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("ENABLE_CARTESIA_TTS", "true")
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
    monkeypatch.setattr(language_audio.openai, "TTS", recorder("openai"))
    monkeypatch.setattr(language_audio.tts, "FallbackAdapter", _FallbackSpy)

    fallback = build_tts(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["elevenlabs", "google", "openai", "cartesia"]
    assert created[-1] == (
        "cartesia",
        {
            "model": "sonic",
            "language": "ar",
            "voice": "cartesia-voice",
            "speed": 1.0,
            "emotion": "Excited",
        },
    )


def test_build_tts_skips_deepgram_tts_for_arabic(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

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

    assert fallback.instances == ["elevenlabs", "google", "openai"]
    assert [provider for provider, _ in created] == [
        "elevenlabs",
        "google",
        "openai",
    ]
