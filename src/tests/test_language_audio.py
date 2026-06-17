from __future__ import annotations

from config import language_audio
from config.language_audio import LanguageProfile, build_stt, build_tts


class _FallbackSpy:
    def __init__(self, instances, **kwargs):
        self.instances = instances
        self.kwargs = kwargs


def test_build_stt_uses_deepgram_primary_with_openai_and_google_fallbacks(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("DEEPGRAM_STT_MODEL", "nova-3")
    monkeypatch.setenv("OPENAI_STT_MODEL", "gpt-realtime-whisper")
    monkeypatch.setenv("GOOGLE_STT_MODEL", "telephony")
    monkeypatch.setattr(language_audio.openai, "STT", recorder("openai"))
    monkeypatch.setattr(language_audio.google, "STT", recorder("google"))
    monkeypatch.setattr(language_audio.deepgram, "STT", recorder("deepgram"))
    monkeypatch.setattr(language_audio.stt, "FallbackAdapter", _FallbackSpy)

    fallback = build_stt(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["deepgram", "openai", "google"]
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
        "openai",
        {
            "model": "gpt-realtime-whisper",
            "language": "ar",
            "detect_language": True,
            "use_realtime": True,
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


def test_build_stt_skips_google_fallback_when_credentials_are_missing(monkeypatch):
    created = []

    def deepgram_create(**kwargs):
        created.append(("deepgram", kwargs))
        return "deepgram"

    def openai_create(**kwargs):
        created.append(("openai", kwargs))
        return "openai"

    def google_create(**kwargs):
        created.append(("google", kwargs))
        raise ValueError("Application default credentials must be available")

    monkeypatch.setenv("DEEPGRAM_STT_MODEL", "nova-3")
    monkeypatch.setenv("OPENAI_STT_MODEL", "gpt-realtime-whisper")
    monkeypatch.setenv("GOOGLE_STT_MODEL", "telephony")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(language_audio.openai, "STT", openai_create)
    monkeypatch.setattr(language_audio.deepgram, "STT", deepgram_create)
    monkeypatch.setattr(language_audio.google, "STT", google_create)
    monkeypatch.setattr(language_audio.stt, "FallbackAdapter", _FallbackSpy)

    fallback = build_stt(LanguageProfile(label="English", code="en-US"))

    assert fallback.instances == ["deepgram", "openai"]
    assert fallback.kwargs == {"attempt_timeout": 10.0, "max_retry_per_stt": 1}
    assert [provider for provider, _ in created] == ["deepgram", "openai", "google"]


def test_build_stt_reraises_non_credential_google_errors(monkeypatch):
    def recorder(provider):
        def create(**kwargs):
            if provider == "google":
                raise ValueError("invalid google stt model")
            return provider

        return create

    monkeypatch.setattr(language_audio.deepgram, "STT", recorder("deepgram"))
    monkeypatch.setattr(language_audio.openai, "STT", recorder("openai"))
    monkeypatch.setattr(language_audio.google, "STT", recorder("google"))

    try:
        build_stt(LanguageProfile(label="English", code="en-US"))
    except ValueError as exc:
        assert str(exc) == "invalid google stt model"
    else:
        raise AssertionError("expected Google STT configuration error to be raised")


def test_build_tts_uses_requested_provider_order(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    monkeypatch.setenv("CARTESIA_TTS_MODEL", "sonic")
    monkeypatch.setenv("CARTESIA_TTS_VOICE", "cartesia-voice")
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

    fallback = build_tts(LanguageProfile(label="English", code="en-US"))

    assert fallback.instances == [
        "elevenlabs",
        "cartesia",
        "google",
        "openai",
    ]
    assert fallback.kwargs == {"max_retry_per_tts": 2}
    assert created[0] == (
        "elevenlabs",
        {
            "model": "eleven_multilingual_v2",
            "voice_id": "eleven-voice",
            "language": "en-US",
            "auto_mode": True,
        },
    )
    assert created[1] == (
        "cartesia",
        {
            "model": "sonic",
            "language": "en-US",
            "voice": "cartesia-voice",
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


def test_build_tts_skips_deepgram_tts_for_arabic(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    monkeypatch.setenv("CARTESIA_TTS_MODEL", "sonic")
    monkeypatch.setenv("CARTESIA_TTS_VOICE", "cartesia-voice")
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

    assert fallback.instances == ["elevenlabs", "cartesia", "google", "openai"]
    assert [provider for provider, _ in created] == [
        "elevenlabs",
        "cartesia",
        "google",
        "openai",
    ]
