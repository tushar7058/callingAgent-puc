from __future__ import annotations

from config import language_audio
from config.language_audio import LanguageProfile, build_stt, build_tts


class _FallbackSpy:
    def __init__(self, instances, **kwargs):
        self.instances = instances
        self.kwargs = kwargs


def test_build_stt_uses_google_primary_with_deepgram_and_elevenlabs_fallbacks(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setattr(language_audio.google, "STT", recorder("google"))
    monkeypatch.setattr(language_audio.deepgram, "STT", recorder("deepgram"))
    monkeypatch.setattr(language_audio.elevenlabs, "STT", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.stt, "FallbackAdapter", _FallbackSpy)

    fallback = build_stt(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["google", "deepgram", "elevenlabs"]
    assert fallback.kwargs == {"attempt_timeout": 10.0, "max_retry_per_stt": 1}
    assert created[0] == (
        "google",
        {
            "model": "latest_long",
            "languages": ["ar", "en-US"],
            "detect_language": True,
            "spoken_punctuation": False,
            "punctuate": True,
            "interim_results": True,
            "location": "global",
        },
    )
    assert created[1][0] == "deepgram"
    assert created[1][1]["model"] == "nova-3"
    assert created[1][1]["language"] == "ar"
    assert created[1][1]["detect_language"] is True
    assert created[2] == (
        "elevenlabs",
        {
            "model_id": "scribe_v2_realtime",
            "language_code": "ar",
            "use_realtime": True,
        },
    )


def test_build_tts_uses_google_primary_with_elevenlabs_and_english_deepgram_fallbacks(
    monkeypatch,
):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setattr(language_audio.google, "TTS", recorder("google"))
    monkeypatch.setattr(language_audio.elevenlabs, "TTS", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.deepgram, "TTS", recorder("deepgram"))
    monkeypatch.setattr(language_audio.tts, "FallbackAdapter", _FallbackSpy)

    fallback = build_tts(LanguageProfile(label="English", code="en-US"))

    assert fallback.instances == ["google", "elevenlabs", "deepgram"]
    assert fallback.kwargs == {"max_retry_per_tts": 2}
    assert created[0][0] == "google"
    assert created[0][1]["language"] == "en-US"
    assert created[1] == (
        "elevenlabs",
        {
            "model": "eleven_turbo_v2_5",
            "voice_id": "Xb7hH8MSUJpSbSDYk0k2",
            "language": "en-US",
        },
    )
    assert created[2] == ("deepgram", {"model": "aura-2-asteria-en"})


def test_build_tts_skips_deepgram_tts_for_arabic(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setattr(language_audio.google, "TTS", recorder("google"))
    monkeypatch.setattr(language_audio.elevenlabs, "TTS", recorder("elevenlabs"))
    monkeypatch.setattr(language_audio.deepgram, "TTS", recorder("deepgram"))
    monkeypatch.setattr(language_audio.tts, "FallbackAdapter", _FallbackSpy)

    fallback = build_tts(LanguageProfile(label="Arabic", code="ar"))

    assert fallback.instances == ["google", "elevenlabs"]
    assert [provider for provider, _ in created] == ["google", "elevenlabs"]
