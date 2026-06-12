from __future__ import annotations

from agent import build_llm, build_turn_handling_options


class _FallbackSpy:
    def __init__(self, instances, **kwargs):
        self.instances = instances
        self.kwargs = kwargs


def test_turn_handling_enables_preemptive_generation_for_lower_latency():
    options = build_turn_handling_options(turn_detection="vad")

    assert options["preemptive_generation"] == {
        "enabled": True,
        "preemptive_tts": False,
    }


def test_turn_handling_uses_bilingual_safe_endpointing_and_auto_interruptions():
    options = build_turn_handling_options(turn_detection="vad")

    assert options["endpointing"] == {
        "mode": "fixed",
        "min_delay": 0.5,
        "max_delay": 2.0,
    }
    assert options["interruption"] == {
        "resume_false_interruption": True,
        "false_interruption_timeout": 1.5,
    }
    assert "mode" not in options["interruption"]
    assert "min_words" not in options["interruption"]


def test_turn_handling_caps_long_user_turns():
    options = build_turn_handling_options(turn_detection="vad")

    assert options["user_turn_limit"] == {
        "max_duration": 30.0,
    }


def test_llm_fallbacks_cap_generated_tokens(monkeypatch):
    created = []

    def recorder(provider):
        def create(**kwargs):
            created.append((provider, kwargs))
            return provider

        return create

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3-32b")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setattr("agent.openai.LLM", recorder("openai"))
    monkeypatch.setattr("agent.google.LLM", recorder("google"))
    monkeypatch.setattr("agent.groq.LLM", recorder("groq"))
    monkeypatch.setattr("agent.llm.FallbackAdapter", _FallbackSpy)

    fallback = build_llm()

    assert fallback.instances == ["openai", "google", "groq"]
    assert fallback.kwargs == {"attempt_timeout": 15.0}
    assert created[0][1]["max_completion_tokens"] == 200
    assert created[1][1]["max_output_tokens"] == 200
    assert created[2][1]["max_completion_tokens"] == 200
