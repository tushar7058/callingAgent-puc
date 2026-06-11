from __future__ import annotations

from agent import build_turn_handling_options


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
