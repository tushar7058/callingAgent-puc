from __future__ import annotations

from agent import build_turn_handling_options


def test_turn_handling_disables_preemptive_generation_for_lower_jitter():
    options = build_turn_handling_options(turn_detection="vad")

    assert options["preemptive_generation"] == {
        "enabled": False,
        "preemptive_tts": False,
    }


def test_turn_handling_uses_snappier_endpointing_and_adaptive_interruptions():
    options = build_turn_handling_options(turn_detection="vad")

    assert options["endpointing"] == {
        "mode": "fixed",
        "min_delay": 0.35,
        "max_delay": 2.0,
    }
    assert options["interruption"] == {
        "mode": "adaptive",
        "min_duration": 0.35,
        "min_words": 0,
        "resume_false_interruption": True,
        "false_interruption_timeout": 1.5,
    }
