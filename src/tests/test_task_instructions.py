from __future__ import annotations

from tasks.data_collection import AccommodationTask, ActivityTask, TravelPreferenceTask


def test_option_tasks_record_selection_and_do_not_end_call():
    tasks = [
        TravelPreferenceTask(agent_instructions="base"),
        AccommodationTask(agent_instructions="base"),
        ActivityTask(agent_instructions="base"),
    ]

    for task in tasks:
        instructions = task.instructions
        assert "prefer to shortlist" in instructions
        assert "Do not end the call" in instructions
        assert "prefer to book" not in instructions

    assert "call record_travel_preference" in tasks[0].instructions
    assert "provider name" in tasks[0].instructions
    assert "departure time" in tasks[0].instructions
    assert "call record_accommodation" in tasks[1].instructions
    assert "hotel rating" in tasks[1].instructions
    assert "room type" in tasks[1].instructions
    assert "call record_activity" in tasks[2].instructions
    assert "activity category" in tasks[2].instructions
    assert "tour type" in tasks[2].instructions
