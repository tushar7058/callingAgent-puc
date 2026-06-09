from __future__ import annotations

from tasks.data_collection import AccommodationTask, ActivityTask, TravelPlanTask


async def test_data_collection_tasks_do_not_recommend_or_search_live_options():
    tasks = [
        TravelPlanTask(agent_instructions="base"),
        AccommodationTask(agent_instructions="base"),
        ActivityTask(agent_instructions="base"),
    ]

    for task in tasks:
        instructions = task.instructions
        assert "Do NOT search live availability" in instructions
        assert "recommend options" in instructions
        assert "Only collect" in instructions
        assert "shortlist" not in instructions
        assert "best_TravelOption" not in instructions
        assert "gethotel_details" not in instructions
        assert "locationbasedactivitydetail" not in instructions


async def test_tasks_collect_target_schema_fields():
    assert (
        "transportation mode" in TravelPlanTask(agent_instructions="base").instructions
    )

    accommodation_instructions = AccommodationTask(
        agent_instructions="base"
    ).instructions
    assert "breakfast included" in accommodation_instructions
    assert "kids friendly" in accommodation_instructions
    assert "experience expectation" in accommodation_instructions

    activity_instructions = ActivityTask(agent_instructions="base").instructions
    assert "interest profile" in activity_instructions


async def test_tasks_instruct_model_to_record_values_in_selected_language():
    task = TravelPlanTask(
        agent_instructions=(
            "Use Arabic for this task and keep all questions, prompts, "
            "confirmations, and recorded user-facing JSON values in Arabic. "
            "Keep phone numbers, email addresses, ISO dates, booleans, and "
            "numeric counts unchanged."
        )
    )

    assert "recorded user-facing JSON values" in task.instructions
    assert "numeric counts unchanged" in task.instructions
