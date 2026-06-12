from __future__ import annotations

from tasks.data_collection import (
    TASK_START_INSTRUCTIONS,
    AccommodationTask,
    ActivityTask,
    CustomerInfoTask,
    TravelPlanTask,
    _start_task,
)


class _SpeechHandle:
    def __init__(self):
        self.awaited = False

    def __await__(self):
        self.awaited = True
        if False:
            yield
        return None


class _SessionSpy:
    def __init__(self):
        self.calls = []
        self.handle = _SpeechHandle()

    def generate_reply(self, **kwargs):
        self.calls.append(kwargs)
        return self.handle


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


async def test_travel_plan_skips_transportation_mode_for_international_trips():
    instructions = TravelPlanTask(agent_instructions="base").instructions

    assert "Do not ask for transportation mode for international trips" in instructions
    assert (
        "leave transportation_mode empty unless the caller volunteers it"
        in instructions
    )
    assert (
        "Only ask transportation mode for domestic or same-country trips"
        in instructions
    )


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


async def test_task_transitions_start_immediately_and_advance_after_recording():
    instructions = TravelPlanTask(agent_instructions="base").instructions

    assert "When this task starts, ask exactly one question to begin:" in instructions
    assert "Where will you be traveling from" in instructions
    assert "silently record them with the appropriate tool" in instructions
    assert "TaskGroup can immediately continue to the next task" in instructions

    assert (
        "Start the current data collection task immediately" in TASK_START_INSTRUCTIONS
    )
    assert "ask exactly one clear question" in TASK_START_INSTRUCTIONS
    assert "one short sentence" in TASK_START_INSTRUCTIONS
    assert "do not wait for the caller to ask what is next" in TASK_START_INSTRUCTIONS


async def test_customer_info_is_not_treated_as_initial_greeting_task():
    instructions = CustomerInfoTask(agent_instructions="base").instructions

    assert "The user has already been greeted" in instructions
    assert "Do not introduce yourself or say hello" in instructions


async def test_task_start_schedules_speech_without_awaiting_playout():
    session = _SessionSpy()

    await _start_task(session)

    assert session.calls == [
        {
            "instructions": TASK_START_INSTRUCTIONS,
            "allow_interruptions": True,
            "tool_choice": "none",
        }
    ]
    assert session.handle.awaited is False
