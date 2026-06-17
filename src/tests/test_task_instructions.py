from __future__ import annotations

from tasks.data_collection import (
    TASK_START_INSTRUCTIONS,
    AccommodationTask,
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
    travel_plan_instructions = TravelPlanTask(agent_instructions="base").instructions
    assert "activities as yes or no" in travel_plan_instructions
    assert "special assistance" in travel_plan_instructions
    assert "senior support" in travel_plan_instructions

    accommodation_instructions = AccommodationTask(
        agent_instructions="base"
    ).instructions
    assert "accommodation type" in accommodation_instructions
    assert "hotel rating" in accommodation_instructions
    assert "breakfast included" not in accommodation_instructions
    assert "experience expectation" not in accommodation_instructions


async def test_travel_plan_offers_flight_only_for_international_trips():
    instructions = TravelPlanTask(agent_instructions="base").instructions

    assert "Ask for transportation mode" in instructions
    assert "For international trips, offer only flight" in instructions
    assert "Do not offer flight for domestic or same-country trips" in instructions


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


async def test_tasks_redirect_out_of_scope_requests_to_current_collection_task():
    instructions = TravelPlanTask(agent_instructions="base").instructions

    assert (
        "If the user asks for anything outside travel lead collection" in instructions
    )
    assert "politely decline in one short sentence" in instructions
    assert "return to the current question" in instructions


async def test_customer_info_is_not_treated_as_initial_greeting_task():
    instructions = CustomerInfoTask(agent_instructions="base").instructions

    assert "The user has already been greeted" in instructions
    assert "Do not introduce yourself or say hello" in instructions


async def test_customer_info_collects_only_name_and_traveler_count():
    instructions = CustomerInfoTask(agent_instructions="base").instructions

    assert "full name and total travelers" in instructions
    assert "Do not ask for phone number or email address" in instructions
    assert "record them immediately" in instructions


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
