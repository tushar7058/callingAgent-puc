from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agents.sales_agent import DefaultAgent, _without_end_call_tools
from models.data_collection import CustomerInfoResults


class DummyTool:
    def __init__(self, tool_id: str):
        self.id = tool_id


class DummySpeechHandle:
    def __init__(self, calls: list):
        self.calls = calls

    def __await__(self):
        async def done():
            self.calls.append(("speech_done",))

        return done().__await__()


class DummySession:
    def __init__(self):
        self.calls = []

    def say(self, text, **kwargs):
        self.calls.append(("say", text, kwargs))
        return DummySpeechHandle(self.calls)

    def shutdown(self, **kwargs):
        self.calls.append(("shutdown", kwargs))


class DummyJobContext:
    def __init__(self):
        self.deleted = False
        self.shutdown_reason = None
        self.proc = SimpleNamespace(userdata={})

    async def delete_room(self):
        self.deleted = True

    def shutdown(self, reason=""):
        self.shutdown_reason = reason


def test_without_end_call_tools_filters_by_tool_id():
    tools = [DummyTool("end_call"), DummyTool("best_TravelOption")]

    assert [tool.id for tool in _without_end_call_tools(tools)] == ["best_TravelOption"]


def test_default_agent_intermediate_task_tools_exclude_end_call():
    agent = DefaultAgent()

    assert any(tool.id == "end_call" for tool in agent.tools)
    assert all(tool.id != "end_call" for tool in _without_end_call_tools(agent.tools))


def test_default_agent_forces_short_voice_responses():
    instructions = DefaultAgent()._agent_instructions

    assert "maximum one short sentence" in instructions
    assert "Ask one question at a time" in instructions


def test_task_agent_instructions_do_not_duplicate_base_prompt():
    instructions = DefaultAgent()._build_task_agent_instructions("Arabic")

    assert instructions.count("You are AIRA") == 1
    assert "Use Arabic for this task" in instructions


async def test_on_exit_sends_partial_data_when_call_cuts_mid_collection(
    monkeypatch, tmp_path
):
    sent_payloads = []
    userdata = {
        "dc_external_sent": False,
        "dc_task_results": {
            "customer_info": CustomerInfoResults(
                full_name="Jane Doe",
                phone_number="12345",
            )
        },
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "agents.sales_agent.get_job_context",
        lambda: SimpleNamespace(proc=SimpleNamespace(userdata=userdata)),
    )
    monkeypatch.setattr(
        "agents.sales_agent.send_to_external_api",
        lambda payload: sent_payloads.append(payload),
    )

    await DefaultAgent().on_exit()

    assert userdata["dc_external_sent"] is True
    assert sent_payloads[0]["customer_info"]["full_name"] == "Jane Doe"
    assert sent_payloads[0]["customer_info"]["contact"]["phone_number"] == "12345"
    assert sent_payloads[0]["travel_plan"]["source"] is None


async def test_finish_closeout_says_goodbye_then_deletes_room(monkeypatch):
    session = DummySession()
    job_ctx = DummyJobContext()

    monkeypatch.setattr(DefaultAgent, "session", property(lambda self: session))
    monkeypatch.setattr("agents.sales_agent.get_job_context", lambda: job_ctx)

    await DefaultAgent()._say_goodbye_and_end_call("English")

    assert session.calls[0] == (
        "say",
        "Thank you, I have saved the details. The Mannai Travels team will contact you shortly to help with the next steps. Have a wonderful day.",
        {"allow_interruptions": False},
    )
    assert ("speech_done",) in session.calls
    assert job_ctx.deleted is True
    assert ("shutdown", {"drain": False}) in session.calls
    assert job_ctx.shutdown_reason == "data_collection_complete"


async def test_finish_closeout_uses_arabic_goodbye(monkeypatch):
    session = DummySession()
    job_ctx = DummyJobContext()

    monkeypatch.setattr(DefaultAgent, "session", property(lambda self: session))
    monkeypatch.setattr("agents.sales_agent.get_job_context", lambda: job_ctx)

    await DefaultAgent()._say_goodbye_and_end_call("Arabic")

    assert session.calls[0][0] == "say"
    assert "شكرًا لك" in session.calls[0][1]


def test_data_collection_task_group_disables_final_context_summary(monkeypatch):
    captured = {}

    class FakeTaskGroup:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    callback = object()

    monkeypatch.setattr("agents.sales_agent.TaskGroup", FakeTaskGroup)

    task_group = DefaultAgent()._create_data_collection_task_group(callback)

    assert isinstance(task_group, FakeTaskGroup)
    assert captured["on_task_completed"] is callback
    assert captured["summarize_chat_ctx"] is False


async def test_finish_data_collection_speaks_before_waiting_for_external_send(
    monkeypatch, tmp_path
):
    session = DummySession()
    job_ctx = DummyJobContext()
    events = []
    send_can_finish = asyncio.Event()

    async def fake_send_external(self, payload, status):
        events.append(("send_started", status))
        await send_can_finish.wait()
        events.append(("send_finished", status))

    def say(text, **kwargs):
        events.append(("say", text))
        send_can_finish.set()
        return DummySpeechHandle(session.calls)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DefaultAgent, "session", property(lambda self: session))
    monkeypatch.setattr("agents.sales_agent.get_job_context", lambda: job_ctx)
    monkeypatch.setattr(
        DefaultAgent, "_send_data_collection_external", fake_send_external
    )
    monkeypatch.setattr(session, "say", say)

    await DefaultAgent()._finish_data_collection({})

    event_names = [event[0] for event in events]

    assert event_names.index("say") < event_names.index("send_finished")
    assert ("send_finished", "complete") in events
    assert ("shutdown", {"drain": False}) in session.calls
