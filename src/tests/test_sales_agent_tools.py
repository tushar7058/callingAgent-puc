from __future__ import annotations

from agents.sales_agent import DefaultAgent, _without_end_call_tools


class DummyTool:
    def __init__(self, tool_id: str):
        self.id = tool_id


def test_without_end_call_tools_filters_by_tool_id():
    tools = [DummyTool("end_call"), DummyTool("best_TravelOption")]

    assert [tool.id for tool in _without_end_call_tools(tools)] == ["best_TravelOption"]


def test_default_agent_intermediate_task_tools_exclude_end_call():
    agent = DefaultAgent()

    assert any(tool.id == "end_call" for tool in agent.tools)
    assert all(tool.id != "end_call" for tool in _without_end_call_tools(agent.tools))
