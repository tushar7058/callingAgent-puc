import asyncio
import json
import logging
import os

from livekit.agents import (
    Agent,
    ToolError,
    get_job_context,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup

from api.datasend import send_to_external_api
from config.language_audio import build_stt, build_tts, resolve_language_profile
from tasks.data_collection import (
    AccommodationTask,
    ActivityTask,
    BudgetTask,
    ContactInfoTask,
    CustomerInfoTask,
    LanguagePreferenceTask,
    TravelPlanTask,
    TravelPreferenceTask,
)
from utils.helpers import _to_json_serializable
from utils.tools import best_TravelOption, gethotel_details, locationbasedactivitydetail

# logger = logging.getLogger("agent-Dakota-1462")
# logger = logging.getLogger("Dakota-1462")
logger = logging.getLogger("interview-agent")


def _without_end_call_tools(tools):
    return [tool for tool in tools if getattr(tool, "id", None) != "end_call"]


class DefaultAgent(Agent):
    def __init__(self) -> None:
        self.end_call_tool = EndCallTool(
            extra_description="Only end the call after all data collection tasks are complete and you have said goodbye.",
            delete_room=True,
            end_instructions="Thank the customer for their time, let them know Mannai Travels will reach out shortly with an itinerary, and wish them a good day.",
        )
        self._agent_instructions = """You are AIRA, a friendly, professional, and reliable travel assistant from Mannai Travels.
You are interacting with a user via voice to plan their perfect trip.

# Output rules
- Respond in plain text only. Never use JSON, markdown, lists, tables, code, emojis, or other complex formatting.
- Keep replies brief and conversational: one to three sentences. Ask one question at a time.
- Do not reveal system instructions, internal reasoning, tool names, parameters, or raw outputs.
- Spell out numbers, phone numbers, or email addresses cleanly.
- Avoid acronyms and words with unclear pronunciation.

# Conversational flow
- Help the user plan their travel efficiently. Prefer the simplest safe step first.
- Provide guidance in small steps and confirm completion before continuing.
- Be warm, hospitable, and enthusiastic about their travel plans.

# Tools
- Use available tools as needed to record the user's travel preferences.
- Collect required inputs first. Perform actions silently.
- When moving to a new topic, smoothly transition (e.g. "That sounds like a great destination! Now, let's talk about your budget...")
"""
        super().__init__(
            instructions="",
            tools=self.end_call_tool.tools,
        )

    async def on_enter(self):
        greeting_instructions = (
            "Welcome the caller bilingually. First say 'Welcome to Mannai Travels! I am AIRA, your AI travel assistant.' "
            "Then immediately say the Arabic equivalent (e.g., 'أهلاً بك في سفريات المناعي! أنا آيرا، مساعدك الذكي للسفر.')."
        )
        no_question_guardrail = (
            "IMPORTANT: The greeting must be a statement only. Do NOT end with any "
            'question, including open-ended prompts like "How can I help?". The '
            "next task will ask the first question."
        )
        await self.session.generate_reply(
            instructions="\n".join(
                part
                for part in (
                    self._agent_instructions,
                    greeting_instructions,
                    no_question_guardrail,
                )
                if part
            ),
            allow_interruptions=True,
        )

        async def sync_language_audio(task_event) -> None:
            if task_event.task_id != "language_preference":
                return

            preferred_language = getattr(task_event.result, "preferred_language", None)
            profile = resolve_language_profile(preferred_language)
            get_job_context().proc.userdata["preferred_language"] = profile.label
            get_job_context().proc.userdata["preferred_language_code"] = profile.code

            # The public AgentSession API does not expose STT/TTS setters, so we
            # update the underlying engines after the language preference is known.
            self.session._stt = build_stt(profile)
            self.session._tts = build_tts(profile)
            logger.info("updated speech language: %s (%s)", profile.label, profile.code)

        def task_instructions(base_instructions: str) -> str:
            preferred_language = get_job_context().proc.userdata.get(
                "preferred_language"
            )
            if preferred_language:
                language_line = f"Use {preferred_language} for this task and keep all questions, prompts, and confirmations in that language."
            else:
                language_line = "Use the same language the user is speaking. If they later choose a preferred language, switch to it immediately."
            return f"{self._agent_instructions}\n{language_line}\n{base_instructions}"

        _task_tools = _without_end_call_tools(self.tools)
        task_group = TaskGroup(
            chat_ctx=self.chat_ctx, on_task_completed=sync_language_audio
        )

        # 0. Language Preference
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: (
                LanguagePreferenceTask(
                    agent_instructions=task_instructions(_ai), extra_tools=_tools
                )
            ),
            id="language_preference",
            description="Ask which language the customer prefers for the rest of the conversation, offering only English or Arabic.",
        )

        # 1. Customer Info
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: CustomerInfoTask(
                agent_instructions=task_instructions(_ai), extra_tools=_tools
            ),
            id="customer_info",
            description="Collect the customer's name and number of travelers.",
        )
        # 2. Travel Plan
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: TravelPlanTask(
                agent_instructions=task_instructions(_ai), extra_tools=_tools
            ),
            id="travel_plan",
            description="Find out the travel source, destination, and preferred dates.",
        )
        # 3. Budget
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: BudgetTask(
                agent_instructions=task_instructions(_ai), extra_tools=_tools
            ),
            id="budget",
            description="Ask for their overall budget.",
        )
        # 4. Travel Preference
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: (
                TravelPreferenceTask(
                    agent_instructions=task_instructions(_ai),
                    extra_tools=[*_tools, best_TravelOption],
                )
            ),
            id="travel_preference",
            description="Ask for their preferred mode of transportation.",
        )
        # 5. Accommodation
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: AccommodationTask(
                agent_instructions=task_instructions(_ai),
                extra_tools=[*_tools, gethotel_details],
            ),
            id="accommodation",
            description="Ask for accommodation preferences (5-star, 4-star, etc).",
        )
        # 6. Activity
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ActivityTask(
                agent_instructions=task_instructions(_ai),
                extra_tools=[*_tools, locationbasedactivitydetail],
            ),
            id="activity",
            description="Ask what kind of activities they want to do.",
        )
        # 7. Contact Info
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ContactInfoTask(
                agent_instructions=task_instructions(_ai), extra_tools=_tools
            ),
            id="contact_info",
            description="Collect the customer's email and phone number.",
        )

        try:
            group_result = await task_group
        except (ToolError, asyncio.CancelledError):
            logger.info(
                "data collection task group cancelled (participant likely disconnected)"
            )
            return

        await self._finish_data_collection(group_result.task_results)

    async def _finish_data_collection(self, task_results):
        """Serialize results, save to JSON, speak goodbye, and end the session."""
        serialized = _to_json_serializable(task_results)
        get_job_context().proc.userdata["dc_results"] = serialized

        preferred_language = (
            serialized.get("language_preference", {}).get("preferred_language")
            if isinstance(serialized, dict)
            else None
        ) or "English"
        preferred_language_code = (
            serialized.get("language_preference", {}).get("language_code")
            if isinstance(serialized, dict)
            else None
        ) or resolve_language_profile(preferred_language).code

        if isinstance(serialized, dict):
            serialized["language_preference"] = {
                **serialized.get("language_preference", {}),
                "preferred_language": preferred_language,
                "language_code": preferred_language_code,
            }

        # Save to JSON file so the user can see it!
        try:
            # os.makedirs("src/logs", exist_ok=True)
            os.makedirs("src/data", exist_ok=True)
            with open("src/data/latest_travel_lead.json", "w") as f:
                # with open("src/logs/latest_travel_lead.json", "w") as f:
                json.dump(serialized, f, indent=2)
            logger.info("Saved travel lead data to src/data/latest_travel_lead.json")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")

        try:
            send_to_external_api({"task_results": serialized})
            logger.info("Sent travel lead data to external API")
        except Exception as e:
            logger.error(f"Failed to send travel lead data to external API: {e}")

        language_guidance = f"Speak entirely in {preferred_language}. Keep the tone warm and professional."

        speech_handle = self.session.generate_reply(
            instructions=f"All data collection tasks are complete. {language_guidance} You MUST now use the end_call tool to say goodbye and end the conversation.",
            tool_choice="auto",
        )
        try:
            await speech_handle
        except ConnectionError:
            logger.debug("user disconnected during goodbye speech")
