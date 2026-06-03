from livekit.agents import AgentTask, RunContext, function_tool, get_job_context

from models.data_collection import (
    AccommodationPreferenceResults,
    ActivityPreferenceResults,
    BudgetResults,
    ContactInfoResults,
    CustomerInfoResults,
    LanguagePreferenceResults,
    TravelPlanResults,
    TravelPreferenceResults,
)


def _build_task_instructions(agent_instructions: str, task_instructions: str, is_first: bool = False):
    prefix = "" if is_first else "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
    suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
    return prefix + agent_instructions + "\n" + task_instructions + suffix

class CustomerInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Collect the customer's full name and how many people are traveling.", is_first=True),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_customer_info")
    async def record_customer_info(self, context: RunContext, full_name: str, number_of_travelers: int | None = None):
        """Call when you have collected the customer's name and number of travelers."""
        self.complete(CustomerInfoResults(full_name=full_name, number_of_travelers=number_of_travelers))

class LanguagePreferenceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Ask which language the customer prefers for the conversation.\n- Offer only two options: English or Arabic.\n- IMPORTANT: Ask this question bilingually in both English and Arabic (e.g., 'Would you prefer to continue in English, or Arabic? هل تفضل الاستمرار باللغة الإنجليزية أم العربية؟').\n- Record the exact chosen option using the tool.",
                is_first=False,
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_language_preference")
    async def record_language_preference(self, context: RunContext, preferred_language: str):
        """Call when you have collected the customer's preferred language."""
        get_job_context().proc.userdata["preferred_language"] = preferred_language
        self.complete(LanguagePreferenceResults(preferred_language=preferred_language))

class TravelPlanTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Find out the travel source and destination and preferred travel dates or duration."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_travel_plan")
    async def record_travel_plan(self, context: RunContext, source: str, destination: str, travel_dates: str | None = None):
        """Call when you have collected the source, destination, and dates."""
        self.complete(TravelPlanResults(source=source, destination=destination, travel_dates=travel_dates))

class BudgetTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Ask for their overall budget for the trip."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_budget")
    async def record_budget(self, context: RunContext, budget: str):
        """Call when you have collected the budget."""
        self.complete(BudgetResults(budget=budget))

class TravelPreferenceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Ask for their preferred mode of transportation (e.g., flight, train, bus)."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_travel_preference")
    async def record_travel_preference(self, context: RunContext, transportation_mode: str):
        """Call when you have collected the transportation preference."""
        self.complete(TravelPreferenceResults(transportation_mode=transportation_mode))

class AccommodationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Ask for accommodation preferences (e.g., 5-star, 4-star, villa)."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_accommodation")
    async def record_accommodation(self, context: RunContext, accommodation_type: str):
        """Call when you have collected the accommodation preference."""
        self.complete(AccommodationPreferenceResults(accommodation_type=accommodation_type))

class ActivityTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Ask what kind of activities they want to do (e.g., adventure, sightseeing, relaxing)."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_activity")
    async def record_activity(self, context: RunContext, activities: str):
        """Call when you have collected the activity preferences."""
        self.complete(ActivityPreferenceResults(activities=activities))

class ContactInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(agent_instructions, "- Collect the customer's contact information (email and phone number)."),
            tools=list(extra_tools) if extra_tools else [],
        )
    async def on_enter(self):
        await self.session.generate_reply(instructions="Begin this task now.", allow_interruptions=True, tool_choice="auto")

    @function_tool(name="record_contact_info")
    async def record_contact_info(self, context: RunContext, phone_number: str | None = None, email_address: str | None = None):
        """Call when you have collected contact info."""
        self.complete(ContactInfoResults(phone_number=phone_number, email_address=email_address))
