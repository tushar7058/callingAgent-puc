from livekit.agents import AgentTask, RunContext, function_tool, get_job_context

from config.language_audio import resolve_language_profile
from models.data_collection import (
    AccommodationPreferenceResults,
    BudgetResults,
    CustomerInfoResults,
    GreetingResults,
    LanguagePreferenceResults,
    TravelPlanResults,
)

TASK_START_INSTRUCTIONS = (
    "Start the current data collection task immediately. Acknowledge any previous "
    "answer in at most five words, then ask exactly one clear question for this "
    "task in one short sentence. Do not recap, do not say goodbye, do not wait "
    "for the caller to ask what is next, and do not mention internal task names "
    "or tools."
)


def _build_task_instructions(
    agent_instructions: str,
    task_instructions: str,
    *,
    first_question: str,
):
    first_question_instruction = (
        f"\nWhen this task starts, ask exactly one question to begin: {first_question}"
    )
    suffix = "\nIMPORTANT: Only collect the information for THIS specific task. As soon as all required details for this task are available, silently record them with the appropriate tool so the TaskGroup can immediately continue to the next task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
    return (
        agent_instructions
        + "\n"
        + task_instructions
        + first_question_instruction
        + suffix
    )


async def _start_task(session) -> None:
    session.generate_reply(
        instructions=TASK_START_INSTRUCTIONS,
        allow_interruptions=True,
        tool_choice="none",
    )


class GreetingTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=agent_instructions + "\n" + "Welcome the caller bilingually. First say 'Welcome to Mannai Travels! I am AIRA, your AI travel assistant.' Then immediately say the Arabic equivalent (e.g., 'أهلاً بك في سفريات المناعي! أنا آيرا، مساعدك الذكي للسفر.'). IMPORTANT: The greeting must be a statement only. Do NOT end with any question. Once you have greeted the user, immediately call the tool to finish the greeting.",
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Start the conversation by greeting the user as instructed. Do not wait for them to speak first.",
            allow_interruptions=True,
            tool_choice="none",
        )

    @function_tool(name="finish_greeting")
    async def finish_greeting(self, context: RunContext):
        """Call this tool immediately after you have successfully greeted the user bilingually."""
        self.complete(GreetingResults())


class CustomerInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "You must collect four mandatory fields: the customer’s full name, total travelers, email address, and mobile number. Ask for these fields if they are not provided. Do NOT call the record tool until you have collected all four fields. Confirm the phone number with the user, don't rely on SIP. You may capture traveler split (adults, kids, seniors) only if volunteered.",
                first_question="May I have your full name and how many travelers will be going?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_customer_info")
    async def record_customer_info(
        self,
        context: RunContext,
        full_name: str,
        phone_number: str,
        email_address: str,
        total_travelers: int,
        adults: int | None = None,
        kids: int | None = None,
        senior_citizens: int | None = None,
    ):
        """Call when you have collected customer identity, contact, and travel group information."""
        if phone_number is None:
            try:
                phone_number = get_job_context().proc.userdata.get("sip_phone_number")
            except RuntimeError:
                phone_number = None

        self.complete(
            CustomerInfoResults(
                full_name=full_name,
                phone_number=phone_number,
                email_address=email_address,
                total_travelers=total_travelers,
                adults=adults,
                kids=kids,
                senior_citizens=senior_citizens,
            )
        )


class LanguagePreferenceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Ask which language the customer prefers for the conversation.\n- Offer only two options: English or Arabic.\n- IMPORTANT: Ask this question bilingually in both English and Arabic (e.g., 'Would you prefer to continue in English, or Arabic? هل تفضل الاستمرار باللغة الإنجليزية أم العربية؟').\n- Record the exact chosen option using the tool.",
                first_question="Would you prefer to continue in English, or Arabic? هل تفضل الاستمرار باللغة الإنجليزية أم العربية؟",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_language_preference")
    async def record_language_preference(
        self, context: RunContext, preferred_language: str
    ):
        """Call when you have collected the customer's preferred language."""
        profile = resolve_language_profile(preferred_language)
        get_job_context().proc.userdata["preferred_language"] = profile.label
        get_job_context().proc.userdata["preferred_language_code"] = profile.code
        self.complete(
            LanguagePreferenceResults(
                preferred_language=profile.label,
                language_code=profile.code,
            )
        )


class TravelPlanTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the travel source, destination, destination type or trip style, whether they want activities as yes or no, start date, end date, whether dates are flexible, transportation mode, and any special assistance needs.\n- Ask for transportation mode. For international trips, offer only flight as the transportation option. Do not offer flight for domestic or same-country trips unless the caller volunteers it.\n- For special assistance, capture senior support, kids support, and accessibility needs only when volunteered or naturally relevant.\n- Record dates as YYYY-MM-DD when the user provides exact dates. If dates are flexible, set flexible to true and leave unknown exact dates empty.",
                first_question="Where will you be traveling from, and what destination are you planning to visit?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_travel_plan")
    async def record_travel_plan(
        self,
        context: RunContext,
        source: str,
        destination: str,
        destination_type: str | None = None,
        activities: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        flexible: bool | None = None,
        transportation_mode: str | None = None,
        senior_support: bool | None = None,
        kids_support: bool | None = None,
        accessibility: str | None = None,
    ):
        """Call when you have collected travel plan fields."""
        self.complete(
            TravelPlanResults(
                source=source,
                destination=destination,
                destination_type=destination_type,
                activities=activities,
                start_date=start_date,
                end_date=end_date,
                flexible=flexible,
                transportation_mode=transportation_mode,
                senior_support=senior_support,
                kids_support=kids_support,
                accessibility=accessibility,
            )
        )


class BudgetTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect only the budget range if available and whether the budget is flexible.",
                first_question="What overall budget would you like us to plan around?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_budget")
    async def record_budget(
        self,
        context: RunContext,
        budget_range: str | None = None,
        flexibility: bool | None = None,
    ):
        """Call when you have collected budget fields."""
        self.complete(
            BudgetResults(
                budget_range=budget_range,
                flexibility=flexibility,
            )
        )


class AccommodationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect only accommodation type and hotel rating.\n- Do not ask about amenities, room type, hotel names, live availability, or prices.",
                first_question="What accommodation type and hotel rating would you prefer?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_accommodation")
    async def record_accommodation(
        self,
        context: RunContext,
        accommodation_type: str | None = None,
        hotel_rating: str | None = None,
    ):
        """Call when you have collected accommodation preference fields."""
        self.complete(
            AccommodationPreferenceResults(
                accommodation_type=accommodation_type,
                hotel_rating=hotel_rating,
            )
        )

