from livekit.agents import AgentTask, RunContext, function_tool, get_job_context

from config.language_audio import resolve_language_profile
from models.data_collection import (
    AccommodationPreferenceResults,
    ActivityPreferenceResults,
    BudgetResults,
    CustomerInfoResults,
    FoodPreferenceResults,
    LanguagePreferenceResults,
    SupportServicesResults,
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
    is_first: bool = False,
):
    prefix = (
        ""
        if is_first
        else "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
    )
    first_question_instruction = (
        f"\nWhen this task starts, ask exactly one question to begin: {first_question}"
    )
    suffix = "\nIMPORTANT: Do NOT search live availability, recommend options, give prices, provide travel advice, say goodbye, recap the full conversation, or tell the user you are done. Only collect the information for THIS specific task. As soon as all required details for this task are available, silently record them with the appropriate tool so the TaskGroup can immediately continue to the next task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
    return (
        prefix
        + agent_instructions
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


class CustomerInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the customer's full name, phone number, email address, total travelers, and traveler split when available: adults, kids, and senior citizens.\n- If the user does not know the split, record the total and leave the split fields empty.",
                first_question="May I have your full name, phone number, email address, and how many travelers will be going?",
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
        phone_number: str | None = None,
        email_address: str | None = None,
        total_travelers: int | None = None,
        adults: int | None = None,
        kids: int | None = None,
        senior_citizens: int | None = None,
    ):
        """Call when you have collected customer identity, contact, and travel group information."""
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
                is_first=False,
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
                "- Collect the travel source, destination, destination type or trip style, start date, end date, and whether dates are flexible.\n- Only ask transportation mode for domestic or same-country trips. Do not ask for transportation mode for international trips; leave transportation_mode empty unless the caller volunteers it.\n- Record dates as YYYY-MM-DD when the user provides exact dates. If dates are flexible, set flexible to true and leave unknown exact dates empty.",
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
        start_date: str | None = None,
        end_date: str | None = None,
        flexible: bool | None = None,
        transportation_mode: str | None = None,
    ):
        """Call when you have collected travel plan fields."""
        self.complete(
            TravelPlanResults(
                source=source,
                destination=destination,
                destination_type=destination_type,
                start_date=start_date,
                end_date=end_date,
                flexible=flexible,
                transportation_mode=transportation_mode,
            )
        )


class BudgetTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the budget range if available, budget type such as Economy, Standard, Premium, or Luxury, whether they are flexible, and any budget notes.",
                first_question="What overall budget or budget type would you like us to plan around?",
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
        budget_type: str | None = None,
        flexibility: bool | None = None,
        notes: str | None = None,
    ):
        """Call when you have collected budget fields."""
        self.complete(
            BudgetResults(
                budget_range=budget_range,
                budget_type=budget_type,
                flexibility=flexibility,
                notes=notes,
            )
        )


class AccommodationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect accommodation type, hotel rating, room type, important preferences, experience expectation, estimated price if the user volunteers it, and notes.\n- For preferences, capture whether they want breakfast included, pool, spa, wifi, kids friendly property, and near attractions.\n- Do not suggest hotel names, search availability, or quote prices.",
                first_question="What kind of accommodation would you prefer, such as hotel rating, room type, or must-have amenities?",
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
        room_type: str | None = None,
        breakfast_included: bool | None = None,
        pool: bool | None = None,
        spa: bool | None = None,
        wifi: bool | None = None,
        kids_friendly: bool | None = None,
        near_attractions: bool | None = None,
        experience_expectation: str | None = None,
        estimated_price: str | None = None,
        notes: str | None = None,
    ):
        """Call when you have collected accommodation preference fields."""
        self.complete(
            AccommodationPreferenceResults(
                accommodation_type=accommodation_type,
                hotel_rating=hotel_rating,
                room_type=room_type,
                breakfast_included=breakfast_included,
                pool=pool,
                spa=spa,
                wifi=wifi,
                kids_friendly=kids_friendly,
                near_attractions=near_attractions,
                experience_expectation=experience_expectation,
                estimated_price=estimated_price,
                notes=notes,
            )
        )


class ActivityTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the activity interest profile as a list, such as sightseeing, culture, nature, relaxation, adventure, shopping, or food.\n- Do not suggest activity names, search live availability, or quote ticket prices.",
                first_question="What kinds of activities are you interested in during the trip?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_activity")
    async def record_activity(
        self,
        context: RunContext,
        interest_profile: list[str],
    ):
        """Call when you have collected the customer's activity interests."""
        self.complete(
            ActivityPreferenceResults(
                interest_profile=interest_profile,
            )
        )


class FoodPreferenceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect food preference type, dietary requirements if any, and special food notes.\n- Preference type examples: Vegetarian, Non-Vegetarian, Mix of Both, Halal, Jain, or No Preference.",
                first_question="Do you have any food preferences or dietary requirements we should note?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_food_preferences")
    async def record_food_preferences(
        self,
        context: RunContext,
        preference_type: str | None = None,
        dietary_requirements: str | None = None,
        special_notes: str | None = None,
    ):
        """Call when you have collected food preference fields."""
        self.complete(
            FoodPreferenceResults(
                preference_type=preference_type,
                dietary_requirements=dietary_requirements,
                special_notes=special_notes,
            )
        )


class SupportServicesTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect required support services as a list, such as flight booking, airport transfer, visa assistance, travel insurance, hotel booking, or tour guide.\n- Also collect special assistance needs: senior support, kids support, and accessibility needs.",
                first_question="Which support services would you like, such as flights, transfers, visa help, insurance, hotel booking, or a tour guide?",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await _start_task(self.session)

    @function_tool(name="record_support_services")
    async def record_support_services(
        self,
        context: RunContext,
        required_services: list[str],
        senior_support: bool | None = None,
        kids_support: bool | None = None,
        accessibility: str | None = None,
    ):
        """Call when you have collected support service and assistance fields."""
        self.complete(
            SupportServicesResults(
                required_services=required_services,
                senior_support=senior_support,
                kids_support=kids_support,
                accessibility=accessibility,
            )
        )
