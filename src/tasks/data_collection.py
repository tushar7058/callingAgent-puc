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


def _build_task_instructions(
    agent_instructions: str, task_instructions: str, is_first: bool = False
):
    prefix = (
        ""
        if is_first
        else "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
    )
    suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
    return prefix + agent_instructions + "\n" + task_instructions + suffix


class CustomerInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the customer's full name and how many people are traveling.",
                is_first=True,
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_customer_info")
    async def record_customer_info(
        self,
        context: RunContext,
        full_name: str,
        number_of_travelers: int | None = None,
    ):
        """Call when you have collected the customer's name and number of travelers."""
        self.complete(
            CustomerInfoResults(
                full_name=full_name, number_of_travelers=number_of_travelers
            )
        )


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
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_language_preference")
    async def record_language_preference(
        self, context: RunContext, preferred_language: str
    ):
        """Call when you have collected the customer's preferred language."""
        get_job_context().proc.userdata["preferred_language"] = preferred_language
        self.complete(LanguagePreferenceResults(preferred_language=preferred_language))


class TravelPlanTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Find out the travel source and destination and preferred travel dates or duration.",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_travel_plan")
    async def record_travel_plan(
        self,
        context: RunContext,
        source: str,
        destination: str,
        travel_dates: str | None = None,
    ):
        """Call when you have collected the source, destination, and dates."""
        self.complete(
            TravelPlanResults(
                source=source, destination=destination, travel_dates=travel_dates
            )
        )


class BudgetTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions, "- Ask for their overall budget for the trip."
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_budget")
    async def record_budget(self, context: RunContext, budget: str):
        """Call when you have collected the budget."""
        self.complete(BudgetResults(budget=budget))


class TravelPreferenceTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Ask for their preferred mode of transportation (e.g., flight, train, bus).\n- IMPORTANT: Once they reply, call the 'best_TravelOption' tool to fetch realtime options for their route. Include source, destination, travel dates, preferred mode, customer language, and destination country when known.\n- Present only the reliable options from the tool back to the user and ask which one they prefer to shortlist.\n- If the tool says realtime search is not configured or unavailable, do not invent details. Tell the customer Mannai Travels will verify live availability before final booking.\n- Only after they confirm a specific option, call record_travel_preference with structured details: transportation mode, selected option name, provider name, estimated price, duration, departure time, arrival time, and notes when known. Do not end the call.",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_travel_preference")
    async def record_travel_preference(
        self,
        context: RunContext,
        transportation_mode: str,
        selected_option: str | None = None,
        provider_name: str | None = None,
        estimated_price: str | None = None,
        duration: str | None = None,
        departure_time: str | None = None,
        arrival_time: str | None = None,
        notes: str | None = None,
    ):
        """Call when the customer confirms a transport option to shortlist. Include provider, price, timing, duration, and notes when known."""
        self.complete(
            TravelPreferenceResults(
                transportation_mode=transportation_mode,
                selected_option=selected_option,
                provider_name=provider_name,
                estimated_price=estimated_price,
                duration=duration,
                departure_time=departure_time,
                arrival_time=arrival_time,
                notes=notes,
            )
        )


class AccommodationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Ask for accommodation preferences (e.g., 5-star, 4-star, villa, beach resort, city hotel, family room).\n- IMPORTANT: Once they reply, call the 'gethotel_details' tool using their destination city. Include accommodation preference, travel dates, customer language, and destination country when known.\n- Suggest only the reliable hotels from the tool's response and ask which one they prefer to shortlist.\n- If the tool says realtime search is not configured or unavailable, do not invent hotel names, rates, or availability. Tell the customer Mannai Travels will verify live availability before final booking.\n- Only after they confirm a specific hotel, call record_accommodation with structured details: accommodation type, hotel name, hotel rating, room type, location preference, meal plan, amenities, estimated price, and notes when known. Do not end the call.",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_accommodation")
    async def record_accommodation(
        self,
        context: RunContext,
        accommodation_type: str,
        hotel_name: str | None = None,
        hotel_rating: str | None = None,
        room_type: str | None = None,
        location_preference: str | None = None,
        meal_plan: str | None = None,
        amenities: str | None = None,
        estimated_price: str | None = None,
        notes: str | None = None,
    ):
        """Call when the customer confirms an accommodation option to shortlist. Include hotel, rating, room, location, meal, amenities, price, and notes when known."""
        self.complete(
            AccommodationPreferenceResults(
                accommodation_type=accommodation_type,
                hotel_name=hotel_name,
                hotel_rating=hotel_rating,
                room_type=room_type,
                location_preference=location_preference,
                meal_plan=meal_plan,
                amenities=amenities,
                estimated_price=estimated_price,
                notes=notes,
            )
        )


class ActivityTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Ask what kind of activities they want to do (e.g., adventure, sightseeing, relaxing, private tours, food, culture).\n- IMPORTANT: Once they reply, call the 'locationbasedactivitydetail' tool for their destination. Include activity preference, travel dates, customer language, and destination country when known.\n- Suggest only the reliable activities from the tool's response and ask which one they prefer to shortlist.\n- If the tool says realtime search is not configured or unavailable, do not invent ticket prices, schedules, or availability. Tell the customer Mannai Travels will verify live availability before final booking.\n- Only after they confirm a specific activity, call record_activity with structured details: requested activity preference, selected activity, activity category, tour type, preferred time, estimated price, and notes when known. Do not end the call.",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_activity")
    async def record_activity(
        self,
        context: RunContext,
        activities: str,
        selected_activity: str | None = None,
        activity_category: str | None = None,
        tour_type: str | None = None,
        preferred_time: str | None = None,
        estimated_price: str | None = None,
        notes: str | None = None,
    ):
        """Call when the customer confirms an activity to shortlist. Include selected activity, category, tour type, timing, price, and notes when known."""
        self.complete(
            ActivityPreferenceResults(
                activities=activities,
                selected_activity=selected_activity,
                activity_category=activity_category,
                tour_type=tour_type,
                preferred_time=preferred_time,
                estimated_price=estimated_price,
                notes=notes,
            )
        )


class ContactInfoTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        super().__init__(
            instructions=_build_task_instructions(
                agent_instructions,
                "- Collect the customer's contact information (email and phone number).",
            ),
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Begin this task now.",
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_contact_info")
    async def record_contact_info(
        self,
        context: RunContext,
        phone_number: str | None = None,
        email_address: str | None = None,
    ):
        """Call when you have collected contact info."""
        self.complete(
            ContactInfoResults(phone_number=phone_number, email_address=email_address)
        )
