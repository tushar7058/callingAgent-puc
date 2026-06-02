from livekit.agents import AgentTask, function_tool, RunContext
from models.data_collection import (
    ContactIdentificationResults,
    CompanyDetailsResults,
    UseCaseResults,
    RequirementsAndConstraintsResults,
    TimelineAndUrgencyResults
)

class ContactIdentificationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = "The user has already been greeted. Do not introduce yourself or say hello. Directly ask for the required information.\n"
        task_instructions = "- Collect the prospect's full name, work email, role, and company name.\n- If they do not want to share a role, leave it blank.\n- Confirm the spelling of the email if spoken aloud."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_contact_identification")
    async def record_contact_identification(
        self,
        context: RunContext,
        contact_name: str,
        work_email: str,
        company_name: str,
        role_title: str | None = None
    ):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    contact_name (str)
    work_email (str)
    company_name (str)
    role_title (str | None) (optional)"""
        self.complete(ContactIdentificationResults(contact_name=contact_name, work_email=work_email, company_name=company_name, role_title=role_title))


class CompanyDetailsTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "- Collect the company website if known and approximate company size.\n- If exact size is unknown, capture the closest estimate."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_company_details")
    async def record_company_details(
        self,
        context: RunContext,
        company_website: str | None = None,
        company_size: str | None = None
    ):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    company_website (str | None) (optional)
    company_size (str | None) (optional)"""
        self.complete(CompanyDetailsResults(company_website=company_website, company_size=company_size))


class UseCaseTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "- Understand what the prospect wants to build, buy, or improve.\n- Capture the primary use case in plain language.\n- If they mention a current tool, vendor, or workaround, capture that too."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_use_case")
    async def record_use_case(
        self,
        context: RunContext,
        primary_use_case: str,
        current_solution: str | None = None
    ):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    primary_use_case (str)
    current_solution (str | None) (optional)"""
        self.complete(UseCaseResults(primary_use_case=primary_use_case, current_solution=current_solution))


class RequirementsAndConstraintsTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "- Capture notable requirements, blockers, or constraints.\n- Add one list item per distinct requirement or constraint."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        self._partial_results: list[RequirementsAndConstraintsResults] = []
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are collecting multiple data points for this task. "
                "As the user provides each data point, call edit_requirements_and_constraints_list. "
                "When the user confirms the list is complete, call record_requirements_and_constraints."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="edit_requirements_and_constraints_list")
    async def edit_requirements_and_constraints_list(self, context: RunContext, constraint: str):
        """Update the partial list: add a new data point to the running list.

Args:
    constraint (str)"""
        self._partial_results.append(RequirementsAndConstraintsResults(constraint=constraint))
        return (
            f"Data point added (list now has {len(self._partial_results)} item(s)). "
            "Ask if the user wants to add more items or if the list is complete. "
            "When done, call record_requirements_and_constraints."
        )

    @function_tool(name="record_requirements_and_constraints")
    async def record_requirements_and_constraints(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class TimelineAndUrgencyTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = ""
        task_instructions = "- Understand how soon they want to evaluate or move forward and whether there is a specific deadline."
        no_goodbye_suffix = "\nIMPORTANT: Do NOT say goodbye, recap the full conversation, or tell the user you are done. Only focus on collecting the information for THIS specific task. If the information was already provided earlier in the conversation, confirm it briefly and then record it immediately using the appropriate tool."
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_timeline_and_urgency")
    async def record_timeline_and_urgency(self, context: RunContext, urgency: str, target_date: str | None = None):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    urgency (str)
    target_date (str | None) (optional)"""
        self.complete(TimelineAndUrgencyResults(urgency=urgency, target_date=target_date))
