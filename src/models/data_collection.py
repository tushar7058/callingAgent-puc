from dataclasses import dataclass

@dataclass
class ContactIdentificationResults:
    contact_name: str
    work_email: str
    company_name: str
    role_title: str | None = None

@dataclass
class CompanyDetailsResults:
    company_website: str | None = None
    company_size: str | None = None

@dataclass
class UseCaseResults:
    primary_use_case: str
    current_solution: str | None = None

@dataclass
class RequirementsAndConstraintsResults:
    constraint: str

@dataclass
class TimelineAndUrgencyResults:
    urgency: str
    target_date: str | None = None
