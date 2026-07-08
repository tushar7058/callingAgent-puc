from dataclasses import dataclass


@dataclass
class GreetingResults:
    pass


@dataclass
class CustomerInfoResults:
    full_name: str
    phone_number: str | None = None
    email_address: str | None = None
    total_travelers: int | None = None
    adults: int | None = None
    kids: int | None = None
    senior_citizens: int | None = None


@dataclass
class LanguagePreferenceResults:
    preferred_language: str
    language_code: str | None = None


@dataclass
class TravelPlanResults:
    source: str
    destination: str
    destination_type: str | None = None
    activities: bool | None = None
    start_date: str | None = None
    end_date: str | None = None
    flexible: bool | None = None
    transportation_mode: str | None = None
    senior_support: bool | None = None
    kids_support: bool | None = None
    accessibility: str | None = None


@dataclass
class BudgetResults:
    budget_range: str | None = None
    flexibility: bool | None = None


@dataclass
class AccommodationPreferenceResults:
    accommodation_type: str | None = None
    hotel_rating: str | None = None
