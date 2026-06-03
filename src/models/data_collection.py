from dataclasses import dataclass


@dataclass
class CustomerInfoResults:
    full_name: str
    number_of_travelers: int | None = None

@dataclass
class LanguagePreferenceResults:
    preferred_language: str
    language_code: str | None = None

@dataclass
class TravelPlanResults:
    source: str
    destination: str
    travel_dates: str | None = None

@dataclass
class BudgetResults:
    budget: str

@dataclass
class TravelPreferenceResults:
    transportation_mode: str  # Flight, Train, Bus, etc.

@dataclass
class AccommodationPreferenceResults:
    accommodation_type: str  # 5-star, 4-star, villa, etc.

@dataclass
class ActivityPreferenceResults:
    activities: str

@dataclass
class ContactInfoResults:
    phone_number: str | None = None
    email_address: str | None = None
