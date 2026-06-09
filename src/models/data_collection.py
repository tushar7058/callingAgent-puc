from dataclasses import dataclass


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
    start_date: str | None = None
    end_date: str | None = None
    flexible: bool | None = None
    transportation_mode: str | None = None


@dataclass
class BudgetResults:
    budget_range: str | None = None
    budget_type: str | None = None
    flexibility: bool | None = None
    notes: str | None = None


@dataclass
class AccommodationPreferenceResults:
    accommodation_type: str | None = None
    hotel_rating: str | None = None
    room_type: str | None = None
    breakfast_included: bool | None = None
    pool: bool | None = None
    spa: bool | None = None
    wifi: bool | None = None
    kids_friendly: bool | None = None
    near_attractions: bool | None = None
    experience_expectation: str | None = None
    estimated_price: str | None = None
    notes: str | None = None


@dataclass
class ActivityPreferenceResults:
    interest_profile: list[str]


@dataclass
class FoodPreferenceResults:
    preference_type: str | None = None
    dietary_requirements: str | None = None
    special_notes: str | None = None


@dataclass
class SupportServicesResults:
    required_services: list[str]
    senior_support: bool | None = None
    kids_support: bool | None = None
    accessibility: str | None = None
