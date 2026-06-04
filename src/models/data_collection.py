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
    selected_option: str | None = None
    provider_name: str | None = None
    estimated_price: str | None = None
    duration: str | None = None
    departure_time: str | None = None
    arrival_time: str | None = None
    notes: str | None = None


@dataclass
class AccommodationPreferenceResults:
    accommodation_type: str  # 5-star, 4-star, villa, etc.
    hotel_name: str | None = None
    hotel_rating: str | None = None
    room_type: str | None = None
    location_preference: str | None = None
    meal_plan: str | None = None
    amenities: str | None = None
    estimated_price: str | None = None
    notes: str | None = None


@dataclass
class ActivityPreferenceResults:
    activities: str
    selected_activity: str | None = None
    activity_category: str | None = None
    tour_type: str | None = None
    preferred_time: str | None = None
    estimated_price: str | None = None
    notes: str | None = None


@dataclass
class ContactInfoResults:
    phone_number: str | None = None
    email_address: str | None = None
