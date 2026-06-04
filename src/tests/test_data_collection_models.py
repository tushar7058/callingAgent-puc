from __future__ import annotations

from models.data_collection import (
    AccommodationPreferenceResults,
    ActivityPreferenceResults,
    TravelPreferenceResults,
)
from utils.helpers import _to_json_serializable


def test_travel_preference_serializes_structured_option_details():
    result = TravelPreferenceResults(
        transportation_mode="flight",
        selected_option="Mumbai to Goa morning flight",
        provider_name="IndiGo",
        estimated_price="INR 6,500",
        duration="one hour twenty minutes",
        departure_time="08:30",
        arrival_time="09:50",
        notes="Client prefers fastest option.",
    )

    assert _to_json_serializable(result) == {
        "transportation_mode": "flight",
        "selected_option": "Mumbai to Goa morning flight",
        "provider_name": "IndiGo",
        "estimated_price": "INR 6,500",
        "duration": "one hour twenty minutes",
        "departure_time": "08:30",
        "arrival_time": "09:50",
        "notes": "Client prefers fastest option.",
    }


def test_accommodation_serializes_client_hotel_preferences():
    result = AccommodationPreferenceResults(
        accommodation_type="five-star beach resort",
        hotel_name="W Goa",
        hotel_rating="five-star",
        room_type="sea-view room",
        location_preference="near Vagator Beach",
        meal_plan="breakfast included",
        amenities="pool, spa, beach access",
        estimated_price="INR 22,000 per night",
        notes="Client selected W Goa from realtime options.",
    )

    assert _to_json_serializable(result)["hotel_name"] == "W Goa"
    assert _to_json_serializable(result)["hotel_rating"] == "five-star"
    assert _to_json_serializable(result)["amenities"] == "pool, spa, beach access"


def test_activity_serializes_selected_activity_details():
    result = ActivityPreferenceResults(
        activities="unique cultural activities",
        selected_activity="private tour of a spice plantation",
        activity_category="culture",
        tour_type="private tour",
        preferred_time="morning",
        estimated_price="to be confirmed",
        notes="Client does not want normal sightseeing.",
    )

    assert _to_json_serializable(result) == {
        "activities": "unique cultural activities",
        "selected_activity": "private tour of a spice plantation",
        "activity_category": "culture",
        "tour_type": "private tour",
        "preferred_time": "morning",
        "estimated_price": "to be confirmed",
        "notes": "Client does not want normal sightseeing.",
    }
