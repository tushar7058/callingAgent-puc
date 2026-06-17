from __future__ import annotations

from models.data_collection import (
    AccommodationPreferenceResults,
    BudgetResults,
    CustomerInfoResults,
    LanguagePreferenceResults,
    TravelPlanResults,
)
from utils.helpers import build_travel_lead_payload


def test_build_travel_lead_payload_matches_external_contract():
    payload = build_travel_lead_payload(
        {
            "language_preference": LanguagePreferenceResults(
                preferred_language="English",
                language_code="en-US",
            ),
            "customer_info": CustomerInfoResults(
                full_name="John Doe",
                phone_number="70058949269",
                email_address="john.doe@gmail.com",
                total_travelers=10,
            ),
            "travel_plan": TravelPlanResults(
                source="Mumbai",
                destination="Doha",
                destination_type="City & Culture",
                activities=True,
                start_date="2026-06-10",
                end_date="2026-06-15",
                flexible=True,
                transportation_mode="flight",
                senior_support=False,
                kids_support=True,
                accessibility=None,
            ),
            "budget": BudgetResults(
                budget_range="forty thousand",
                flexibility=True,
            ),
            "accommodation": AccommodationPreferenceResults(
                accommodation_type="five star resort",
                hotel_rating="5-star",
            ),
        },
        dateandtime="2026-06-10T12:34:56",
    )

    assert payload == {
        "communication_mode": "Agent",
        "Dateandtime": "2026-06-10T12:34:56",
        "language_preference": {
            "preferred_language": "English",
            "language_code": "en-US",
        },
        "customer_info": {
            "full_name": "John Doe",
            "contact": {
                "phone_number": "70058949269",
                "email_address": "john.doe@gmail.com",
            },
            "travel_group": {
                "total_travelers": 10,
                "adults": None,
                "kids": None,
                "senior_citizens": None,
            },
        },
        "travel_plan": {
            "source": "Mumbai",
            "destination": "Doha",
            "destination_type": "City & Culture",
            "activities": True,
            "travel_dates": {
                "start_date": "2026-06-10",
                "end_date": "2026-06-15",
                "flexible": True,
            },
            "transportation_mode": "flight",
            "special_assistance": {
                "senior_support": False,
                "kids_support": True,
                "accessibility": None,
            },
        },
        "budget": {
            "budget_range": "forty thousand",
            "flexibility": True,
        },
        "accommodation": {
            "accommodation_type": "five star resort",
            "hotel_rating": "5-star",
        },
    }


def test_build_travel_lead_payload_preserves_schema_for_partial_results():
    payload = build_travel_lead_payload(
        {
            "customer_info": CustomerInfoResults(
                full_name="Jane Doe",
                phone_number="12345",
            )
        },
        dateandtime="2026-06-10T12:34:56",
    )

    assert payload["communication_mode"] == "Agent"
    assert payload["Dateandtime"] == "2026-06-10T12:34:56"
    assert payload["customer_info"]["full_name"] == "Jane Doe"
    assert payload["customer_info"]["contact"]["phone_number"] == "12345"
    assert payload["travel_plan"]["source"] is None
    assert payload["travel_plan"]["activities"] is None
    assert payload["travel_plan"]["special_assistance"]["accessibility"] is None


def test_build_travel_lead_payload_localizes_arabic_language_label():
    payload = build_travel_lead_payload(
        {
            "language_preference": LanguagePreferenceResults(
                preferred_language="Arabic",
                language_code="ar",
            ),
            "travel_plan": TravelPlanResults(
                source="مومباي",
                destination="الدوحة",
                transportation_mode="رحلة طيران",
            ),
        },
        dateandtime="2026-06-10T12:34:56",
    )

    assert payload["communication_mode"] == "Agent"
    assert payload["Dateandtime"] == "2026-06-10T12:34:56"
    assert payload["language_preference"] == {
        "preferred_language": "العربية",
        "language_code": "ar",
    }
    assert payload["travel_plan"]["source"] == "مومباي"
    assert payload["travel_plan"]["transportation_mode"] == "رحلة طيران"
