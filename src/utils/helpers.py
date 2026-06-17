from dataclasses import asdict, is_dataclass
from datetime import datetime


def _to_json_serializable(obj):
    """Convert dataclasses and nested structures to JSON-serializable form."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


def _task_result(task_results: dict, key: str) -> dict:
    value = task_results.get(key) or {}
    value = _to_json_serializable(value)
    return value if isinstance(value, dict) else {}


def _language_label(language: dict) -> str:
    code = (language.get("language_code") or "").strip().lower()
    preferred = (language.get("preferred_language") or "").strip()
    if code == "ar" or preferred.lower() in {"arabic", "ar"}:
        return "العربية"
    return preferred or "English"


def build_travel_lead_payload(
    task_results: dict | None, *, dateandtime: str | None = None
) -> dict:
    """Build the external travel lead payload, preserving nulls for missing data."""
    task_results = task_results or {}
    dateandtime = dateandtime or datetime.now().isoformat(timespec="seconds")
    language = _task_result(task_results, "language_preference")
    customer = _task_result(task_results, "customer_info")
    contact = _task_result(task_results, "contact_info")
    travel_plan = _task_result(task_results, "travel_plan")
    budget = _task_result(task_results, "budget")
    accommodation = _task_result(task_results, "accommodation")

    return {
        "communication_mode": "Agent",
        "Dateandtime": dateandtime,
        "language_preference": {
            "preferred_language": _language_label(language),
            "language_code": language.get("language_code") or "en-US",
        },
        "customer_info": {
            "full_name": customer.get("full_name"),
            "contact": {
                "phone_number": customer.get("phone_number")
                or contact.get("phone_number"),
                "email_address": customer.get("email_address")
                or contact.get("email_address"),
            },
            "travel_group": {
                "total_travelers": customer.get("total_travelers")
                or customer.get("number_of_travelers"),
                "adults": customer.get("adults"),
                "kids": customer.get("kids"),
                "senior_citizens": customer.get("senior_citizens"),
            },
        },
        "travel_plan": {
            "source": travel_plan.get("source"),
            "destination": travel_plan.get("destination"),
            "destination_type": travel_plan.get("destination_type"),
            "activities": travel_plan.get("activities"),
            "travel_dates": {
                "start_date": travel_plan.get("start_date"),
                "end_date": travel_plan.get("end_date"),
                "flexible": travel_plan.get("flexible"),
            },
            "transportation_mode": travel_plan.get("transportation_mode"),
            "special_assistance": {
                "senior_support": travel_plan.get("senior_support"),
                "kids_support": travel_plan.get("kids_support"),
                "accessibility": travel_plan.get("accessibility"),
            },
        },
        "budget": {
            "budget_range": budget.get("budget_range") or budget.get("budget"),
            "flexibility": budget.get("flexibility"),
        },
        "accommodation": {
            "accommodation_type": accommodation.get("accommodation_type"),
            "hotel_rating": accommodation.get("hotel_rating"),
        },
    }
