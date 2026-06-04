from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from livekit.agents import function_tool, get_job_context

DEFAULT_SEARCH_TIMEOUT = 8.0
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


@dataclass
class SearchItem:
    title: str
    detail: str
    link: str | None = None


@dataclass(frozen=True)
class SearchContext:
    language: str | None = None
    country: str | None = None


COUNTRY_ALIASES = {
    "ae": "ae",
    "arab emirates": "ae",
    "united arab emirates": "ae",
    "uae": "ae",
    "qa": "qa",
    "qatar": "qa",
    "sa": "sa",
    "ksa": "sa",
    "saudi": "sa",
    "saudi arabia": "sa",
    "om": "om",
    "oman": "om",
    "bh": "bh",
    "bahrain": "bh",
    "kw": "kw",
    "kuwait": "kw",
    "tr": "tr",
    "turkey": "tr",
    "gb": "gb",
    "uk": "gb",
    "united kingdom": "gb",
    "us": "us",
    "usa": "us",
    "united states": "us",
    "in": "in",
    "india": "in",
}

LANGUAGE_ALIASES = {
    "ar": "ar",
    "arabic": "ar",
    "en": "en",
    "en-us": "en",
    "english": "en",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
}


def _search_timeout() -> float:
    raw_timeout = _env_value("TRAVEL_SEARCH_TIMEOUT")
    if not raw_timeout:
        return DEFAULT_SEARCH_TIMEOUT

    try:
        return max(1.0, float(raw_timeout))
    except ValueError:
        return DEFAULT_SEARCH_TIMEOUT


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default

    stripped_value = value.strip()
    return stripped_value or default


def _serpapi_search_url() -> str:
    return _env_value("SERPAPI_SEARCH_URL", SERPAPI_SEARCH_URL) or SERPAPI_SEARCH_URL


def _search_result_limit() -> str:
    raw_limit = _env_value("TRAVEL_SEARCH_RESULT_LIMIT")
    if not raw_limit:
        return "5"

    try:
        return str(max(1, min(10, int(raw_limit))))
    except ValueError:
        return "5"


def _request_json(url: str, params: dict[str, str], timeout: float) -> dict[str, Any]:
    parsed_url = parse.urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Realtime search URL is not configured correctly.")

    existing_params = dict(parse.parse_qsl(parsed_url.query))
    query_string = parse.urlencode({**existing_params, **params})
    request_url = parse.urlunparse(parsed_url._replace(query=query_string))
    req = request.Request(request_url, method="GET")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"error": f"HTTP {exc.code}: {exc.reason}"}
        if isinstance(data, dict):
            data.setdefault("error", f"HTTP {exc.code}: {exc.reason}")
            return data
        return {"error": f"HTTP {exc.code}: {exc.reason}"}

    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


async def _fetch_serpapi_results(
    query: str,
    *,
    search_language: str | None = None,
    search_country: str | None = None,
    locations: list[str] | None = None,
) -> tuple[list[SearchItem], str | None]:
    api_key = _env_value("SERPAPI_API_KEY")
    if not api_key:
        return [], (
            "Realtime search is not configured. Set SERPAPI_API_KEY to let AIRA "
            "fetch live flights, hotels, and sightseeing options."
        )

    search_context = _resolve_search_context(
        search_language=search_language,
        search_country=search_country,
        destination_text=" ".join(locations or []),
    )
    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "num": _search_result_limit(),
    }
    if search_context.language:
        params["hl"] = search_context.language
    if search_context.country:
        params["gl"] = search_context.country

    try:
        data = await asyncio.to_thread(
            _request_json,
            _serpapi_search_url(),
            params,
            _search_timeout(),
        )
    except (
        TimeoutError,
        ValueError,
        error.URLError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        return [], f"Realtime search is temporarily unavailable: {exc}"

    if data.get("error"):
        return [], f"Realtime search returned an error: {data['error']}"

    return _extract_search_items(data), None


def _resolve_search_context(
    *,
    search_language: str | None,
    search_country: str | None,
    destination_text: str | None = None,
) -> SearchContext:
    language = _normalize_language(search_language) or _language_from_job_context()
    country = _normalize_country(search_country) or _country_from_explicit_text(
        destination_text
    )

    if not language:
        language = _normalize_language(_env_value("TRAVEL_SEARCH_LANGUAGE"))
    if not country:
        country = _normalize_country(_env_value("TRAVEL_SEARCH_COUNTRY"))

    return SearchContext(language=language, country=country)


def _language_from_job_context() -> str | None:
    try:
        preferred_language_code = get_job_context().proc.userdata.get(
            "preferred_language_code"
        )
    except RuntimeError:
        return None
    return _normalize_language(preferred_language_code)


def _country_from_explicit_text(text: str | None) -> str | None:
    if not text:
        return None
    normalized_text = text.strip().lower()
    if not normalized_text:
        return None

    words = {
        word.strip(".,;:()[]{}") for word in normalized_text.replace("-", " ").split()
    }
    for alias, country_code in COUNTRY_ALIASES.items():
        alias_words = alias.split()
        if len(alias_words) == 1 and alias in words:
            return country_code
        if len(alias_words) > 1 and alias in normalized_text:
            return country_code
    return None


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]
    base_language = normalized.split("-", maxsplit=1)[0]
    if len(base_language) == 2 and base_language.isalpha():
        return base_language
    return None


def _normalize_country(country: str | None) -> str | None:
    if not country:
        return None
    normalized = country.strip().lower()
    if not normalized:
        return None
    if normalized in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[normalized]
    if len(normalized) == 2 and normalized.isalpha():
        return normalized

    return None


def _extract_search_items(data: dict[str, Any], *, limit: int = 3) -> list[SearchItem]:
    items: list[SearchItem] = []

    answer_box = data.get("answer_box")
    if isinstance(answer_box, dict):
        title = str(answer_box.get("title") or answer_box.get("answer") or "").strip()
        detail = str(
            answer_box.get("snippet")
            or answer_box.get("description")
            or answer_box.get("answer")
            or ""
        ).strip()
        if title and detail:
            items.append(
                SearchItem(title=title, detail=detail, link=answer_box.get("link"))
            )

    for result in data.get("local_results", {}).get("places", []):
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "").strip()
        detail_parts = [
            str(result.get("rating") or "").strip(),
            str(result.get("price") or "").strip(),
            str(result.get("address") or "").strip(),
            str(result.get("description") or "").strip(),
        ]
        detail = " | ".join(part for part in detail_parts if part)
        if title and detail:
            items.append(
                SearchItem(title=title, detail=detail, link=result.get("link"))
            )

    for result in data.get("organic_results", []):
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "").strip()
        detail = str(result.get("snippet") or result.get("rich_snippet") or "").strip()
        if title and detail:
            items.append(
                SearchItem(title=title, detail=detail, link=result.get("link"))
            )

    seen: set[str] = set()
    unique_items: list[SearchItem] = []
    for item in items:
        key = item.title.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
        if len(unique_items) >= limit:
            break

    return unique_items


def _format_items(
    prefix: str, items: list[SearchItem], error_message: str | None
) -> str:
    if error_message:
        return (
            f"{prefix} {error_message} Do not invent prices, flight times, hotel "
            "availability, or activity availability. Ask the customer for any "
            "missing preferences and tell them Mannai Travels will confirm live "
            "availability before booking."
        )

    if not items:
        return (
            f"{prefix} I could not find reliable live options for this request. "
            "Ask one follow-up question to narrow the request, then explain that "
            "Mannai Travels will verify current availability before booking."
        )

    options = [
        f"{index}) {item.title}: {item.detail}"
        for index, item in enumerate(items, start=1)
    ]
    return f"{prefix} Live search found: " + " ".join(options)


@function_tool(name="gethotel_details")
async def gethotel_details(
    destination_city: str,
    accommodation_preference: str | None = None,
    travel_dates: str | None = None,
    search_language: str | None = None,
    search_country: str | None = None,
) -> str:
    """Fetch live hotel options for a destination city. Include accommodation preference, dates, customer language, and customer country when available."""

    query_parts = [
        "current hotel availability",
        destination_city,
        accommodation_preference or "",
        travel_dates or "",
        "rates official booking",
    ]
    query = " ".join(part for part in query_parts if part).strip()
    items, error_message = await _fetch_serpapi_results(
        query,
        search_language=search_language,
        search_country=search_country,
        locations=[destination_city],
    )
    return _format_items(f"Hotel options for {destination_city}.", items, error_message)


@function_tool(name="best_TravelOption")
async def best_TravelOption(  # noqa: N802 - existing LiveKit tool id uses this name.
    source: str,
    destination: str,
    travel_dates: str | None = None,
    preferred_mode: str | None = None,
    search_language: str | None = None,
    search_country: str | None = None,
) -> str:
    """Fetch live travel options for a route. Include dates, preferred mode, customer language, and destination country when available."""

    mode_text = preferred_mode or "flight train bus"
    query_parts = [
        "current travel options",
        mode_text,
        "from",
        source,
        "to",
        destination,
        travel_dates or "",
        "prices schedules",
    ]
    query = " ".join(part for part in query_parts if part).strip()
    items, error_message = await _fetch_serpapi_results(
        query,
        search_language=search_language,
        search_country=search_country,
        locations=[destination, source],
    )
    return _format_items(
        f"Travel options from {source} to {destination}.", items, error_message
    )


@function_tool(name="locationbasedactivitydetail")
async def locationbasedactivitydetail(
    location: str,
    activity_preference: str | None = None,
    travel_dates: str | None = None,
    search_language: str | None = None,
    search_country: str | None = None,
) -> str:
    """Fetch live sightseeing and activity recommendations. Include activity preference, dates, customer language, and destination country when available."""

    query_parts = [
        "current sightseeing tours activities",
        location,
        activity_preference or "",
        travel_dates or "",
        "tickets availability",
    ]
    query = " ".join(part for part in query_parts if part).strip()
    items, error_message = await _fetch_serpapi_results(
        query,
        search_language=search_language,
        search_country=search_country,
        locations=[location],
    )
    return _format_items(f"Activities in {location}.", items, error_message)
