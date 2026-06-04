from __future__ import annotations

import io
import json
from urllib import error

import pytest

from utils import tools


def _serpapi_payload(*results: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    return {"organic_results": list(results)}


def test_serpapi_search_url_constant_is_not_loaded_from_blank_env(monkeypatch):
    monkeypatch.setenv("SERPAPI_SEARCH_URL", "")

    assert tools.SERPAPI_SEARCH_URL == "https://serpapi.com/search.json"
    assert tools._serpapi_search_url() == "https://serpapi.com/search.json"


def test_request_json_merges_provider_url_query_params(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"organic_results":[]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return DummyResponse()

    monkeypatch.setattr(tools.request, "urlopen", fake_urlopen)

    tools._request_json(
        "https://serpapi.com/search?engine=google",
        {"api_key": "test-key", "engine": "google", "q": "Goa"},
        1.0,
    )

    assert captured["url"].startswith("https://serpapi.com/search?")
    assert captured["url"].count("?") == 1
    assert "engine=google" in captured["url"]
    assert "q=Goa" in captured["url"]


@pytest.mark.asyncio
async def test_best_travel_option_uses_live_search_results(monkeypatch):
    captured = {}

    def fake_request_json(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _serpapi_payload(
            {
                "title": "Doha to Dubai flights",
                "snippet": "Live fares from Qatar Airways and Flydubai today.",
                "link": "https://example.com/flights",
            },
            {
                "title": "Doha to Dubai bus options",
                "snippet": "Current intercity options and timings.",
                "link": "https://example.com/bus",
            },
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_SEARCH_URL", raising=False)
    monkeypatch.delenv("TRAVEL_SEARCH_LANGUAGE", raising=False)
    monkeypatch.delenv("TRAVEL_SEARCH_COUNTRY", raising=False)
    monkeypatch.setenv("TRAVEL_SEARCH_TIMEOUT", "2.5")
    monkeypatch.setattr(tools, "_request_json", fake_request_json)

    result = await tools.best_TravelOption._func(
        "Doha", "Dubai", "next Friday", "flight", "Arabic", "United Arab Emirates"
    )

    assert captured["url"] == tools.SERPAPI_SEARCH_URL
    assert captured["params"]["api_key"] == "test-key"
    assert captured["params"]["hl"] == "ar"
    assert captured["params"]["gl"] == "ae"
    assert (
        "current travel options flight from Doha to Dubai next Friday"
        in captured["params"]["q"]
    )
    assert captured["timeout"] == 2.5
    assert "Live search found" in result
    assert "Doha to Dubai flights" in result
    assert "Qatar Airways" in result


@pytest.mark.asyncio
async def test_hotel_tool_does_not_force_country_for_ambiguous_destination(
    monkeypatch,
):
    captured = {}

    def fake_request_json(url, params, timeout):
        captured["params"] = params
        return _serpapi_payload(
            {
                "title": "Goa hotel availability",
                "snippet": "Current rates for the requested Goa destination.",
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_SEARCH_LANGUAGE", raising=False)
    monkeypatch.delenv("TRAVEL_SEARCH_COUNTRY", raising=False)
    monkeypatch.setattr(tools, "_request_json", fake_request_json)

    result = await tools.gethotel_details._func("Goa", "boutique", "June")

    assert "hl" not in captured["params"]
    assert "gl" not in captured["params"]
    assert "Goa hotel availability" in result


@pytest.mark.asyncio
async def test_activity_tool_uses_explicit_country_from_customer_destination(
    monkeypatch,
):
    captured = {}

    def fake_request_json(url, params, timeout):
        captured["params"] = params
        return _serpapi_payload(
            {
                "title": "Goa USA experiences",
                "snippet": "Current activities for Goa in the United States.",
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_SEARCH_LANGUAGE", raising=False)
    monkeypatch.delenv("TRAVEL_SEARCH_COUNTRY", raising=False)
    monkeypatch.setattr(tools, "_request_json", fake_request_json)

    result = await tools.locationbasedactivitydetail._func(
        "Goa USA", "unique local experiences", "10th to 15th of next month"
    )

    assert captured["params"]["gl"] == "us"
    assert "Goa USA experiences" in result


@pytest.mark.asyncio
async def test_activity_tool_ignores_blank_provider_env_values(monkeypatch):
    captured = {}

    def fake_request_json(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _serpapi_payload(
            {
                "title": "Unique Goa experiences",
                "snippet": "Current tours and experiences in Goa.",
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_SEARCH_URL", "")
    monkeypatch.setenv("TRAVEL_SEARCH_RESULT_LIMIT", "")
    monkeypatch.delenv("TRAVEL_SEARCH_LANGUAGE", raising=False)
    monkeypatch.delenv("TRAVEL_SEARCH_COUNTRY", raising=False)
    monkeypatch.setattr(tools, "_request_json", fake_request_json)

    result = await tools.locationbasedactivitydetail._func(
        "Goa", "unique local experiences", "10th to 15th of next month"
    )

    assert captured["url"] == tools.SERPAPI_SEARCH_URL
    assert captured["params"]["num"] == "5"
    assert "gl" not in captured["params"]
    assert "Unique Goa experiences" in result


@pytest.mark.asyncio
async def test_activity_tool_handles_invalid_provider_url(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_SEARCH_URL", "not-a-url")

    result = await tools.locationbasedactivitydetail._func("Goa")

    assert "Realtime search is temporarily unavailable" in result
    assert "Realtime search URL is not configured correctly" in result
    assert "Do not invent" in result


@pytest.mark.asyncio
async def test_activity_tool_reports_serpapi_http_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"Invalid API key"}'),
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(tools.request, "urlopen", fake_urlopen)

    result = await tools.locationbasedactivitydetail._func("Goa")

    assert "Realtime search returned an error" in result
    assert "Invalid API key" in result
    assert "Do not invent" in result


def test_resolve_search_context_prefers_customer_answers_over_env(monkeypatch):
    monkeypatch.setenv("TRAVEL_SEARCH_LANGUAGE", "fr")
    monkeypatch.setenv("TRAVEL_SEARCH_COUNTRY", "qa")

    context = tools._resolve_search_context(
        search_language="Arabic",
        search_country="Saudi Arabia",
        destination_text="Doha Dubai",
    )

    assert context == tools.SearchContext(language="ar", country="sa")


@pytest.mark.asyncio
async def test_hotel_tool_reports_missing_realtime_configuration(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = await tools.gethotel_details._func("Dubai", "five-star", "next week")

    assert "Realtime search is not configured" in result
    assert "Set SERPAPI_API_KEY" in result
    assert "Do not invent" in result


@pytest.mark.asyncio
async def test_activity_tool_reports_provider_errors(monkeypatch):
    def fake_request_json(url, params, timeout):
        return {"error": "monthly quota exceeded"}

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_request_json", fake_request_json)

    result = await tools.locationbasedactivitydetail._func(
        "Istanbul", "sightseeing", "June"
    )

    assert "Realtime search returned an error" in result
    assert "monthly quota exceeded" in result
    assert "Do not invent" in result


def test_extract_search_items_handles_answer_box_local_and_organic_results():
    data = {
        "answer_box": {
            "title": "Best hotels in Muscat",
            "snippet": "Updated list from a travel guide.",
            "link": "https://example.com/answer",
        },
        "local_results": {
            "places": [
                {
                    "title": "Grand Muscat Hotel",
                    "rating": "4.7",
                    "price": "$$",
                    "address": "Muscat",
                }
            ]
        },
        "organic_results": [
            {
                "title": "Muscat tours",
                "snippet": "Current day trips and sightseeing tours.",
            }
        ],
    }

    items = tools._extract_search_items(json.loads(json.dumps(data)))

    assert [item.title for item in items] == [
        "Best hotels in Muscat",
        "Grand Muscat Hotel",
        "Muscat tours",
    ]
