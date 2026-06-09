"""Helpers for sending finalized lead data to an external API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from dotenv import load_dotenv

load_dotenv(".env")


def _resolve_external_api_url() -> str:
    return (os.getenv("EXTERNAL_API_URL") or os.getenv("EXTERNAL_API") or "").strip()


def send_to_external_api(payload: dict[str, Any], *, timeout: float = 10.0) -> None:
    """POST payload to the configured external API endpoint.

    The endpoint is read from EXTERNAL_API_URL or EXTERNAL_API. If no endpoint is
    configured, the function returns without doing anything.
    """

    endpoint = _resolve_external_api_url()
    if not endpoint:
        return

    headers = {"Content-Type": "application/json"}
    token = os.getenv("EXTERNAL_API_TOKEN") or os.getenv("EXTERNAL_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except error.URLError as exc:
        raise RuntimeError(f"failed to send payload to external API: {exc}") from exc
