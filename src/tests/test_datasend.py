from __future__ import annotations

import json

from api import datasend


def test_send_to_external_api_posts_json(monkeypatch):
    captured = {}

    def get_header(name: str):
        for key, value in captured["headers"].items():
            if key.lower() == name.lower():
                return value
        raise KeyError(name)

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return DummyResponse()

    monkeypatch.setenv("EXTERNAL_API_URL", "https://example.com/webhook")
    monkeypatch.setenv("EXTERNAL_API_TOKEN", "secret-token")
    monkeypatch.setattr(datasend.request, "urlopen", fake_urlopen)

    datasend.send_to_external_api({"task_results": {"hello": "world"}}, timeout=3.5)

    assert captured["url"] == "https://example.com/webhook"
    assert get_header("Content-Type") == "application/json"
    assert get_header("Authorization") == "Bearer secret-token"
    assert captured["timeout"] == 3.5
    assert captured["body"] == {"task_results": {"hello": "world"}}
