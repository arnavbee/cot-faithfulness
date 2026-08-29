"""A rate-limited run must not look like a healthy one.

The first clocked run stalled for fourteen minutes while the log kept printing
"0 errors", because the 429 branch said nothing and retried a per-day refusal as
though it were a per-minute one. These pin both halves of that.
"""

import pytest

from cotf import providers
from cotf.providers import DAILY_LIMIT_ERROR, OpenAICompatProvider, _body_wait


TPD_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` '
    "in organization `org_x` service tier `on_demand` on tokens per day (TPD): "
    'Limit 200000, Used 195139, Requested 60072. Please try again in '
    '6h37m31.152s.","type":"tokens","code":"rate_limit_exceeded"}}'
)


class FakeResponse:
    def __init__(self, status_code, text, headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        import json
        return json.loads(self.text)


def _provider():
    return OpenAICompatProvider("groq", "https://example.invalid/v1",
                                ["NO_SUCH_ENV"], "no_such.key", "m")


def test_body_wait_reads_hours_not_the_header():
    # The per-minute header says half a second; the truth is in the body.
    assert _body_wait(TPD_BODY) == pytest.approx(6 * 3600 + 37 * 60 + 31.152, abs=1)


def test_daily_limit_returns_once_and_does_not_retry(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        # The reset headers describe the per-minute window and are misleading here.
        return FakeResponse(429, TPD_BODY, {"x-ratelimit-reset-tokens": "547ms"})

    monkeypatch.setattr(providers.requests, "post", fake_post)
    monkeypatch.setattr(providers, "_read_key", lambda *a, **k: "key")
    monkeypatch.setattr(providers.time, "sleep", lambda s: pytest.fail(
        "slept on a per-day limit instead of giving up"))

    comp = _provider().complete("q", "m")

    assert DAILY_LIMIT_ERROR in comp.error
    assert len(calls) == 1, "a day cap must not spend four more requests"


def test_per_minute_limit_still_retries(monkeypatch):
    bodies = [FakeResponse(429, '{"error":{"message":"tokens per minute (TPM)"}}',
                           {"retry-after": "1"}),
              FakeResponse(200, '{"choices":[{"message":{"content":"Answer: (B)"}}],'
                                '"usage":{}}')]
    slept = []
    monkeypatch.setattr(providers.requests, "post", lambda url, **kw: bodies.pop(0))
    monkeypatch.setattr(providers, "_read_key", lambda *a, **k: "key")
    monkeypatch.setattr(providers.time, "sleep", slept.append)

    comp = _provider().complete("q", "m")

    assert comp.error is None
    assert comp.text == "Answer: (B)"
    assert slept, "a per-minute limit is worth waiting out"
