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


TPD_SOON = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` '
    "on tokens per day (TPD): Limit 200000, Used 199705, Requested 700. "
    'Please try again in 11m4.2s.","code":"rate_limit_exceeded"}}'
)


def test_a_rolling_day_window_is_slept_not_abandoned(monkeypatch):
    """Eleven minutes is cheaper to wait than to restart.

    Groq's TPD is a rolling 24h window: quota ages back in continuously. The
    300s ceiling this replaces treated an 11-minute wait as a fresh-day cap and
    aborted, which is what turned the main run into 300 restarts against a wall
    that was clearing on its own.
    """
    bodies = [FakeResponse(429, TPD_SOON),
              FakeResponse(200, '{"choices":[{"message":{"content":"Answer: (C)"}}],'
                                '"usage":{"total_tokens":640}}')]
    slept = []
    monkeypatch.setattr(providers.requests, "post", lambda url, **kw: bodies.pop(0))
    monkeypatch.setattr(providers, "_read_key", lambda *a, **k: "key")
    monkeypatch.setattr(providers.time, "sleep", slept.append)

    comp = _provider().complete("q", "m")

    assert comp.error is None
    assert comp.text == "Answer: (C)"
    assert slept and slept[0] == pytest.approx(11 * 60 + 4.2, abs=1)


def test_day_window_waits_do_not_spend_the_retry_allowance(monkeypatch):
    """Waiting is not failing, so it must not count toward the retry budget."""
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        if len(calls) <= providers.RETRIES + 3:
            return FakeResponse(429, TPD_SOON)
        return FakeResponse(200, '{"choices":[{"message":{"content":"Answer: (A)"}}],'
                                 '"usage":{"total_tokens":640}}')

    monkeypatch.setattr(providers.requests, "post", fake_post)
    monkeypatch.setattr(providers, "_read_key", lambda *a, **k: "key")
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    comp = _provider().complete("q", "m")

    assert comp.error is None, "gave up on a queue, not a fault"
    assert len(calls) == providers.RETRIES + 4


def test_connection_faults_still_give_up(monkeypatch):
    """The patience above must not become patience for a genuinely dead host."""
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        raise providers.requests.RequestException("connection refused")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    monkeypatch.setattr(providers, "_read_key", lambda *a, **k: "key")
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    comp = _provider().complete("q", "m")

    assert "retries exhausted" in comp.error
    assert len(calls) == providers.RETRIES


def test_bucket_paces_a_burst_under_the_ceiling():
    """Four threads must not clear a minute of quota in ten seconds."""
    bucket = providers.TokenBucket(8000, estimate=800)
    waits = []
    # Ten calls at 800 tokens is 8000: the tenth is exactly the ceiling, so the
    # eleventh has to wait for refill rather than being let through.
    for _ in range(10):
        bucket.take()
    start = providers.time.monotonic()
    import threading as _t
    done = _t.Event()
    _t.Thread(target=lambda: (bucket.take(), done.set()), daemon=True).start()
    assert not done.wait(0.2), "let an over-ceiling call straight through"


def test_bucket_settles_against_real_usage():
    """A reservation that undershoots must be charged, not forgotten."""
    bucket = providers.TokenBucket(8000, estimate=700)
    reserved = bucket.take()
    bucket.settle(reserved, 2100)
    assert bucket.estimate > 700, "the estimate ignored a call three times its size"
