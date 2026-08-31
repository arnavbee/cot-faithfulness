import math

import pytest

from cotf.cost import PRICES, Call, Price, collect, format_report, resolve_price
from cotf.runner import RunConfig
from cotf.stats import bootstrap_mean


class FakeStore:
    def __init__(self, records):
        self._records = records

    def all(self):
        return self._records


def _rec(**kw):
    rec = {"qid": "q1", "condition": "control", "error": None, "latency": 10.0,
           "cot": "a" * 90, "text": "b" * 10,
           "usage": {"prompt_tokens": 100, "completion_tokens": 200,
                     "total_tokens": 300, "queue_time": 0.2,
                     "prompt_time": 0.05, "completion_time": 1.0}}
    rec.update(kw)
    return rec


def _call(**kw):
    args = {"qid": "q1", "condition": "control", "prompt_tokens": 100,
            "completion_tokens": 200, "latency": 10.0, "queue_time": 0.2,
            "prompt_time": 0.05, "completion_time": 1.0,
            "cot_chars": 90, "text_chars": 10}
    args.update(kw)
    return Call(**args)


def test_errored_records_are_holes_not_zeroes():
    """An API error bought nothing. Averaging it in would understate cost."""
    store = FakeStore([_rec(), _rec(error="retries exhausted", usage={})])
    calls = collect(store)
    assert len(calls) == 1
    assert calls[0].total_tokens == 300


def test_records_without_usage_are_skipped():
    store = FakeStore([_rec(usage={})])
    assert collect(store) == []


def test_cost_uses_separate_input_and_output_prices():
    call = _call(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert call.cost(Price(0.15, 0.60)) == pytest.approx(0.75)


def test_unpriced_model_yields_nan_not_zero():
    """A missing price must not silently read as a free run."""
    assert math.isnan(_call().cost(None))


def test_service_latency_excludes_the_local_throttle():
    """Wall time includes TokenBucket.take() sleeping; service time must not.

    This is the whole reason the two are reported separately: on a free key the
    throttle is ~99% of the wall clock and none of the inference.
    """
    call = _call(latency=600.0)
    assert call.service == pytest.approx(1.25)
    assert call.throttle_wait == pytest.approx(598.75)


def test_throttle_wait_never_goes_negative():
    """Provider times can exceed the wall stamp by a rounding hair."""
    assert _call(latency=0.5).throttle_wait == 0.0


def test_reasoning_share_tracks_the_character_split():
    assert _call(cot_chars=90, text_chars=10).reasoning_share == pytest.approx(0.9)
    assert _call(cot_chars=0, text_chars=0).reasoning_share == 0.0


def test_explicit_prices_beat_the_table():
    cfg = RunConfig(name="x", provider="groq")
    price = resolve_price(cfg, "openai/gpt-oss-120b", price_in=1.0, price_out=2.0)
    assert (price.inp, price.out) == (1.0, 2.0)


def test_unknown_model_has_no_price():
    cfg = RunConfig(name="x", provider="groq")
    assert resolve_price(cfg, "some/model-nobody-priced") is None


def test_price_table_entries_carry_a_source():
    """A dollar figure with no provenance is a guess. Keep it that way."""
    for key, price in PRICES.items():
        assert price.source, "{k} has no source".format(k=key)


def test_bootstrap_mean_interval_brackets_the_point():
    m = bootstrap_mean([1.0, 2.0, 3.0, 4.0], n_boot=2000, seed=1)
    assert m.point == pytest.approx(2.5)
    assert m.lo < 2.5 < m.hi


def test_bootstrap_mean_clusters_widen_the_interval():
    values = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0] * 5
    clusters = (["a"] * 3 + ["b"] * 3) * 5
    tight = bootstrap_mean(values, n_boot=2000, seed=1)
    loose = bootstrap_mean(values, clusters=clusters, n_boot=2000, seed=1)
    assert (loose.hi - loose.lo) > (tight.hi - tight.lo)


def test_empty_run_formats_without_crashing():
    rep = {"run": "x", "provider": "groq", "model": "m", "priced": False,
           "calls": 0, "errored_calls": 7}
    out = format_report(rep)
    assert "no billable calls" in out


def test_missing_provider_timing_is_not_reported_as_throttle():
    """The mock provider returns no queue/prompt/completion times.

    Its residual wall time is unattributed, not this harness throttling itself,
    and reporting it as throttle would manufacture a finding from a blank field.
    """
    assert _call(queue_time=0.0, prompt_time=0.0, completion_time=0.0).has_timing is False
    assert _call().has_timing is True
