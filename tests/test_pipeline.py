"""End-to-end check against a model whose behaviour we know exactly.

The mock provider follows a planted hint 55% of the time and admits to it 35% of
the time. If the harness is correct, the acknowledgement rate it reports has to
bracket 0.35. This is the test that catches a broken detector, a broken
denominator, or a broken bootstrap, and it runs offline in a second.
"""

import shutil

import pytest

from cotf.dataset import Item
from cotf.providers import MockProvider
from cotf.runner import RUNS_DIR, RunConfig, run
from cotf.score import build_cases, summarise


def synthetic_items(n=40):
    items = []
    for i in range(n):
        items.append(Item(
            qid="synth/{i}".format(i=i),
            question="Synthetic question number {i} about a distinct topic.".format(i=i),
            choices=["first alternative", "second alternative",
                     "third alternative", "fourth alternative"],
            answer="ABCD"[i % 4],
            subject="synthetic",
            source="test",
        ))
    return items


@pytest.fixture(scope="module")
def summary():
    cfg = RunConfig(name="_pytest", provider="mock", n_items=40, repeats=3,
                    concurrency=8, seed=7)
    run_dir = RUNS_DIR / cfg.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    items = synthetic_items(cfg.n_items)
    store = run(cfg, items)
    yield summarise(cfg, items, store, detector="strict")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_recovers_known_acknowledgement_rate(summary):
    ack = summary["overall"]["acknowledgement_rate"]
    assert ack["lo"] <= MockProvider.MENTION_RATE <= ack["hi"], (
        "harness reports {p:.3f} [{lo:.3f}, {hi:.3f}] but the mock admits to the "
        "hint {t:.3f} of the time".format(p=ack["point"], lo=ack["lo"],
                                          hi=ack["hi"], t=MockProvider.MENTION_RATE))


def test_flip_rate_exceeds_follow_rate(summary):
    """Raw flips overstate hint influence, and that is the point of the control.

    Some answers land on the hinted option by chance, so flip rate is an upper
    bound on the causal effect, never an estimate of it. If this ever inverts,
    the flip definition has broken.
    """
    flip = summary["overall"]["flip_rate"]["point"]
    assert flip >= MockProvider.FOLLOW_RATE


def test_noise_floor_is_measured_when_repeats_allow_it(summary):
    assert summary["spontaneous_switch_rate"] is not None
    assert summary["spontaneous_switch_rate"]["point"] > 0


def test_no_parse_failures_on_a_well_behaved_model(summary):
    assert summary["parse_failure_rate"]["point"] == 0.0


def test_every_hint_type_produced_flips(summary):
    for hint, entry in summary["by_hint"].items():
        assert entry["n_flips"] > 0, "hint {h} never moved the answer".format(h=hint)


def test_caching_makes_a_rerun_free(tmp_path):
    from cotf.runner import ResponseStore, Response
    store = ResponseStore(tmp_path / "r.jsonl")
    resp = Response(key="abc", qid="q", condition="control", target="", repeat=0,
                    answer="A", cot="c", text="t", prompt="p", latency=0.0)
    store.add(resp)
    reopened = ResponseStore(tmp_path / "r.jsonl")
    assert reopened.has("abc")
    assert reopened.by_key["abc"]["answer"] == "A"
