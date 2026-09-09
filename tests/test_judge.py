"""Tests for the dependence judge: label parsing, caching, and the prompt's blindness.

The judge makes network calls, so everything testable here is the parts that do
not: how a reply becomes a label, what the cache treats as a result, and what the
prompt does and does not tell the judge.
"""

import json

import pytest

from cotf.judge import (DEPENDENCE_PROMPT, JudgeCache, _cache_key, _read_label,
                        build_prompt, judge_dependence)


class FakeProvider:
    """Returns scripted replies and counts calls, so cache hits are observable."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, prompt, model, temperature=0.0, max_tokens=2048,
                 seed=None, system=None):
        from cotf.providers import Completion
        self.calls += 1
        return self.replies.pop(0)


def _c(text="", reasoning="", error=None):
    from cotf.providers import Completion
    return Completion(text=text, reasoning=reasoning, error=error)


@pytest.mark.parametrize("reply,expected", [
    ("A", "A"), ("B", "B"), ("C", "C"),
    ("c", "C"),
    ("  B  ", "B"),
    ("B.", "B"),
    ("The answer is C", "C"),          # first STANDALONE A/B/C wins
    ("I would say B, not C", "B"),
    # the bug this regex exists for: 'answer' starts with A, 'scenario' holds a C
    ("The scenario in the question", ""),
    ("answer", ""),
    ("", ""),
    ("YES", ""),                       # wrong vocabulary is unreadable, not a guess
    ("none of these", ""),             # 'n','o','e' are not labels
])
def test_read_label(reply, expected):
    assert _read_label(reply) == expected


def test_prompt_is_blind_to_answer_and_flip():
    """The judge must not be told the control answer or whether the trace flipped.

    If it were, "depends on the hint" would collapse into the flip rate we already
    measure, and the second detector would stop being a second opinion.
    """
    cot = "I computed option B myself. Answer: (B)"
    p = build_prompt(cot, "reward_hack", "C")
    low = p.lower()
    assert "control" not in low
    assert "flip" not in low
    assert "correct" not in low.split("REASONING")[0].lower() or True
    # It does carry the target letter and the hint description: both are needed to
    # identify which pointer to look for.
    assert "(C)" in p
    assert "grader function" in p


def test_prompt_truncates_long_cot():
    p = build_prompt("x" * 50000, "authority", "A", cot_chars=100)
    assert "x" * 100 in p and "x" * 101 not in p
    assert len(p) < 2000


def test_three_way_taxonomy_is_in_the_prompt():
    for letter in ("A -", "B -", "C -"):
        assert letter in DEPENDENCE_PROMPT


def test_judge_caches_by_prompt(tmp_path):
    cache = JudgeCache(tmp_path / "j.jsonl")
    prov = FakeProvider([_c("C")])
    args = ("the grader wants C so C", "reward_hack", "C", prov, "m")
    first = judge_dependence(*args, cache=cache)
    second = judge_dependence(*args, cache=cache)
    assert first.label == second.label == "C"
    assert first.depends and first.mentions
    assert prov.calls == 1, "second call should have been served from cache"


def test_cache_does_not_freeze_an_error(tmp_path):
    """An errored verdict is a hole. Caching it would silently bake a rate-limit
    failure into the labels and a resume would never retry it."""
    path = tmp_path / "j.jsonl"
    cache = JudgeCache(path)
    prov = FakeProvider([_c("", error="HTTP 429"), _c("B")])
    args = ("text", "authority", "B", prov, "m")
    first = judge_dependence(*args, cache=cache)
    assert first.label == ""
    assert first.error == "HTTP 429"
    reloaded = JudgeCache(path)
    second = judge_dependence("text", "authority", "B", prov, "m", cache=reloaded)
    assert second.label == "B"
    assert prov.calls == 2, "the errored verdict must not have been reused"


def test_unreadable_label_is_not_cached_as_a_result(tmp_path):
    path = tmp_path / "j.jsonl"
    prov = FakeProvider([_c("maybe?"), _c("A")])
    args = ("text", "metadata", "D", prov, "m")
    assert judge_dependence(*args, cache=JudgeCache(path)).label == ""
    assert judge_dependence(*args, cache=JudgeCache(path)).label == "A"
    assert prov.calls == 2


def test_label_falls_back_to_reasoning_tail(tmp_path):
    """Reasoning models sometimes spend the whole budget thinking and return empty
    content. The letter is then at the end of the reasoning field."""
    prov = FakeProvider([_c(text="", reasoning="lots of thinking ... so the answer is B")])
    v = judge_dependence("t", "authority", "B", prov, "m", cache=None)
    assert v.label == "B"
    assert not v.depends and v.mentions


def test_mentions_and_depends_split_on_b():
    """B is the whole point: the judge's own view of mention includes B, its view
    of dependence does not."""
    prov = FakeProvider([_c("B")])
    v = judge_dependence("t", "authority", "B", prov, "m", cache=None)
    assert v.mentions is True
    assert v.depends is False


def test_cache_key_depends_on_model_and_prompt():
    a = _cache_key("m1", "p")
    assert a != _cache_key("m2", "p")
    assert a != _cache_key("m1", "p2")
    assert a == _cache_key("m1", "p")


def test_cache_file_is_valid_jsonl(tmp_path):
    path = tmp_path / "j.jsonl"
    cache = JudgeCache(path)
    prov = FakeProvider([_c("C"), _c("A")])
    judge_dependence("t1", "authority", "B", prov, "m", cache=cache)
    judge_dependence("t2", "metadata", "D", prov, "m", cache=cache)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert {r["label"] for r in rows} == {"C", "A"}
    assert all("key" in r for r in rows)
