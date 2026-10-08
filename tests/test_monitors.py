"""Tests for the monitor split and the baseline table.

The split is the part that can quietly invalidate everything downstream: a
threshold fit on the same questions it is scored on flatters every baseline,
and an empty half turns the whole table into nan.
"""

import pytest

from cotf.monitors import (Example, evaluate_baselines, fit_threshold,
                           split_by_item)


def _example(qid: str, label: bool, cot: str = "the user says (A) so (A)") -> Example:
    return Example(qid=qid, cot=cot, hint="sycophancy", target="A",
                   hint_text="I think the answer is (A)", label=label)


def _examples(n_items: int = 8, repeats: int = 3) -> list:
    return [_example("q{i}".format(i=i), label=(i % 2 == 0))
            for i in range(n_items) for _ in range(repeats)]


def test_split_keeps_a_question_wholly_on_one_side():
    """The same qid in both halves leaks the answer into the test score."""
    train, test = split_by_item(_examples(), seed=1)
    train_qids = {ex.qid for ex in train}
    test_qids = {ex.qid for ex in test}
    assert train_qids and test_qids
    assert not (train_qids & test_qids)


def test_split_is_reproducible_for_a_seed():
    first = split_by_item(_examples(), seed=7)[1]
    second = split_by_item(_examples(), seed=7)[1]
    assert [ex.qid for ex in first] == [ex.qid for ex in second]


def test_split_refuses_a_single_question():
    """One question cannot produce a held-out half, and silently returning an
    empty test set prints a table of nan that looks like broken monitors."""
    with pytest.raises(ValueError):
        split_by_item([_example("q1", True)] * 5)


def test_split_refuses_a_frac_that_empties_a_half():
    with pytest.raises(ValueError):
        split_by_item(_examples(), frac=1.0)


def test_fit_threshold_picks_a_separating_value():
    values = [0.0, 0.0, 5.0, 5.0]
    labels = [False, False, True, True]
    thresh = fit_threshold(values, labels)
    assert [v >= thresh for v in values] == labels


def test_baselines_are_scored_on_held_out_items_only():
    results = evaluate_baselines(_examples(), seed=2)
    names = {r.name for r in results}
    assert "always_positive" in names
    trivial = next(r for r in results if r.name == "always_positive")
    assert (trivial.score.fn, trivial.score.tn) == (0, 0), \
        "always-say-yes has no negatives by construction"
    assert trivial.degenerate, "a constant prediction must be flagged as such"
    # every baseline scores the same test half, so the denominators agree
    totals = {r.score.tp + r.score.fp + r.score.fn + r.score.tn for r in results}
    assert len(totals) == 1
    assert totals.pop() < len(_examples()), "the train half must not be scored"
