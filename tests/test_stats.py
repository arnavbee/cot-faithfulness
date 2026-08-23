import math

import pytest

from cotf.stats import (Rate, bootstrap_rate, diff_ci, score_classifier,
                        bootstrap_f1, wilson)


def test_wilson_bounds_are_sane():
    r = wilson(5, 10)
    assert 0 < r.lo < 0.5 < r.hi < 1
    assert r.point == 0.5


def test_wilson_handles_zero_successes_without_degenerate_interval():
    r = wilson(0, 20)
    assert r.lo == 0.0
    assert r.hi > 0.0, "a 0/20 rate is not evidence the true rate is exactly 0"


def test_bootstrap_matches_point_estimate():
    flags = [True] * 30 + [False] * 70
    r = bootstrap_rate(flags, n_boot=2000, seed=1)
    assert r.point == pytest.approx(0.30)
    assert r.lo < 0.30 < r.hi


def test_clustering_widens_the_interval():
    """20 items x 5 identical responses each is 20 data points, not 100."""
    flags, clusters = [], []
    for item in range(20):
        value = item < 6
        for _ in range(5):
            flags.append(value)
            clusters.append("item{i}".format(i=item))
    naive = bootstrap_rate(flags, n_boot=4000, seed=2)
    clustered = bootstrap_rate(flags, clusters, n_boot=4000, seed=2)
    assert (clustered.hi - clustered.lo) > (naive.hi - naive.lo)


def test_diff_ci_detects_a_real_difference():
    a = [True] * 80 + [False] * 20
    b = [True] * 20 + [False] * 80
    point, lo, hi = diff_ci(a, b, n_boot=3000, seed=3)
    assert point == pytest.approx(0.60)
    assert lo > 0, "a 60 point gap should exclude zero"


def test_diff_ci_includes_zero_when_there_is_no_difference():
    a = [True] * 50 + [False] * 50
    b = [True] * 50 + [False] * 50
    _, lo, hi = diff_ci(a, b, n_boot=3000, seed=4)
    assert lo < 0 < hi


def test_classifier_scoring():
    pred = [True, True, False, False]
    truth = [True, False, True, False]
    s = score_classifier(pred, truth)
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 1, 1, 1)
    assert s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5


def test_f1_is_nan_not_zero_when_undefined():
    s = score_classifier([False, False], [False, False])
    assert math.isnan(s.f1)


def test_bootstrap_f1_brackets_point():
    pred = [True] * 7 + [False] * 13
    truth = [True] * 5 + [False] * 2 + [True] * 3 + [False] * 10
    point, lo, hi = bootstrap_f1(pred, truth, n_boot=1000, seed=5)
    assert lo <= point <= hi
