"""Algorithm 1 assignment. Does not read or write result tables."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constraints import FunctionalDependency, ValueRangeConstraint
from validation import assign_constraint_roles, screen_missing


def test_isolated_entry_error_stays_hard():
    age = [30] * 499 + [-1]
    frame = screen_missing(pd.DataFrame({"age": age}))
    rule = ValueRangeConstraint("age_nonneg", "age", min_val=0)
    kept, log = assign_constraint_roles([rule], frame)
    assert kept[0].role == "hard"
    assert kept[0].n_sup == 500
    assert kept[0].removal_count == 1
    assert "age_nonneg" not in log.dropped


def test_zero_support_is_untested_soft():
    frame = screen_missing(pd.DataFrame({"age": [None] * 20}))
    rule = ValueRangeConstraint("age_nonneg", "age", min_val=0)
    kept, _log = assign_constraint_roles([rule], frame)
    assert kept[0].role == "soft"
    assert kept[0].untested is True
    assert kept[0].n_sup == 0


def test_high_violation_is_dropped():
    frame = pd.DataFrame({"age": [1, 2, 3, 4, 5] * 20})
    rule = ValueRangeConstraint("age_adult", "age", min_val=18)
    kept, log = assign_constraint_roles([rule], frame)
    assert kept == []
    assert log.dropped == ["age_adult"]


def test_fd_removal_count():
    keys = ["a"] * 100 + ["b"] * 100
    values = ["x"] * 99 + ["y"] + ["x"] * 100
    frame = pd.DataFrame({"k": keys, "v": values})
    rule = FunctionalDependency("k_to_v", ["k"], "v")
    kept, _log = assign_constraint_roles([rule], frame)
    assert kept[0].n_sup == 200
    assert kept[0].removal_count == 1
    assert kept[0].role == "hard"


if __name__ == "__main__":
    test_isolated_entry_error_stays_hard()
    test_zero_support_is_untested_soft()
    test_high_violation_is_dropped()
    test_fd_removal_count()
    print("validation algorithm 1: PASS")
