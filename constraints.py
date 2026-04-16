"""
constraints.py - Constraint class hierarchy for tabular data integrity.

Constraint types:
    - ValueRangeConstraint: bounds on a single column
    - FunctionalDependency: one column (set) determines another
    - CrossColumnRule: denial constraint involving multiple columns
    - DistributionHint: expected distribution shape (informational)

Also provides:
    - check_constraints(df, constraints) -> per-constraint violation counts
    - compute_violation_rate(df, constraints) -> overall violation rate
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Constraint(ABC):
    """Abstract base class for all constraint types."""

    def __init__(self, name: str, columns: list, constraint_type: str):
        self.name = name
        self.columns = columns
        self.constraint_type = constraint_type

    @abstractmethod
    def check(self, df: pd.DataFrame) -> pd.Series:
        """
        Check which rows satisfy this constraint.

        Returns
        -------
        pd.Series of bool, length len(df).
            True = row satisfies constraint, False = violation.
        """
        pass

    def violation_count(self, df: pd.DataFrame) -> int:
        """Count rows violating this constraint."""
        satisfied = self.check(df)
        return int((~satisfied).sum())

    def violation_rate(self, df: pd.DataFrame) -> float:
        """Fraction of rows violating this constraint."""
        n = len(df)
        if n == 0:
            return 0.0
        return self.violation_count(df) / n

    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"columns={self.columns})")


# ---------------------------------------------------------------------------
# ValueRangeConstraint
# ---------------------------------------------------------------------------

class ValueRangeConstraint(Constraint):
    """
    Constraint: column values must lie within [min_val, max_val].

    Either min_val or max_val can be None (one-sided bound).
    """

    def __init__(self, name: str, column: str,
                 min_val=None, max_val=None):
        super().__init__(name, [column], "value_range")
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def check(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            # Column missing -- treat as all satisfied (can't check)
            return pd.Series(True, index=df.index)

        col = df[self.column]
        satisfied = pd.Series(True, index=df.index)

        if self.min_val is not None:
            satisfied = satisfied & (col >= self.min_val)
        if self.max_val is not None:
            satisfied = satisfied & (col <= self.max_val)

        return satisfied

    def __repr__(self):
        parts = [f"name='{self.name}'", f"column='{self.column}'"]
        if self.min_val is not None:
            parts.append(f"min={self.min_val}")
        if self.max_val is not None:
            parts.append(f"max={self.max_val}")
        return f"ValueRangeConstraint({', '.join(parts)})"


# ---------------------------------------------------------------------------
# FunctionalDependency
# ---------------------------------------------------------------------------

class FunctionalDependency(Constraint):
    """
    Functional Dependency: determinant columns -> dependent column.

    For every pair of rows that agree on all determinant columns,
    they must also agree on the dependent column.

    For efficiency, we check by grouping: for each unique value of
    the determinant, the dependent column should have exactly one
    unique value.
    """

    def __init__(self, name: str, determinant: list, dependent: str):
        super().__init__(
            name,
            determinant + [dependent],
            "functional_dependency",
        )
        self.determinant = determinant
        self.dependent = dependent

    def check(self, df: pd.DataFrame) -> pd.Series:
        """
        Mark rows as violating if their determinant group has
        more than one distinct value for the dependent column.
        """
        missing = [c for c in self.determinant + [self.dependent]
                   if c not in df.columns]
        if missing:
            return pd.Series(True, index=df.index)

        # Build mapping: determinant -> most frequent dependent value
        grouped = df.groupby(self.determinant)[self.dependent]
        mode_map = grouped.agg(lambda x: x.value_counts().index[0]
                               if len(x) > 0 else np.nan)

        # Check each row: does its dependent match the mode for its group?
        if isinstance(self.determinant, list) and len(self.determinant) == 1:
            keys = df[self.determinant[0]]
            expected = keys.map(mode_map.droplevel(0)
                                if isinstance(mode_map.index,
                                              pd.MultiIndex)
                                else mode_map)
        else:
            # Multi-column determinant
            key_tuples = df[self.determinant].apply(tuple, axis=1)
            expected = key_tuples.map(
                mode_map.to_dict() if hasattr(mode_map, 'to_dict')
                else dict(mode_map)
            )

        satisfied = df[self.dependent].astype(str) == expected.astype(str)
        return satisfied.fillna(True)

    def violation_count(self, df: pd.DataFrame) -> int:
        """
        Count groups with more than one distinct dependent value,
        then count total rows in those groups.
        """
        missing = [c for c in self.determinant + [self.dependent]
                   if c not in df.columns]
        if missing:
            return 0

        grouped = df.groupby(self.determinant)[self.dependent].nunique()
        violating_groups = grouped[grouped > 1]
        if len(violating_groups) == 0:
            return 0

        # Count rows in violating groups
        count = 0
        for key in violating_groups.index:
            if isinstance(key, tuple):
                mask = pd.Series(True, index=df.index)
                for col, val in zip(self.determinant, key):
                    mask = mask & (df[col] == val)
            else:
                mask = df[self.determinant[0]] == key
            count += mask.sum()
        return int(count)

    def __repr__(self):
        return (f"FunctionalDependency(name='{self.name}', "
                f"{self.determinant} -> {self.dependent})")


# ---------------------------------------------------------------------------
# CrossColumnRule
# ---------------------------------------------------------------------------

class CrossColumnRule(Constraint):
    """
    Cross-column denial constraint: a boolean predicate over multiple columns.

    The predicate function takes a DataFrame and returns a boolean Series
    (True = satisfies constraint).
    """

    def __init__(self, name: str, columns: list,
                 predicate, description: str = ""):
        super().__init__(name, columns, "cross_column_rule")
        self.predicate = predicate
        self.description = description

    def check(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            return pd.Series(True, index=df.index)
        try:
            result = self.predicate(df)
            if isinstance(result, pd.Series):
                return result.fillna(True)
            return pd.Series(result, index=df.index)
        except Exception:
            return pd.Series(True, index=df.index)

    def __repr__(self):
        desc = self.description or "custom predicate"
        return (f"CrossColumnRule(name='{self.name}', "
                f"columns={self.columns}, desc='{desc}')")


# ---------------------------------------------------------------------------
# DistributionHint (informational, not strictly enforced)
# ---------------------------------------------------------------------------

class DistributionHint(Constraint):
    """
    Soft constraint: column should follow an expected distribution.

    This is informational -- check() always returns True.
    Used for evaluating distributional fidelity, not hard enforcement.
    """

    def __init__(self, name: str, column: str,
                 distribution: str, params: dict = None):
        super().__init__(name, [column], "distribution_hint")
        self.column = column
        self.distribution = distribution
        self.params = params or {}

    def check(self, df: pd.DataFrame) -> pd.Series:
        # Distribution hints are soft -- always satisfied
        return pd.Series(True, index=df.index)

    def __repr__(self):
        return (f"DistributionHint(name='{self.name}', "
                f"column='{self.column}', dist='{self.distribution}')")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def check_constraints(df: pd.DataFrame,
                      constraints: list) -> dict:
    """
    Check all constraints on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data to check.
    constraints : list of Constraint
        Constraints to evaluate.

    Returns
    -------
    dict mapping constraint name -> {
        "violations": int,
        "rate": float,
        "type": str,
    }
    """
    results = {}
    for c in constraints:
        if isinstance(c, DistributionHint):
            # Skip distribution hints in violation counting
            results[c.name] = {
                "violations": 0,
                "rate": 0.0,
                "type": c.constraint_type,
            }
            continue

        n_violations = c.violation_count(df)
        rate = n_violations / max(len(df), 1)
        results[c.name] = {
            "violations": n_violations,
            "rate": rate,
            "type": c.constraint_type,
        }
    return results


def compute_violation_rate(df: pd.DataFrame,
                           constraints: list) -> float:
    """
    Compute average violation rate across all hard constraints.

    Ignores DistributionHint constraints.
    """
    hard_constraints = [c for c in constraints
                        if not isinstance(c, DistributionHint)]
    if not hard_constraints:
        return 0.0

    rates = []
    for c in hard_constraints:
        rates.append(c.violation_rate(df))

    return float(np.mean(rates))


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a small test DataFrame
    df = pd.DataFrame({
        "age": [25, 30, 15, 45, 50],
        "income": [50000, 60000, -1000, 80000, 90000],
        "education": ["HS", "BS", "HS", "MS", "BS"],
        "edu_num": [9, 13, 9, 14, 13],
    })

    constraints = [
        ValueRangeConstraint("age_range", "age", min_val=17, max_val=90),
        ValueRangeConstraint("income_positive", "income", min_val=0),
        FunctionalDependency("edu_fd", ["education"], "edu_num"),
    ]

    results = check_constraints(df, constraints)
    for name, info in results.items():
        print(f"  {name:20s}: {info['violations']} violations "
              f"(rate={info['rate']:.2f})")

    overall = compute_violation_rate(df, constraints)
    print(f"\nOverall violation rate: {overall:.4f}")
