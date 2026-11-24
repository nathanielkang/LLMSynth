"""Held-out validation for LLMSynth Algorithm 1.

Matches the revised manuscript: observed-entry support, removal count,
empirical screening at tau, hard promotion by n_min and tau_noise, and
untested soft weights. No bootstrap and no Benjamini-Hochberg step.

Locked table cells are not written here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from constraints import (
    CrossColumnRule,
    DistributionHint,
    FunctionalDependency,
    ValueRangeConstraint,
)

TAU = 0.05
TAU_NOISE = 0.005
N_MIN = 60  # ceil(3 / tau)
EPS = 1e-3
N_VAL = 500
K_DISCOVERY = 15
DEFAULT_SENTINELS = {"?", "??", "unknown", "na", "n/a", "none", "null"}


@dataclass
class AssignmentLog:
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    downgraded: list = field(default_factory=list)


def _untested_weight() -> float:
    return -math.log(TAU + EPS)


def _soft_weight(rate: float) -> float:
    return -math.log(float(rate) + EPS)


def screen_missing(df: pd.DataFrame, sentinels=None) -> pd.DataFrame:
    """Recode empty, non-numeric junk in numeric columns, and sentinel codes as missing.

    Does not impute. Prompt imputation is separate and is not used for scoring.
    """
    codes = {str(s).strip().lower() for s in (sentinels or DEFAULT_SENTINELS)}
    out = df.copy()
    for col in out.columns:
        series = out[col]
        as_str = series.astype(str).str.strip()
        missing = series.isna() | as_str.eq("") | as_str.str.lower().isin(codes)
        numeric = pd.to_numeric(series, errors="coerce")
        # Dictionary sentinels that are numeric (-1, 999) are already in `codes`.
        if numeric.notna().mean() >= 0.8:
            out[col] = numeric.mask(missing)
        else:
            out[col] = series.mask(missing)
    return out


def _observed_mask(df: pd.DataFrame, columns: list) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask = mask & df[col].notna()
    return mask


def support_and_removal(constraint, df: pd.DataFrame):
    """Return n_sup, r, v, and whether the rule is binary.

    Unary rate v is the eligible-row violation rate. Binary FDs also return
    the removal count r (rows deleted to leave one dependent value per key).
    """
    if isinstance(constraint, DistributionHint):
        return 0, 0, 0.0, False

    if isinstance(constraint, ValueRangeConstraint):
        eligible = df.loc[_observed_mask(df, [constraint.column])]
        n_sup = int(len(eligible))
        if n_sup == 0:
            return 0, 0, 0.0, False
        r = int((~constraint.check(eligible)).sum())
        return n_sup, r, r / n_sup, False

    if isinstance(constraint, FunctionalDependency):
        cols = list(constraint.determinant) + [constraint.dependent]
        eligible = df.loc[_observed_mask(df, cols)]
        if eligible.empty:
            return 0, 0, 0.0, True
        n_sup = 0
        r = 0
        participating = 0
        for _, group in eligible.groupby(list(constraint.determinant), dropna=False):
            if len(group) < 2:
                continue
            n_sup += int(len(group))
            counts = group[constraint.dependent].astype(str).value_counts()
            r += int(len(group) - int(counts.iloc[0]))
            if len(counts) > 1:
                participating += int(len(group))
        if n_sup == 0:
            return 0, 0, 0.0, True
        return n_sup, r, participating / n_sup, True

    if isinstance(constraint, CrossColumnRule):
        eligible = df.loc[_observed_mask(df, list(constraint.columns))]
        n_sup = int(len(eligible))
        if n_sup == 0:
            return 0, 0, 0.0, False
        r = int((~constraint.check(eligible)).sum())
        return n_sup, r, r / n_sup, False

    n = int(len(df))
    if n == 0:
        return 0, 0, 0.0, False
    r = int(constraint.violation_count(df))
    return n, r, r / n, False


def _stamp(constraint, *, role, weight, n_sup, removal, rate, untested):
    constraint.role = role
    constraint.weight = weight
    constraint.n_sup = int(n_sup)
    constraint.removal_count = int(removal)
    constraint.empirical_rate = float(rate)
    constraint.untested = bool(untested)
    return constraint


def assign_constraint_roles(
    constraints: list,
    df_val: pd.DataFrame,
    *,
    tau: float = TAU,
    tau_noise: float = TAU_NOISE,
    n_min: int = N_MIN,
    eps: float = EPS,
) -> tuple[list, AssignmentLog]:
    """Apply Algorithm 1 on an already screened validation frame.

    Returns the retained hard and soft constraints. Proposals whose empirical
    rate exceeds tau are omitted.
    """
    del eps  # weight formulas use the module constant EPS
    log = AssignmentLog()
    retained = []
    for constraint in constraints:
        if isinstance(constraint, DistributionHint):
            _stamp(
                constraint,
                role="soft",
                weight=0.1,
                n_sup=0,
                removal=0,
                rate=0.0,
                untested=False,
            )
            retained.append(constraint)
            log.kept.append(constraint.name)
            continue

        n_sup, removal, rate, binary = support_and_removal(constraint, df_val)
        if n_sup == 0:
            _stamp(
                constraint,
                role="soft",
                weight=_untested_weight(),
                n_sup=0,
                removal=0,
                rate=0.0,
                untested=True,
            )
            retained.append(constraint)
            log.kept.append(constraint.name)
            continue

        screen = (removal / n_sup) if binary else rate
        if screen > tau:
            log.dropped.append(constraint.name)
            continue

        hard = n_sup >= n_min and removal <= tau_noise * n_sup
        if hard:
            _stamp(
                constraint,
                role="hard",
                weight=math.inf,
                n_sup=n_sup,
                removal=removal,
                rate=rate,
                untested=False,
            )
        elif n_sup < n_min:
            _stamp(
                constraint,
                role="soft",
                weight=_untested_weight(),
                n_sup=n_sup,
                removal=removal,
                rate=rate,
                untested=True,
            )
        else:
            _stamp(
                constraint,
                role="soft",
                weight=_soft_weight(rate),
                n_sup=n_sup,
                removal=removal,
                rate=rate,
                untested=False,
            )
        retained.append(constraint)
        log.kept.append(constraint.name)

    _resolve_range_fd_conflicts(retained, df_val, log)
    return retained, log


def _resolve_range_fd_conflicts(constraints, df_val, log: AssignmentLog):
    """Downgrade a hard FD when its modal value falls outside a hard range.

    Priority follows the manuscript: hard range above FD. Within-tier weights
    are not re-sorted here because a range conflict has a unique higher tier.
    """
    ranges = [
        c for c in constraints
        if isinstance(c, ValueRangeConstraint) and getattr(c, "role", "") == "hard"
    ]
    for fd in constraints:
        if not isinstance(fd, FunctionalDependency) or getattr(fd, "role", "") != "hard":
            continue
        range_on_dep = next((c for c in ranges if c.column == fd.dependent), None)
        if range_on_dep is None or df_val is None or df_val.empty:
            continue
        cols = list(fd.determinant) + [fd.dependent]
        eligible = df_val.loc[_observed_mask(df_val, cols)]
        if eligible.empty:
            continue
        conflict = False
        for _, group in eligible.groupby(list(fd.determinant), dropna=False):
            if len(group) < 2:
                continue
            modal = group[fd.dependent].astype(str).value_counts().index[0]
            probe = group.iloc[[0]].copy()
            probe[fd.dependent] = pd.to_numeric(pd.Series([modal]), errors="coerce").iloc[0]
            if pd.isna(probe[fd.dependent].iloc[0]):
                probe[fd.dependent] = modal
            if not bool(range_on_dep.check(probe).iloc[0]):
                conflict = True
                break
        if conflict:
            fd.role = "soft"
            fd.weight = _soft_weight(getattr(fd, "empirical_rate", TAU))
            fd.untested = False
            log.downgraded.append(fd.name)


def split_validation_frames(df: pd.DataFrame, *, k: int = K_DISCOVERY, n_val: int = N_VAL, seed: int = 42):
    """Disjoint discovery sample and validation sample from a screened frame."""
    if len(df) == 0:
        return df.copy(), df.copy()
    shuffled = df.sample(frac=1.0, random_state=seed)
    k_use = min(k, len(shuffled))
    discovery = shuffled.iloc[:k_use]
    rest = shuffled.iloc[k_use:]
    if len(rest) == 0:
        return discovery, shuffled.iloc[0:0]
    validation = rest.sample(n=min(n_val, len(rest)), random_state=seed + 1)
    return discovery, validation


def validate_proposals(constraints: list, df: pd.DataFrame, *, tau: float = TAU) -> list:
    """Screen, hold out D_val, and return the Algorithm 1 constraint set."""
    screened = screen_missing(df)
    _, validation = split_validation_frames(screened)
    if validation.empty:
        validation = screened
    retained, _log = assign_constraint_roles(constraints, validation, tau=tau)
    return retained
