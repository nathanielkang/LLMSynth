"""Hydra adapter stub for LLMSynth discovery.

``discover_dcs`` is the seam a later run should fill with a real denial
constraint miner (training split in, ``CrossColumnRule`` objects out).
Scoring goes through ``classical_discovery.score_discovered``, which calls
``metrics.constraint_discovery_quality``.

The live scorer credits two cross-column rules when their column sets
overlap. The manuscript text asks for canonical equivalence. A real Hydra
wiring should emit one rule per minimal denial and keep column sets exact,
or this overlap test will not match the Table 4 ledger.

No Hydra binary is vendored. This module does not write a results CSV and
does not store the Table 4 best-guess cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))


def _load_classical_discovery():
    """Load the sibling module without importing ``baselines/__init__.py``."""
    import importlib.util

    path = Path(__file__).resolve().parent / "classical_discovery.py"
    mod_name = "llmsynth_classical_discovery"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_classical = _load_classical_discovery()
MinerNotInstalled = _classical.MinerNotInstalled
score_discovered = _classical.score_discovered

__all__ = ["MinerNotInstalled", "discover_and_score", "discover_dcs", "score_dcs"]


def discover_dcs(frame):
    """Return denial constraints mined from ``frame``.

    Not implemented. Wire Hydra here, then pass the objects to ``score_dcs``.
    Do not substitute the best-guess ledger.
    """
    del frame
    raise MinerNotInstalled(
        "Hydra is not installed under 2_Code. "
        "Implement discover_dcs, then score with score_dcs. "
        "Do not write a guess CSV."
    )


def score_dcs(discovered: list, ground_truth: list) -> dict:
    """Score denial-constraint objects with the manuscript discovery scorer."""
    return score_discovered(discovered, ground_truth, miner="Hydra")


def discover_and_score(frame, ground_truth: list) -> dict:
    """Mine then score. Reaches the scorer only after a real ``discover_dcs``."""
    discovered = discover_dcs(frame)
    out = score_dcs(discovered, ground_truth)
    out["table4_measurement"] = True
    return out


def _smoke_scorer_contract() -> None:
    """Adult-shaped denominator: 3 DCs inside 8 rules, plus one extra DC."""
    from constraints import (
        CrossColumnRule,
        FunctionalDependency,
        ValueRangeConstraint,
    )

    def _dc(name, columns):
        return CrossColumnRule(
            name, columns, lambda df: df.index == df.index, description=name
        )

    gt = [
        ValueRangeConstraint("age", "age", min_val=0, max_val=120),
        ValueRangeConstraint("hours", "hours", min_val=0, max_val=168),
        ValueRangeConstraint("gain", "capital_gain", min_val=0, max_val=None),
        FunctionalDependency("edu", ["education"], "education_num"),
        FunctionalDependency("rel", ["relationship"], "sex"),
        _dc("dc1", ["marital", "relationship"]),
        _dc("dc2", ["sex", "relationship"]),
        _dc("dc3", ["workclass", "occupation"]),
    ]
    found = [
        _dc("dc1", ["marital", "relationship"]),
        _dc("dc2", ["sex", "relationship"]),
        _dc("dc3", ["workclass", "occupation"]),
        _dc("extra", ["fnlwgt", "capital_loss"]),
    ]
    scored = score_dcs(found, gt)
    # tp=3, fp=1, n=8 -> P=0.75, R=0.375, F1=0.50. Column sets are disjoint
    # enough that the overlap matcher agrees with exact equivalence here.
    if scored["n_matched"] != 3:
        raise SystemExit(f"expected 3 DC matches, got {scored}")
    if abs(scored["f1"] - 0.5) > 1e-12:
        raise SystemExit(f"Hydra operating-point F1 {scored['f1']}")
    if scored["table4_measurement"]:
        raise SystemExit("scorer contract was marked as a Table 4 measurement")

    stray_fd = [FunctionalDependency("noise", ["x"], "y")]
    missed = score_dcs(stray_fd, gt)
    if missed["n_matched"] != 0:
        raise SystemExit("an FD must not match a DC or a range")

    try:
        discover_dcs(None)
    except MinerNotInstalled:
        pass
    else:
        raise SystemExit("discover_dcs returned without a miner")

    print(
        "Hydra scorer contract OK "
        "(3/8 with one extra DC -> F1 0.50; FD does not match; not a measurement)"
    )


if __name__ == "__main__":
    _smoke_scorer_contract()
