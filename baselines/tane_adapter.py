"""TANE adapter stub for LLMSynth discovery.

``discover_fds`` is the seam a later run should fill with a real FD miner
(training split in, ``FunctionalDependency`` objects out). Scoring goes
through ``classical_discovery.score_discovered``, which calls
``metrics.constraint_discovery_quality``.

No TANE binary is vendored. This module does not write a results CSV and
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

__all__ = ["MinerNotInstalled", "discover_and_score", "discover_fds", "score_fds"]


def discover_fds(frame):
    """Return FD constraints mined from ``frame``.

    Not implemented. Wire an exact FD miner here, then pass the objects to
    ``score_fds``. Do not substitute the best-guess ledger.
    """
    del frame
    raise MinerNotInstalled(
        "TANE is not installed under 2_Code. "
        "Implement discover_fds, then score with score_fds. "
        "Do not write a guess CSV."
    )


def score_fds(discovered: list, ground_truth: list) -> dict:
    """Score FD objects with the manuscript discovery scorer."""
    return score_discovered(discovered, ground_truth, miner="TANE")


def discover_and_score(frame, ground_truth: list) -> dict:
    """Mine then score. Reaches the scorer only after a real ``discover_fds``."""
    discovered = discover_fds(frame)
    out = score_fds(discovered, ground_truth)
    out["table4_measurement"] = True
    return out


def _smoke_scorer_contract() -> None:
    """Adult-shaped denominator: 2 FDs inside 8 rules. Not a dataset run."""
    from constraints import (
        CrossColumnRule,
        FunctionalDependency,
        ValueRangeConstraint,
    )

    def _dc(name, columns):
        return CrossColumnRule(name, columns, lambda df: df.index == df.index)

    gt = [
        FunctionalDependency("edu", ["education"], "education_num"),
        FunctionalDependency("rel", ["relationship"], "sex"),
        ValueRangeConstraint("age", "age", min_val=0, max_val=120),
        ValueRangeConstraint("hours", "hours", min_val=0, max_val=168),
        ValueRangeConstraint("gain", "capital_gain", min_val=0, max_val=None),
        _dc("dc1", ["a", "b"]),
        _dc("dc2", ["c", "d"]),
        _dc("dc3", ["e", "f"]),
    ]
    found = [
        FunctionalDependency("edu", ["education"], "education_num"),
        FunctionalDependency("rel", ["relationship"], "sex"),
    ]
    perfect = score_fds(found, gt)
    if abs(perfect["precision"] - 1.0) > 1e-12:
        raise SystemExit(f"TANE ceiling precision {perfect['precision']}")
    if abs(perfect["recall"] - 0.25) > 1e-12:
        raise SystemExit(f"TANE ceiling recall {perfect['recall']}")
    if abs(perfect["f1"] - 0.40) > 1e-12:
        raise SystemExit(f"TANE ceiling F1 {perfect['f1']}")
    if perfect["table4_measurement"]:
        raise SystemExit("scorer contract was marked as a Table 4 measurement")

    empty_fd_gt = [
        ValueRangeConstraint(f"r{i}", f"c{i}", min_val=0, max_val=1)
        for i in range(7)
    ]
    unmatched = [FunctionalDependency("noise", ["x"], "y")]
    zero = score_fds(unmatched, empty_fd_gt)
    if zero["f1"] != 0.0 or zero["precision"] != 0.0 or zero["recall"] != 0.0:
        raise SystemExit(f"empty FD slice should score 0, got {zero}")

    try:
        discover_fds(None)
    except MinerNotInstalled:
        pass
    else:
        raise SystemExit("discover_fds returned without a miner")

    print(
        "TANE scorer contract OK "
        "(Adult-shaped ceiling F1 0.40; empty FD slice F1 0; not a measurement)"
    )


if __name__ == "__main__":
    _smoke_scorer_contract()
