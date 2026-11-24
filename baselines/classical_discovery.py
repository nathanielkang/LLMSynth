"""Shared scorer hook for classical constraint miners.

Table 4 numbers are not produced here. ``score_discovered`` only evaluates
constraint lists the caller already has, through
``metrics.constraint_discovery_quality`` (full ground-truth denominator,
type-matched fingerprints). A later TANE or Hydra run should return real
``FunctionalDependency`` / ``CrossColumnRule`` objects and pass them here.

This module does not write a CSV and does not embed best-guess cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))


class MinerNotInstalled(RuntimeError):
    """Raised when the miner body has not been wired to a real implementation."""


def score_discovered(discovered: list, ground_truth: list, *, miner: str) -> dict:
    """Score ``discovered`` against ``ground_truth``.

    The returned dict is the live scorer's output for those two lists.
    ``table4_measurement`` stays false: calling the scorer on hand-built
    objects is a contract check, not a dataset run.
    """
    from metrics import constraint_discovery_quality

    out = dict(constraint_discovery_quality(discovered, ground_truth))
    out["miner"] = miner
    out["paper_table"] = False
    out["table4_measurement"] = False
    return out
