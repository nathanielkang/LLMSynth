"""Locate vendored third-party baseline repositories."""

import os
import sys

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY = os.path.join(_CODE_ROOT, "third_party")

RELDDPM_ROOT = os.path.join(_THIRD_PARTY, "RelDDPM")
TABDIFF_ROOT = os.path.join(_THIRD_PARTY, "TabDiff")


def ensure_relddpm_path() -> str:
    """Add official RelDDPM repo to ``sys.path`` if present."""
    if os.path.isdir(RELDDPM_ROOT) and RELDDPM_ROOT not in sys.path:
        sys.path.insert(0, RELDDPM_ROOT)
    return RELDDPM_ROOT


def ensure_tabdiff_path() -> str:
    """Add official TabDiff repo to ``sys.path`` if present."""
    if os.path.isdir(TABDIFF_ROOT) and TABDIFF_ROOT not in sys.path:
        sys.path.insert(0, TABDIFF_ROOT)
    return TABDIFF_ROOT


def relddpm_available() -> bool:
    return os.path.isdir(RELDDPM_ROOT)


def tabdiff_repo_available() -> bool:
    return os.path.isdir(TABDIFF_ROOT)
