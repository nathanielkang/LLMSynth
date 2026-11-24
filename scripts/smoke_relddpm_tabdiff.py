#!/usr/bin/env python3
"""
Smoke test for RelDDPM + TabDiff adapters.

Runs a tiny Adult subsample (500 rows), 2 epochs, 1 seed, writes one JSON row
per method under results_smoke/.  Exit code 0 on success.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Ensure 2_Code root is on path when invoked from scripts/
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from datasets import get_dataset
from metrics import evaluate_all
from baselines.relddpm_adapter import RelDDPMSynthesizer
from baselines.tabdiff_adapter import TabDiffSynthesizer


def _subsample(dataset, n=500):
    df = dataset["df_train"]
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    out = dict(dataset)
    out["df_train"] = df
    return out


def _run_one(dataset, method_name, synth, constraints, seed=0, epochs=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = dataset
    fit_kw = dict(
        cat_columns=ds["cat_columns"],
        num_columns=ds["num_columns"],
        target_col=ds["target_col"],
    )
    if method_name == "RelDDPM":
        fit_kw["constraints"] = constraints
    synth.fit(ds["df_train"], **fit_kw)
    n = len(ds["df_train"])
    df_synth = synth.generate(n, constraints=constraints if method_name == "RelDDPM" else None)
    metrics = evaluate_all(
        synthetic_df=df_synth,
        real_train_df=ds["df_train"],
        real_test_df=ds["df_test"],
        constraints=ds["ground_truth_constraints"],
        target_col=ds["target_col"],
        task=ds["task"],
        cat_columns=ds["cat_columns"],
        num_columns=ds["num_columns"],
    )
    row = {
        "dataset": ds["name"],
        "method": method_name,
        "seed": seed,
        "epochs": epochs,
        "n_train": n,
        "n_synthetic": len(df_synth),
    }
    row.update(metrics)
    return row


def main():
    out_dir = os.path.join(_CODE_ROOT, "results_smoke")
    os.makedirs(out_dir, exist_ok=True)

    dataset = _subsample(get_dataset("adult"), n=500)
    gt = dataset["ground_truth_constraints"]
    epochs = 2

    rows = []
    for name, cls in [("RelDDPM", RelDDPMSynthesizer), ("TabDiff", TabDiffSynthesizer)]:
        print(f"\n=== Smoke: {name} ===")
        synth = cls(epochs=epochs, batch_size=256, verbose=True)
        row = _run_one(dataset, name, synth, gt, seed=0, epochs=epochs)
        rows.append(row)
        print(f"  avg_violation_rate={row.get('avg_violation_rate')}  "
              f"avg_tvd={row.get('avg_tvd')}  f1={row.get('f1_macro')}")

    json_path = os.path.join(out_dir, "smoke_relddpm_tabdiff.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {json_path}")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
