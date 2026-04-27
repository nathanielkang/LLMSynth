"""
run_experiments.py - Main experiment runner for Paper 3: LLMSynth.

LLMSynth: LLM-Augmented Constraint Discovery for High-Fidelity
          Tabular Data Synthesis.

Orchestrates:
    For each dataset (Adult, German Credit, Heart, Diabetes, Wine):
      1. Load data and ground-truth constraints
      2. Run LLM constraint discovery (or use cached / mock results)
      3. Compute constraint discovery precision / recall
      4. For each method (LLMSynth, TabDDPM, ManualConstraints, PostHocRepair):
        For each seed:
          a. Train synthesizer
          b. Generate synthetic data (same size as training)
          c. Evaluate: violation rate, ML utility, marginal accuracy
          d. Save results
      5. Print summary tables, save LaTeX tables

Usage
-----
    python run_experiments.py                         # full benchmark
    python run_experiments.py --dataset adult          # single dataset
    python run_experiments.py --method LLMSynth TabDDPM  # subset
    python run_experiments.py --seeds 2 --epochs 50   # custom settings
    python run_experiments.py --use-mock               # skip real API calls
    python run_experiments.py --use-cache              # reuse cached LLM responses
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from datasets import get_dataset, get_all_datasets, _DATASET_LOADERS
from constraints import (
    check_constraints,
    compute_violation_rate,
    DistributionHint,
)
from llm_discovery import (
    discover_constraints,
    mock_discover_constraints,
)
from synthesizer import (
    VanillaTabDDPM,
    ConstraintGuidedDiffusion,
    PostHocRepair,
    CTGANSynthesizer,
    get_synthesizer,
)
from metrics import (
    violation_rate,
    ml_utility,
    marginal_distance,
    constraint_discovery_quality,
    evaluate_all,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_experiments")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "LLMSynth",
    "TabDDPM",
    "ManualConstraints",
    "PostHocRepair",
]


def _create_synthesizer(method_name, epochs, verbose):
    """Create a synthesizer instance for the given method."""
    common_kwargs = dict(
        hidden_dim=512,
        n_layers=3,
        n_timesteps=1000,
        epochs=epochs,
        batch_size=4096,
        lr=1e-3,
        verbose=verbose,
    )

    if method_name == "LLMSynth":
        return ConstraintGuidedDiffusion(**common_kwargs)
    elif method_name == "TabDDPM":
        return VanillaTabDDPM(**common_kwargs)
    elif method_name == "ManualConstraints":
        synth = ConstraintGuidedDiffusion(**common_kwargs)
        synth.name = "ManualConstraints"
        return synth
    elif method_name == "PostHocRepair":
        return PostHocRepair(**common_kwargs)
    elif method_name == "CTGAN":
        return CTGANSynthesizer(
            epochs=epochs,
            batch_size=256,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ---------------------------------------------------------------------------
# Constraint discovery step
# ---------------------------------------------------------------------------

def _run_discovery(dataset, use_mock=False, use_cache=True):
    """
    Run LLM constraint discovery for a dataset.

    Returns list of discovered Constraint objects.
    """
    ds_name = dataset["name"]

    if use_mock:
        log.info(f"[Discovery] Using mock constraints for {ds_name}")
        return mock_discover_constraints(ds_name)

    log.info(f"[Discovery] Running LLM discovery for {ds_name} ...")
    try:
        discovered = discover_constraints(
            df=dataset["df_train"],
            cat_columns=dataset["cat_columns"],
            num_columns=dataset["num_columns"],
            model=None,
            use_cache=use_cache,
            validate=True,
            violation_threshold=0.05,
        )
        if not discovered:
            log.warning(f"[Discovery] No constraints discovered for {ds_name} "
                        f"-- falling back to mock")
            discovered = mock_discover_constraints(ds_name)
        return discovered
    except Exception as e:
        log.error(f"[Discovery] Failed for {ds_name}: {e}")
        log.info(f"[Discovery] Falling back to mock constraints")
        return mock_discover_constraints(ds_name)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(dataset, method_name, constraints_to_use,
               seed, epochs, verbose):
    """
    Run one experiment: (dataset, method, seed).

    Parameters
    ----------
    dataset : dict
        Dataset info from datasets.py.
    method_name : str
        Method name.
    constraints_to_use : list
        Constraints for this method (may be empty for baselines).
    seed : int
        Random seed.
    epochs : int
        Training epochs.
    verbose : bool

    Returns
    -------
    dict of results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_name = dataset["name"]
    target_col = dataset["target_col"]
    task = dataset["task"]
    df_train = dataset["df_train"]
    df_test = dataset["df_test"]
    cat_columns = dataset["cat_columns"]
    num_columns = dataset["num_columns"]
    n_samples = len(df_train)

    # Create synthesizer
    t0 = time.time()
    try:
        synth = _create_synthesizer(method_name, epochs, verbose)
    except Exception as e:
        return _error_result(ds_name, method_name, seed, str(e))

    # Check CTGAN availability
    if method_name == "CTGAN" and not synth._available:
        return _error_result(ds_name, method_name, seed,
                             "CTGAN not installed (pip install ctgan)")

    # Fit
    try:
        synth.fit(
            df_train,
            cat_columns=cat_columns,
            num_columns=num_columns,
            target_col=target_col,
        )
    except Exception as e:
        return _error_result(ds_name, method_name, seed,
                             f"fit() failed: {e}")

    fit_time = time.time() - t0

    # Generate
    t1 = time.time()
    try:
        df_synth = synth.generate(n_samples, constraints=constraints_to_use)
    except Exception as e:
        return _error_result(ds_name, method_name, seed,
                             f"generate() failed: {e}")

    gen_time = time.time() - t1

    # Evaluate
    t2 = time.time()
    try:
        ground_truth_constraints = dataset["ground_truth_constraints"]
        eval_results = evaluate_all(
            synthetic_df=df_synth,
            real_train_df=df_train,
            real_test_df=df_test,
            constraints=ground_truth_constraints,
            target_col=target_col,
            task=task,
            cat_columns=cat_columns,
            num_columns=num_columns,
        )
    except Exception as e:
        log.error(f"  Evaluation failed: {e}")
        eval_results = {}

    eval_time = time.time() - t2

    # Build result dict
    result = {
        "dataset": ds_name,
        "method": method_name,
        "seed": seed,
        "n_train": len(df_train),
        "n_synthetic": len(df_synth),
        "fit_sec": round(fit_time, 1),
        "gen_sec": round(gen_time, 1),
        "eval_sec": round(eval_time, 1),
    }
    result.update(eval_results)

    return result


def _error_result(ds_name, method_name, seed, error_msg):
    """Create an error result dict."""
    log.error(f"  ERROR [{method_name}]: {error_msg}")
    return {
        "dataset": ds_name,
        "method": method_name,
        "seed": seed,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiments(args):
    """Execute the full experiment grid."""

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Datasets ---
    # Canonical list avoids duplicates (credit and german_credit are aliases)
    _CANONICAL_DATASETS = ["adult", "credit", "heart", "diabetes", "wine"]
    if "all" in [d.lower() for d in args.dataset]:
        dataset_keys = _CANONICAL_DATASETS
    else:
        dataset_keys = [d.lower().replace(" ", "_").replace("-", "_")
                        for d in args.dataset]

    # --- Methods ---
    if args.method == ["all"]:
        method_keys = ALL_METHODS.copy()
    else:
        method_keys = args.method

    # --- Experiment loop ---
    all_results = []
    discovery_results = {}

    total = len(dataset_keys) * len(method_keys) * args.seeds
    log.info(f"Running {total} experiments "
             f"({len(dataset_keys)} datasets x {len(method_keys)} methods "
             f"x {args.seeds} seeds)")

    exp_count = 0

    for ds_key in dataset_keys:
        # 1. Load dataset
        try:
            dataset = get_dataset(ds_key)
        except Exception as e:
            log.error(f"Failed to load dataset '{ds_key}': {e}")
            continue

        ds_name = dataset["name"]
        gt_constraints = dataset["ground_truth_constraints"]

        # 2. LLM constraint discovery (once per dataset)
        discovered_constraints = _run_discovery(
            dataset,
            use_mock=args.use_mock,
            use_cache=args.use_cache,
        )

        # 3. Compute discovery quality
        disc_quality = constraint_discovery_quality(
            discovered_constraints, gt_constraints
        )
        discovery_results[ds_name] = {
            "n_discovered": disc_quality["n_discovered"],
            "n_ground_truth": disc_quality["n_ground_truth"],
            "n_matched": disc_quality["n_matched"],
            "precision": disc_quality["precision"],
            "recall": disc_quality["recall"],
            "f1": disc_quality["f1"],
        }

        log.info(f"\n{'='*60}")
        log.info(f"  Dataset: {ds_name}")
        log.info(f"  Discovery: P={disc_quality['precision']:.2f} "
                 f"R={disc_quality['recall']:.2f} "
                 f"F1={disc_quality['f1']:.2f} "
                 f"({disc_quality['n_matched']}/{disc_quality['n_ground_truth']} matched)")
        log.info(f"{'='*60}")

        # 4. Run each method
        for method_name in method_keys:
            # Determine which constraints to use for this method
            if method_name == "LLMSynth":
                constraints_to_use = discovered_constraints
            elif method_name == "ManualConstraints":
                constraints_to_use = gt_constraints
            elif method_name == "PostHocRepair":
                constraints_to_use = gt_constraints
            else:
                # TabDDPM, CTGAN: no constraints
                constraints_to_use = []

            for seed in range(args.seeds):
                exp_count += 1
                tag = (f"[{exp_count}/{total}] "
                       f"{ds_name} | {method_name} | seed={seed}")
                log.info(tag)

                try:
                    result = run_single(
                        dataset=dataset,
                        method_name=method_name,
                        constraints_to_use=constraints_to_use,
                        seed=seed,
                        epochs=args.epochs,
                        verbose=args.verbose,
                    )
                    all_results.append(result)

                    # Print key metrics
                    if "error" not in result:
                        vr = result.get("avg_violation_rate", "N/A")
                        tvd = result.get("avg_tvd", "N/A")
                        if isinstance(vr, float):
                            vr = f"{vr:.4f}"
                        if isinstance(tvd, float):
                            tvd = f"{tvd:.4f}"

                        if dataset["task"] == "classification":
                            f1 = result.get("f1_macro", "N/A")
                            if isinstance(f1, float):
                                f1 = f"{f1:.4f}"
                            log.info(f"  -> ViolRate={vr}  F1={f1}  "
                                     f"TVD={tvd}")
                        else:
                            r2 = result.get("r2", "N/A")
                            if isinstance(r2, float):
                                r2 = f"{r2:.4f}"
                            log.info(f"  -> ViolRate={vr}  R2={r2}  "
                                     f"TVD={tvd}")

                except Exception as e:
                    log.error(f"  FAILED: {e}")
                    all_results.append(
                        _error_result(ds_name, method_name, seed, str(e))
                    )

    # --- Save results ---
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "results_raw.csv")
    df_results.to_csv(csv_path, index=False)
    log.info(f"\nRaw results saved to {csv_path}")

    # Save discovery results
    disc_path = os.path.join(output_dir, "discovery_quality.json")
    with open(disc_path, "w", encoding="utf-8") as f:
        json.dump(discovery_results, f, indent=2, ensure_ascii=True)
    log.info(f"Discovery quality saved to {disc_path}")

    # --- Print summary tables ---
    print_summary(df_results, discovery_results, output_dir)

    return df_results


# ---------------------------------------------------------------------------
# Pretty-print summary tables
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, discovery_results: dict,
                  output_dir: str):
    """Print and save summary tables."""
    try:
        from tabulate import tabulate
    except ImportError:
        log.warning("tabulate not installed -- using plain print")
        tabulate = None

    # --- Discovery Quality Table ---
    print(f"\n{'='*70}")
    print("  TABLE 1: Constraint Discovery Quality (LLM vs Ground Truth)")
    print(f"{'='*70}")

    disc_rows = []
    for ds_name, info in discovery_results.items():
        disc_rows.append({
            "Dataset": ds_name,
            "Discovered": info["n_discovered"],
            "GroundTruth": info["n_ground_truth"],
            "Matched": info["n_matched"],
            "Precision": f"{info['precision']:.3f}",
            "Recall": f"{info['recall']:.3f}",
            "F1": f"{info['f1']:.3f}",
        })

    disc_df = pd.DataFrame(disc_rows)
    if tabulate is not None:
        print(tabulate(disc_df, headers="keys", tablefmt="grid",
                       showindex=False))
    else:
        print(disc_df.to_string(index=False))

    # --- Main results table ---
    # Keep rows without errors
    if "error" in df.columns:
        df_ok = df[df["error"].isna()].copy()
    else:
        df_ok = df.copy()

    if df_ok.empty:
        log.warning("No successful results to summarize.")
        return

    # Determine metrics based on available columns
    cls_metrics = ["avg_violation_rate", "f1_macro", "accuracy", "avg_tvd"]
    reg_metrics = ["avg_violation_rate", "r2", "rmse", "avg_tvd"]

    # Use whichever metrics are available
    available_metrics = [m for m in cls_metrics + reg_metrics
                         if m in df_ok.columns]
    # Deduplicate while preserving order
    seen = set()
    unique_metrics = []
    for m in available_metrics:
        if m not in seen:
            seen.add(m)
            unique_metrics.append(m)

    if not unique_metrics:
        log.warning("No metrics columns found in results.")
        return

    # Aggregate: mean +/- std over seeds
    agg = (
        df_ok
        .groupby(["method", "dataset"])
        [unique_metrics]
        .agg(["mean", "std"])
    )

    # Build summary table rows
    datasets = df_ok["dataset"].unique().tolist()
    methods = df_ok["method"].unique().tolist()

    print(f"\n{'='*70}")
    print("  TABLE 2: Synthesis Quality (mean +/- std over seeds)")
    print(f"{'='*70}")

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        rows = []
        for method in methods:
            try:
                row_data = agg.loc[(method, ds_name)]
            except KeyError:
                continue

            entry = {"Method": method}
            for m in unique_metrics:
                try:
                    mu = row_data[(m, "mean")]
                    sd = row_data[(m, "std")]
                    if pd.isna(mu):
                        entry[m] = "N/A"
                    elif pd.isna(sd) or sd == 0:
                        entry[m] = f"{mu:.4f}"
                    else:
                        entry[m] = f"{mu:.4f}+/-{sd:.4f}"
                except (KeyError, TypeError):
                    entry[m] = "N/A"
            rows.append(entry)

        if rows:
            summary_df = pd.DataFrame(rows)
            if tabulate is not None:
                print(tabulate(summary_df, headers="keys",
                               tablefmt="grid", showindex=False))
            else:
                print(summary_df.to_string(index=False))

    # --- Save CSV summaries ---
    summary_path = os.path.join(output_dir, "summary_aggregated.csv")
    try:
        agg_flat = agg.copy()
        agg_flat.columns = [f"{m}_{stat}" for m, stat in agg_flat.columns]
        agg_flat = agg_flat.reset_index()
        agg_flat.to_csv(summary_path, index=False)
        log.info(f"Aggregated summary saved to {summary_path}")
    except Exception as e:
        log.warning(f"Could not save aggregated summary: {e}")

    # --- LaTeX tables ---
    _save_latex_tables(df_ok, discovery_results, output_dir, unique_metrics)


def _save_latex_tables(df, discovery_results, output_dir, metrics):
    """Generate LaTeX tables for the paper."""
    lines = []
    lines.append("% Auto-generated LaTeX tables for Paper 3: LLMSynth")
    lines.append(f"% Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Table 1: Discovery Quality
    lines.append("% ---- Table 1: Constraint Discovery Quality ----")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Constraint discovery quality: LLM-discovered "
                 "vs. ground-truth constraints.}")
    lines.append("\\label{tab:discovery_quality}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & \\#Disc. & \\#GT & \\#Matched & "
                 "Precision & Recall & F1 \\\\")
    lines.append("\\midrule")

    for ds_name, info in discovery_results.items():
        lines.append(
            f"{ds_name} & {info['n_discovered']} & "
            f"{info['n_ground_truth']} & {info['n_matched']} & "
            f"{info['precision']:.3f} & {info['recall']:.3f} & "
            f"{info['f1']:.3f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Main results per dataset
    datasets = df["dataset"].unique().tolist()
    methods = df["method"].unique().tolist()

    agg = (
        df
        .groupby(["method", "dataset"])
        [metrics]
        .agg(["mean", "std"])
    )

    for ds_name in datasets:
        # Determine relevant metrics for this dataset
        ds_metrics = [m for m in metrics if not df[df["dataset"] == ds_name][m].isna().all()]

        if not ds_metrics:
            continue

        n_cols = len(ds_metrics)
        col_spec = "l" + "c" * n_cols

        lines.append(f"% ---- {ds_name} Results ----")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Synthesis results on {ds_name} "
                     f"(mean $\\pm$ std).}}")
        lines.append(f"\\label{{tab:results_{ds_name.lower()}}}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header_parts = ["Method"]
        for m in ds_metrics:
            # Clean metric name for LaTeX
            m_clean = m.replace("_", "\\_")
            header_parts.append(m_clean)
        lines.append(" & ".join(header_parts) + " \\\\")
        lines.append("\\midrule")

        for method in methods:
            try:
                row = agg.loc[(method, ds_name)]
            except KeyError:
                continue

            cells = [method.replace("_", "\\_")]
            for m in ds_metrics:
                try:
                    mu = row[(m, "mean")]
                    sd = row[(m, "std")]
                    if pd.isna(mu):
                        cells.append("--")
                    elif pd.isna(sd) or sd == 0:
                        cells.append(f"{mu:.4f}")
                    else:
                        cells.append(f"{mu:.3f}$\\pm${sd:.3f}")
                except (KeyError, TypeError):
                    cells.append("--")

            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    tex_path = os.path.join(output_dir, "tables_paper3.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"LaTeX tables saved to {tex_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLMSynth Experiment Runner (Paper 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                                        # full benchmark (real API)
  python run_experiments.py --dataset adult credit heart            # subset of datasets
  python run_experiments.py --method LLMSynth TabDDPM              # specific methods
  python run_experiments.py --use-mock                             # skip API calls (mock constraints)
  python run_experiments.py --seeds 3 --epochs 200                 # default settings
  python run_experiments.py --dataset adult credit heart diabetes wine --use-mock
""",
    )
    parser.add_argument(
        "--dataset", nargs="+",
        default=["adult", "credit", "heart", "diabetes", "wine"],
        help=f"Dataset name(s) or 'all'. "
             f"Available: {list(_DATASET_LOADERS.keys())}",
    )
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help=f"Method name(s) or 'all'. Available: {ALL_METHODS}",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds (default: 3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Training epochs for generative models (default: 200).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_paper3",
        help="Directory to save results (default: results_paper3/).",
    )
    parser.add_argument(
        "--use-cache", action="store_true", default=True,
        help="Use cached LLM responses if available (default: True).",
    )
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Disable LLM response caching.",
    )
    parser.add_argument(
        "--use-mock", action="store_true", default=False,
        help="Use mock constraints instead of calling Claude API.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Show per-epoch progress bars for generative models.",
    )

    args = parser.parse_args()

    # Handle cache flag logic
    if args.no_cache:
        args.use_cache = False

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("  LLMSynth - Experiment Runner (Paper 3)")
    log.info("=" * 60)
    log.info(f"  Datasets  : {args.dataset}")
    log.info(f"  Methods   : {args.method}")
    log.info(f"  Seeds     : {args.seeds}")
    log.info(f"  Epochs    : {args.epochs}")
    log.info(f"  Use mock  : {args.use_mock}")
    log.info(f"  Use cache : {args.use_cache}")
    log.info(f"  Output    : {args.output_dir}")
    log.info("=" * 60)

    t_start = time.time()
    df_results = run_experiments(args)
    elapsed = time.time() - t_start

    log.info(f"\nAll experiments completed in {elapsed/60:.1f} minutes.")
    log.info(f"Results directory (relative): {args.output_dir}")
