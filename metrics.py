"""
metrics.py - Evaluation metrics for LLMSynth experiments.

Metrics:
    1. violation_rate:      Per-constraint violation rate on synthetic data
    2. ml_utility:          Train CatBoost on synthetic, test on real (F1/R2)
    3. marginal_distance:   1-way total variation distance (TVD)
    4. constraint_discovery_quality:  Precision/Recall of discovered constraints

All functions take DataFrames and return dicts or floats.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constraints import (
    Constraint,
    ValueRangeConstraint,
    FunctionalDependency,
    CrossColumnRule,
    DistributionHint,
    check_constraints,
    compute_violation_rate,
)


# ---------------------------------------------------------------------------
# 1. Constraint Violation Rate
# ---------------------------------------------------------------------------

def violation_rate(synthetic_df: pd.DataFrame,
                   constraints: list) -> dict:
    """
    Compute per-constraint violation rate on synthetic data.

    Parameters
    ----------
    synthetic_df : pd.DataFrame
        Generated data to evaluate.
    constraints : list of Constraint
        Constraints to check.

    Returns
    -------
    dict with:
        "per_constraint": {name: rate},
        "avg_violation_rate": float,
        "total_violations": int,
    """
    results = check_constraints(synthetic_df, constraints)

    per_constraint = {}
    total_violations = 0
    rates = []

    for name, info in results.items():
        per_constraint[name] = info["rate"]
        total_violations += info["violations"]
        if info["type"] != "distribution_hint":
            rates.append(info["rate"])

    avg_rate = float(np.mean(rates)) if rates else 0.0

    return {
        "per_constraint": per_constraint,
        "avg_violation_rate": avg_rate,
        "total_violations": total_violations,
    }


# ---------------------------------------------------------------------------
# 2. ML Utility (Train on Synthetic, Test on Real)
# ---------------------------------------------------------------------------

def ml_utility(synthetic_df: pd.DataFrame,
               real_test_df: pd.DataFrame,
               target_col: str,
               task: str = "classification",
               cat_columns: list = None) -> dict:
    """
    ML Utility: train CatBoost on synthetic data, evaluate on real test data.

    Parameters
    ----------
    synthetic_df : pd.DataFrame
        Synthetic training data (includes target_col).
    real_test_df : pd.DataFrame
        Real test data (includes target_col).
    target_col : str
        Name of the target column.
    task : str
        "classification" or "regression".
    cat_columns : list, optional
        Categorical feature columns.

    Returns
    -------
    dict with metric values (F1, accuracy for classification; R2, RMSE for regression).
    """
    if cat_columns is None:
        cat_columns = []

    # Prepare features
    feature_cols = [c for c in synthetic_df.columns if c != target_col]

    # Ensure both dataframes have the same columns
    common_cols = [c for c in feature_cols
                   if c in real_test_df.columns and c in synthetic_df.columns]

    X_train = synthetic_df[common_cols].copy()
    y_train = synthetic_df[target_col].copy()
    X_test = real_test_df[common_cols].copy()
    y_test = real_test_df[target_col].copy()

    # Encode categoricals
    label_encoders = {}
    for col in cat_columns:
        if col in common_cols:
            le = LabelEncoder()
            # Fit on combined unique values
            all_vals = pd.concat([
                X_train[col].astype(str),
                X_test[col].astype(str)
            ]).unique()
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le

    # Convert to numeric
    for col in common_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train = X_train.fillna(0).values.astype(np.float32)
    X_test = X_test.fillna(0).values.astype(np.float32)

    if task == "classification":
        return _classification_utility(X_train, y_train, X_test, y_test)
    else:
        return _regression_utility(X_train, y_train, X_test, y_test)


def _classification_utility(X_train, y_train, X_test, y_test):
    """Classification ML utility using CatBoost."""
    from sklearn.metrics import f1_score, accuracy_score

    # Encode target
    le = LabelEncoder()
    all_labels = pd.concat([
        pd.Series(y_train).astype(str),
        pd.Series(y_test).astype(str)
    ]).unique()
    le.fit(all_labels)
    y_train_enc = le.transform(pd.Series(y_train).astype(str))
    y_test_enc = le.transform(pd.Series(y_test).astype(str))

    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42,
            thread_count=-1,
        )
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test).flatten()
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        warnings.warn("catboost not installed, using sklearn GBM",
                       stacklevel=2)
        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42,
        )
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)

    f1 = float(f1_score(y_test_enc, y_pred, average="macro"))
    acc = float(accuracy_score(y_test_enc, y_pred))

    return {
        "f1_macro": f1,
        "accuracy": acc,
    }


def _regression_utility(X_train, y_train, X_test, y_test):
    """Regression ML utility using CatBoost."""
    from sklearn.metrics import r2_score, mean_squared_error

    y_train_num = pd.to_numeric(pd.Series(y_train), errors="coerce").fillna(0).values
    y_test_num = pd.to_numeric(pd.Series(y_test), errors="coerce").fillna(0).values

    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42,
            thread_count=-1,
        )
        model.fit(X_train, y_train_num)
        y_pred = model.predict(X_test).flatten()
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        warnings.warn("catboost not installed, using sklearn GBM",
                       stacklevel=2)
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42,
        )
        model.fit(X_train, y_train_num)
        y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test_num, y_pred))
    rmse_val = float(np.sqrt(mean_squared_error(y_test_num, y_pred)))

    return {
        "r2": r2,
        "rmse": rmse_val,
    }


# ---------------------------------------------------------------------------
# 3. Marginal Distribution Distance (1-way TVD)
# ---------------------------------------------------------------------------

def marginal_distance(synthetic_df: pd.DataFrame,
                      real_df: pd.DataFrame,
                      cat_columns: list = None,
                      num_columns: list = None,
                      n_bins: int = 20) -> dict:
    """
    Compute 1-way Total Variation Distance (TVD) for each column,
    then average across columns.

    For categorical: TVD = 0.5 * sum |p_real(v) - p_synth(v)|
    For numerical: discretize into n_bins, then compute TVD.

    Parameters
    ----------
    synthetic_df, real_df : pd.DataFrame
    cat_columns, num_columns : list
    n_bins : int
        Number of bins for discretizing numerical columns.

    Returns
    -------
    dict with "per_column": {col: tvd}, "avg_tvd": float
    """
    if cat_columns is None:
        cat_columns = synthetic_df.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
    if num_columns is None:
        num_columns = [c for c in synthetic_df.columns
                       if c not in cat_columns]

    common_cols = [c for c in synthetic_df.columns if c in real_df.columns]
    per_column = {}

    for col in common_cols:
        if col in cat_columns:
            tvd = _categorical_tvd(
                real_df[col].astype(str),
                synthetic_df[col].astype(str),
            )
        else:
            tvd = _numerical_tvd(
                pd.to_numeric(real_df[col], errors="coerce").dropna(),
                pd.to_numeric(synthetic_df[col], errors="coerce").dropna(),
                n_bins=n_bins,
            )
        per_column[col] = tvd

    avg_tvd = float(np.mean(list(per_column.values()))) \
        if per_column else 0.0

    return {
        "per_column": per_column,
        "avg_tvd": avg_tvd,
    }


def _categorical_tvd(real_series, synth_series):
    """TVD for categorical columns."""
    all_vals = set(real_series.unique()) | set(synth_series.unique())
    if not all_vals:
        return 0.0

    real_counts = real_series.value_counts(normalize=True)
    synth_counts = synth_series.value_counts(normalize=True)

    tvd = 0.0
    for v in all_vals:
        p_real = real_counts.get(v, 0.0)
        p_synth = synth_counts.get(v, 0.0)
        tvd += abs(p_real - p_synth)

    return float(tvd / 2.0)


def _numerical_tvd(real_series, synth_series, n_bins=20):
    """TVD for numerical columns (discretized into bins)."""
    if len(real_series) == 0 or len(synth_series) == 0:
        return 1.0

    # Determine bin edges from combined range
    all_vals = pd.concat([real_series, synth_series])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    real_hist, _ = np.histogram(real_series, bins=bin_edges, density=False)
    synth_hist, _ = np.histogram(synth_series, bins=bin_edges, density=False)

    # Normalize
    real_hist = real_hist / max(real_hist.sum(), 1)
    synth_hist = synth_hist / max(synth_hist.sum(), 1)

    tvd = 0.5 * np.sum(np.abs(real_hist - synth_hist))
    return float(tvd)


# ---------------------------------------------------------------------------
# 4. Constraint Discovery Quality
# ---------------------------------------------------------------------------

def constraint_discovery_quality(discovered: list,
                                 ground_truth: list) -> dict:
    """
    Evaluate quality of LLM-discovered constraints vs ground truth.

    Matching logic:
        - ValueRangeConstraint: match by column name
        - FunctionalDependency: match by determinant+dependent
        - CrossColumnRule: match by column set overlap
        - DistributionHint: ignored

    Returns
    -------
    dict with precision, recall, f1, n_discovered, n_ground_truth,
         n_matched
    """
    # Filter out DistributionHints for matching
    disc_hard = [c for c in discovered
                 if not isinstance(c, DistributionHint)]
    gt_hard = [c for c in ground_truth
               if not isinstance(c, DistributionHint)]

    if not gt_hard:
        return {
            "precision": 1.0 if not disc_hard else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not disc_hard else 0.0,
            "n_discovered": len(disc_hard),
            "n_ground_truth": 0,
            "n_matched": 0,
        }

    # Build fingerprints for matching
    gt_fingerprints = [_constraint_fingerprint(c) for c in gt_hard]
    disc_fingerprints = [_constraint_fingerprint(c) for c in disc_hard]

    # Count matches
    matched_gt = set()
    matched_disc = set()

    for i, d_fp in enumerate(disc_fingerprints):
        for j, g_fp in enumerate(gt_fingerprints):
            if j not in matched_gt and _fingerprints_match(d_fp, g_fp):
                matched_gt.add(j)
                matched_disc.add(i)
                break

    n_matched = len(matched_gt)
    precision = n_matched / max(len(disc_hard), 1)
    recall = n_matched / max(len(gt_hard), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-10)
           if (precision + recall) > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_discovered": len(disc_hard),
        "n_ground_truth": len(gt_hard),
        "n_matched": n_matched,
    }


def _constraint_fingerprint(c):
    """Create a fingerprint for constraint matching."""
    if isinstance(c, ValueRangeConstraint):
        return ("value_range", frozenset([c.column]))
    elif isinstance(c, FunctionalDependency):
        return ("fd", frozenset(c.determinant), c.dependent)
    elif isinstance(c, CrossColumnRule):
        return ("rule", frozenset(c.columns))
    else:
        return ("other", frozenset(c.columns))


def _fingerprints_match(fp1, fp2):
    """Check if two constraint fingerprints match."""
    # Same type required
    if fp1[0] != fp2[0]:
        return False

    if fp1[0] == "value_range":
        # Match if same column
        return fp1[1] == fp2[1]
    elif fp1[0] == "fd":
        # Match if same determinant and dependent
        return fp1[1] == fp2[1] and fp1[2] == fp2[2]
    elif fp1[0] == "rule":
        # Match if significant overlap in columns
        overlap = len(fp1[1] & fp2[1])
        return overlap >= min(len(fp1[1]), len(fp2[1]))
    else:
        return fp1[1] == fp2[1]


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics at once
# ---------------------------------------------------------------------------

def evaluate_all(synthetic_df: pd.DataFrame,
                 real_train_df: pd.DataFrame,
                 real_test_df: pd.DataFrame,
                 constraints: list,
                 target_col: str,
                 task: str = "classification",
                 cat_columns: list = None,
                 num_columns: list = None) -> dict:
    """
    Evaluate all metrics for a synthetic dataset.

    Returns
    -------
    dict with all metric results.
    """
    results = {}

    # 1. Violation rate
    vr = violation_rate(synthetic_df, constraints)
    results["avg_violation_rate"] = vr["avg_violation_rate"]
    results["total_violations"] = vr["total_violations"]

    # 2. ML Utility
    try:
        ml = ml_utility(
            synthetic_df, real_test_df,
            target_col=target_col,
            task=task,
            cat_columns=cat_columns,
        )
        results.update(ml)
    except Exception as e:
        print(f"  [WARN] ML utility failed: {e}")
        if task == "classification":
            results["f1_macro"] = 0.0
            results["accuracy"] = 0.0
        else:
            results["r2"] = 0.0
            results["rmse"] = float("inf")

    # 3. Marginal distance
    try:
        md = marginal_distance(
            synthetic_df, real_train_df,
            cat_columns=cat_columns,
            num_columns=num_columns,
        )
        results["avg_tvd"] = md["avg_tvd"]
    except Exception as e:
        print(f"  [WARN] Marginal distance failed: {e}")
        results["avg_tvd"] = 1.0

    return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from constraints import ValueRangeConstraint, FunctionalDependency

    # Create dummy data
    np.random.seed(42)
    real_df = pd.DataFrame({
        "age": np.random.randint(18, 65, size=200),
        "income": np.random.normal(50000, 15000, size=200).clip(0),
        "edu": np.random.choice(["HS", "BS", "MS"], size=200),
        "target": np.random.choice(["A", "B"], size=200),
    })

    synth_df = pd.DataFrame({
        "age": np.random.randint(10, 80, size=200),
        "income": np.random.normal(45000, 20000, size=200),
        "edu": np.random.choice(["HS", "BS", "MS", "PhD"], size=200),
        "target": np.random.choice(["A", "B"], size=200),
    })

    constraints = [
        ValueRangeConstraint("age_range", "age", min_val=18, max_val=65),
        ValueRangeConstraint("income_pos", "income", min_val=0),
    ]

    # Test violation rate
    vr = violation_rate(synth_df, constraints)
    print(f"Violation rate: {vr['avg_violation_rate']:.4f}")

    # Test marginal distance
    md = marginal_distance(synth_df, real_df,
                           cat_columns=["edu", "target"],
                           num_columns=["age", "income"])
    print(f"Avg TVD: {md['avg_tvd']:.4f}")

    # Test ML utility
    ml = ml_utility(synth_df, real_df, target_col="target",
                    task="classification", cat_columns=["edu"])
    print(f"F1: {ml['f1_macro']:.4f}, Acc: {ml['accuracy']:.4f}")

    # Test discovery quality
    discovered = [
        ValueRangeConstraint("d_age", "age", min_val=18, max_val=65),
    ]
    ground_truth = [
        ValueRangeConstraint("gt_age", "age", min_val=18, max_val=65),
        ValueRangeConstraint("gt_income", "income", min_val=0),
    ]
    dq = constraint_discovery_quality(discovered, ground_truth)
    print(f"Discovery P={dq['precision']:.2f} R={dq['recall']:.2f} "
          f"F1={dq['f1']:.2f}")
