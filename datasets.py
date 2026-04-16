"""
datasets.py - Load and preprocess benchmark datasets with ground-truth constraints.

Datasets:
    1. Adult (income classification)
    2. German Credit (credit-g)
    3. Insurance (medical charges regression)
    4. Heart Disease (heart-statlog classification)
    5. Diabetes (regression, sklearn built-in)
    6. Wine Quality Red (classification, binarized quality)

Each loader returns a dict:
    {
        "name": str,
        "task": "classification" or "regression",
        "target_col": str,
        "df_train": pd.DataFrame,   # includes target column
        "df_test": pd.DataFrame,    # includes target column
        "columns": list[str],
        "cat_columns": list[str],
        "num_columns": list[str],
        "ground_truth_constraints": list[Constraint],
    }

All datasets auto-download via sklearn/openml.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from constraints import (
    ValueRangeConstraint,
    FunctionalDependency,
    CrossColumnRule,
    DistributionHint,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(columns, candidates):
    """Find the first matching column name from a list of candidates."""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


# ---------------------------------------------------------------------------
# Adult Income Dataset
# ---------------------------------------------------------------------------

def load_adult():
    """
    Load the Adult (Census Income) dataset.

    Task: binary classification (income >50K or <=50K)
    ~48,842 samples, 14 features.
    """
    print("[datasets] Loading Adult dataset from OpenML ...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.data.copy()
    df["income"] = data.target.copy()

    # Standardize column names (remove hyphens for easier handling)
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    # Drop rows with missing values for cleaner constraint checking
    df = df.dropna().reset_index(drop=True)

    # Identify column types
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Convert categorical columns to string type
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()

    # Convert numeric columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Train/test split
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["income"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Ground-truth constraints
    constraints = _adult_constraints()

    print(f"  -> Adult: train={len(df_train)}, test={len(df_test)}, "
          f"cols={len(df.columns)}, constraints={len(constraints)}")

    return {
        "name": "Adult",
        "task": "classification",
        "target_col": "income",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": [c for c in cat_cols if c in df.columns],
        "num_columns": [c for c in num_cols if c in df.columns],
        "ground_truth_constraints": constraints,
    }


def _adult_constraints():
    """Define ground-truth constraints for the Adult dataset."""
    constraints = [
        # 1. age >= 17 (working age)
        ValueRangeConstraint(
            name="age_min_17",
            column="age",
            min_val=17,
        ),
        # 2. hours_per_week in [1, 99]
        ValueRangeConstraint(
            name="hours_1_to_99",
            column="hours_per_week",
            min_val=1,
            max_val=99,
        ),
        # 3. capital_gain >= 0
        ValueRangeConstraint(
            name="capital_gain_nonneg",
            column="capital_gain",
            min_val=0,
        ),
        # 4. capital_loss >= 0
        ValueRangeConstraint(
            name="capital_loss_nonneg",
            column="capital_loss",
            min_val=0,
        ),
        # 5. education -> education_num (functional dependency)
        FunctionalDependency(
            name="education_to_edu_num",
            determinant=["education"],
            dependent="education_num",
        ),
        # 6. relationship=Husband => sex=Male
        CrossColumnRule(
            name="husband_implies_male",
            columns=["relationship", "sex"],
            predicate=lambda df: ~(
                (df["relationship"].astype(str).str.strip() == "Husband") &
                (df["sex"].astype(str).str.strip() != "Male")
            ),
            description="If relationship=Husband then sex=Male",
        ),
        # 7. fnlwgt > 0 (census weighting factor)
        ValueRangeConstraint(
            name="fnlwgt_positive",
            column="fnlwgt",
            min_val=1,
        ),
        # 8. education_num in [1, 16]
        ValueRangeConstraint(
            name="edu_num_range",
            column="education_num",
            min_val=1,
            max_val=16,
        ),
        # Distribution hints (soft)
        DistributionHint(
            name="age_distribution",
            column="age",
            distribution="right_skewed",
        ),
    ]
    return constraints


# ---------------------------------------------------------------------------
# German Credit Dataset
# ---------------------------------------------------------------------------

def load_german_credit():
    """
    Load the German Credit (credit-g) dataset.

    Task: binary classification (good/bad credit risk)
    1000 samples, 20 features.
    """
    print("[datasets] Loading German Credit dataset from OpenML ...")

    # Try multiple names for robustness
    data = None
    for _name in ["credit-g", "german-credit"]:
        try:
            data = fetch_openml(_name, version=1, as_frame=True,
                                parser="auto")
            break
        except Exception:
            continue

    if data is None:
        try:
            data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        except Exception as e:
            raise RuntimeError(
                f"Could not load German Credit dataset: {e}. "
                f"Try: pip install scikit-learn --upgrade"
            )

    df = data.data.copy()
    df["credit_risk"] = data.target.copy()

    # Standardize column names
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["credit_risk"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    constraints = _german_credit_constraints()

    print(f"  -> German Credit: train={len(df_train)}, test={len(df_test)}, "
          f"cols={len(df.columns)}, constraints={len(constraints)}")

    return {
        "name": "GermanCredit",
        "task": "classification",
        "target_col": "credit_risk",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": [c for c in cat_cols if c in df.columns],
        "num_columns": [c for c in num_cols if c in df.columns],
        "ground_truth_constraints": constraints,
    }


def _german_credit_constraints():
    """Define ground-truth constraints for German Credit."""
    constraints = [
        # 1. duration > 0
        ValueRangeConstraint(
            name="duration_positive",
            column="duration",
            min_val=1,
        ),
        # 2. credit_amount > 0
        ValueRangeConstraint(
            name="credit_amount_positive",
            column="credit_amount",
            min_val=1,
        ),
        # 3. age >= 18
        ValueRangeConstraint(
            name="age_min_18",
            column="age",
            min_val=18,
        ),
        # 4. installment_commitment in [1, 4]
        ValueRangeConstraint(
            name="installment_1_to_4",
            column="installment_commitment",
            min_val=1,
            max_val=4,
        ),
        # 5. residence_since in [1, 4]
        ValueRangeConstraint(
            name="residence_1_to_4",
            column="residence_since",
            min_val=1,
            max_val=4,
        ),
        # 6. num_dependents in [1, 2]
        ValueRangeConstraint(
            name="num_dependents_range",
            column="num_dependents",
            min_val=1,
            max_val=2,
        ),
        # 7. existing_credits in [1, 4]
        ValueRangeConstraint(
            name="existing_credits_range",
            column="existing_credits",
            min_val=1,
            max_val=4,
        ),
        # Distribution hint
        DistributionHint(
            name="credit_amount_dist",
            column="credit_amount",
            distribution="right_skewed",
        ),
    ]
    return constraints


# ---------------------------------------------------------------------------
# Insurance Dataset
# ---------------------------------------------------------------------------

def load_insurance():
    """
    Load the Insurance dataset (medical charges).

    Task: regression (predict charges)
    1338 samples, 6 features.
    """
    print("[datasets] Loading Insurance dataset from OpenML ...")
    data = fetch_openml("insurance", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    df["charges"] = data.target.copy()

    # Standardize column names
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    constraints = _insurance_constraints()

    print(f"  -> Insurance: train={len(df_train)}, test={len(df_test)}, "
          f"cols={len(df.columns)}, constraints={len(constraints)}")

    return {
        "name": "Insurance",
        "task": "regression",
        "target_col": "charges",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": [c for c in cat_cols if c in df.columns],
        "num_columns": [c for c in num_cols if c in df.columns],
        "ground_truth_constraints": constraints,
    }


def _insurance_constraints():
    """Define ground-truth constraints for Insurance."""
    constraints = [
        # 1. age in [18, 64]
        ValueRangeConstraint(
            name="age_18_to_64",
            column="age",
            min_val=18,
            max_val=64,
        ),
        # 2. bmi > 0
        ValueRangeConstraint(
            name="bmi_positive",
            column="bmi",
            min_val=0.1,
        ),
        # 3. children >= 0
        ValueRangeConstraint(
            name="children_nonneg",
            column="children",
            min_val=0,
        ),
        # 4. charges > 0
        ValueRangeConstraint(
            name="charges_positive",
            column="charges",
            min_val=0.01,
        ),
        # 5. smoker=yes => charges tend to be higher (hard threshold)
        CrossColumnRule(
            name="smoker_higher_charges",
            columns=["smoker", "charges"],
            predicate=lambda df: ~(
                (df["smoker"].astype(str).str.strip().str.lower() == "yes") &
                (pd.to_numeric(df["charges"], errors="coerce") < 1000)
            ),
            description="If smoker=yes then charges should be >= 1000",
        ),
        # Distribution hints
        DistributionHint(
            name="charges_distribution",
            column="charges",
            distribution="right_skewed",
        ),
        DistributionHint(
            name="bmi_distribution",
            column="bmi",
            distribution="roughly_normal",
        ),
    ]
    return constraints


# ---------------------------------------------------------------------------
# Heart Disease Dataset
# ---------------------------------------------------------------------------

def load_heart():
    """
    Load the Heart Disease (heart-statlog) dataset.

    Task: binary classification (presence/absence of heart disease)
    ~270 samples, 13 features.
    Tight range constraints that TabDDPM is likely to violate.
    """
    print("[datasets] Loading Heart Disease dataset from OpenML ...")

    data = None
    for _name in ["heart-statlog", "heart-disease"]:
        try:
            data = fetch_openml(_name, version=1, as_frame=True,
                                parser="auto")
            break
        except Exception:
            continue

    if data is None:
        try:
            data = fetch_openml(data_id=53, as_frame=True, parser="auto")
        except Exception as e:
            raise RuntimeError(
                f"Could not load Heart Disease dataset: {e}"
            )

    df = data.data.copy()
    target = data.target.copy()

    # Standardize column names
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    # Binarize target: 'present'/2 -> 1, 'absent'/1/0 -> 0
    target_str = target.astype(str).str.strip().str.lower()
    unique_targets = set(target_str.unique())

    if "present" in unique_targets:
        df["target"] = (target_str == "present").astype(int)
    elif "2" in unique_targets:
        df["target"] = (target_str == "2").astype(int)
    else:
        # Assume highest numeric value is positive class
        target_num = pd.to_numeric(target, errors="coerce")
        df["target"] = (target_num >= target_num.median()).astype(int)

    # Identify column types
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Train/test split
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["target"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    constraints = _heart_constraints(df)

    print(f"  -> Heart Disease: train={len(df_train)}, "
          f"test={len(df_test)}, cols={len(df.columns)}, "
          f"constraints={len(constraints)}")

    return {
        "name": "Heart",
        "task": "classification",
        "target_col": "target",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": [c for c in cat_cols if c in df.columns],
        "num_columns": [c for c in num_cols if c in df.columns],
        "ground_truth_constraints": constraints,
    }


def _heart_constraints(df):
    """
    Define ground-truth constraints for Heart Disease.

    Uses _find_col to resolve column names across OpenML versions.
    Tight ranges that TabDDPM (small dataset, 270 samples) is likely
    to violate through extrapolation.
    """
    cols = list(df.columns)
    constraints = []

    # 1. age in [28, 80] -- actual data range ~29-77
    if "age" in cols:
        constraints.append(ValueRangeConstraint(
            name="age_range",
            column="age",
            min_val=28,
            max_val=80,
        ))

    # 2. resting blood pressure in [90, 200]
    bp_col = _find_col(cols, [
        "resting_blood_pressure", "rest_bpress", "trestbps",
    ])
    if bp_col:
        constraints.append(ValueRangeConstraint(
            name="rest_bp_range",
            column=bp_col,
            min_val=90,
            max_val=200,
        ))

    # 3. serum cholesterol in [100, 600]
    chol_col = _find_col(cols, [
        "serum_cholestoral", "serum_cholesterol", "chol",
        "cholesterol",
    ])
    if chol_col:
        constraints.append(ValueRangeConstraint(
            name="chol_range",
            column=chol_col,
            min_val=100,
            max_val=600,
        ))

    # 4. max heart rate in [60, 210] -- tight, TabDDPM likely violates
    hr_col = _find_col(cols, [
        "max_heart_rate", "maximum_heart_rate_achieved",
        "max_hr", "thalach",
    ])
    if hr_col:
        constraints.append(ValueRangeConstraint(
            name="max_hr_range",
            column=hr_col,
            min_val=60,
            max_val=210,
        ))

    # 5. oldpeak (ST depression) in [0, 7] -- must be non-negative
    op_col = _find_col(cols, [
        "oldpeak", "ST_depression", "st_depression",
    ])
    if op_col:
        constraints.append(ValueRangeConstraint(
            name="oldpeak_nonneg",
            column=op_col,
            min_val=0.0,
            max_val=7.0,
        ))

    # 6. number of major vessels in [0, 3] -- discrete, very tight
    v_col = _find_col(cols, [
        "number_of_major_vessels", "vessels", "num_vessels",
        "ca", "major_vessels",
    ])
    if v_col:
        constraints.append(ValueRangeConstraint(
            name="vessels_range",
            column=v_col,
            min_val=0,
            max_val=3,
        ))

    # 7. Cross-column: if age >= 60, max_hr should be <= 200
    #    (elderly patients rarely achieve very high heart rates in tests)
    if "age" in cols and hr_col:
        _hr = hr_col  # capture for closure
        constraints.append(CrossColumnRule(
            name="elderly_max_hr",
            columns=["age", _hr],
            predicate=lambda df, hc=_hr: ~(
                (pd.to_numeric(df["age"], errors="coerce") >= 60) &
                (pd.to_numeric(df[hc], errors="coerce") > 200)
            ),
            description="If age >= 60 then max heart rate <= 200",
        ))

    # Distribution hint
    if "age" in cols:
        constraints.append(DistributionHint(
            name="heart_age_dist",
            column="age",
            distribution="roughly_normal",
        ))

    if len(constraints) < 3:
        print(f"  [WARN] Only {len(constraints)} heart constraints matched "
              f"columns. Available columns: {cols}")

    return constraints


# ---------------------------------------------------------------------------
# Diabetes Dataset (sklearn built-in)
# ---------------------------------------------------------------------------

def load_diabetes():
    """
    Load the Diabetes dataset from sklearn.

    Task: regression (predict disease progression score)
    442 samples, 10 standardized features.
    Very tight feature ranges -> TabDDPM likely extrapolates.
    """
    print("[datasets] Loading Diabetes dataset from sklearn ...")
    from sklearn.datasets import load_diabetes as _load_diabetes

    data = _load_diabetes(as_frame=True)
    df = data.frame.copy()

    # Standardize column names
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    # All features are numeric in this dataset
    cat_cols = []
    num_cols = list(df.columns)

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Compute constraints from full data (tight bounds)
    constraints = _diabetes_constraints(df)

    print(f"  -> Diabetes: train={len(df_train)}, test={len(df_test)}, "
          f"cols={len(df.columns)}, constraints={len(constraints)}")

    return {
        "name": "Diabetes",
        "task": "regression",
        "target_col": "target",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": cat_cols,
        "num_columns": num_cols,
        "ground_truth_constraints": constraints,
    }


def _diabetes_constraints(df):
    """
    Define tight ground-truth constraints for the Diabetes dataset.

    Features are standardized (mean~0, very narrow ranges).
    With only 442 samples, TabDDPM will almost certainly extrapolate
    beyond these tight bounds.
    """
    constraints = []

    # 1. Target range (disease progression score)
    target_min = float(df["target"].min())
    target_max = float(df["target"].max())
    constraints.append(ValueRangeConstraint(
        name="target_range",
        column="target",
        min_val=target_min - 1,
        max_val=target_max + 1,
    ))

    # 2-7. Tight bounds on key standardized features
    # Use 1% margin beyond observed range -- ensures 0 training violations
    # but catches TabDDPM extrapolation
    key_features = ["bmi", "bp", "s1", "s2", "s5", "s6"]
    for col in key_features:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            data_range = float(vals.max()) - float(vals.min())
            margin = data_range * 0.01
            constraints.append(ValueRangeConstraint(
                name=f"{col}_tight_range",
                column=col,
                min_val=round(float(vals.min()) - margin, 5),
                max_val=round(float(vals.max()) + margin, 5),
            ))

    # Distribution hint
    constraints.append(DistributionHint(
        name="target_dist",
        column="target",
        distribution="right_skewed",
    ))

    return constraints


# ---------------------------------------------------------------------------
# Wine Quality (Red) Dataset
# ---------------------------------------------------------------------------

def load_wine():
    """
    Load the Wine Quality (Red) dataset.

    Task: binary classification (quality >= 6 -> good, else bad)
    ~1599 samples, 11 physicochemical features.
    Tight pH/acidity/alcohol ranges that TabDDPM may violate.
    """
    print("[datasets] Loading Wine Quality (Red) dataset from OpenML ...")

    data = None
    for _name in ["wine-quality-red", "wine_quality_red"]:
        try:
            data = fetch_openml(_name, version=1, as_frame=True,
                                parser="auto")
            break
        except Exception:
            continue

    if data is None:
        try:
            # Fallback: try by dataset ID
            data = fetch_openml(data_id=40691, as_frame=True,
                                parser="auto")
        except Exception as e:
            raise RuntimeError(
                f"Could not load Wine Quality dataset: {e}"
            )

    df = data.data.copy()
    target = data.target.copy()

    # Standardize column names (handle dots, hyphens, spaces)
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  .replace(".", "_")
                  for c in df.columns]

    # Binarize quality: >= 6 is "good" (1), else "bad" (0)
    quality_vals = pd.to_numeric(target, errors="coerce")
    df["quality_label"] = (quality_vals >= 6).astype(int)

    # Identify column types
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Ensure target is in the dataframe for stratification
    df["quality_label"] = df["quality_label"].astype(int)

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["quality_label"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    constraints = _wine_constraints(df)

    print(f"  -> Wine Quality: train={len(df_train)}, "
          f"test={len(df_test)}, cols={len(df.columns)}, "
          f"constraints={len(constraints)}")

    return {
        "name": "WineQuality",
        "task": "classification",
        "target_col": "quality_label",
        "df_train": df_train,
        "df_test": df_test,
        "columns": list(df.columns),
        "cat_columns": [c for c in cat_cols if c in df.columns],
        "num_columns": [c for c in num_cols if c in df.columns],
        "ground_truth_constraints": constraints,
    }


def _wine_constraints(df):
    """
    Define ground-truth constraints for Wine Quality.

    Tight physicochemical ranges and a cross-column logical rule
    (free SO2 <= total SO2) that TabDDPM is likely to violate.
    """
    cols = list(df.columns)
    constraints = []

    # Resolve column names (handle underscore vs dot variations)
    fa_col = _find_col(cols, ["fixed_acidity", "fixed acidity"])
    va_col = _find_col(cols, ["volatile_acidity", "volatile acidity"])
    ca_col = _find_col(cols, ["citric_acid", "citric acid"])
    ph_col = _find_col(cols, ["pH", "ph"])
    alc_col = _find_col(cols, ["alcohol"])
    sul_col = _find_col(cols, ["sulphates"])
    rs_col = _find_col(cols, ["residual_sugar", "residual sugar"])
    fso2_col = _find_col(cols, [
        "free_sulfur_dioxide", "free sulfur dioxide",
    ])
    tso2_col = _find_col(cols, [
        "total_sulfur_dioxide", "total sulfur dioxide",
    ])

    # 1. fixed_acidity in [3.5, 16.5]
    if fa_col:
        constraints.append(ValueRangeConstraint(
            name="fixed_acidity_range",
            column=fa_col,
            min_val=3.5,
            max_val=16.5,
        ))

    # 2. volatile_acidity in [0.08, 1.70]
    if va_col:
        constraints.append(ValueRangeConstraint(
            name="volatile_acidity_range",
            column=va_col,
            min_val=0.08,
            max_val=1.70,
        ))

    # 3. citric_acid in [0.0, 1.1] -- cannot be negative
    if ca_col:
        constraints.append(ValueRangeConstraint(
            name="citric_acid_range",
            column=ca_col,
            min_val=0.0,
            max_val=1.1,
        ))

    # 4. pH in [2.7, 4.2] -- very tight wine pH range
    if ph_col:
        constraints.append(ValueRangeConstraint(
            name="pH_range",
            column=ph_col,
            min_val=2.7,
            max_val=4.2,
        ))

    # 5. alcohol in [8.0, 15.5]
    if alc_col:
        constraints.append(ValueRangeConstraint(
            name="alcohol_range",
            column=alc_col,
            min_val=8.0,
            max_val=15.5,
        ))

    # 6. sulphates in [0.2, 2.2]
    if sul_col:
        constraints.append(ValueRangeConstraint(
            name="sulphates_range",
            column=sul_col,
            min_val=0.2,
            max_val=2.2,
        ))

    # 7. residual_sugar > 0
    if rs_col:
        constraints.append(ValueRangeConstraint(
            name="residual_sugar_positive",
            column=rs_col,
            min_val=0.5,
        ))

    # 8. Cross-column: free_sulfur_dioxide <= total_sulfur_dioxide
    #    This is a chemical necessity that diffusion models may violate.
    if fso2_col and tso2_col:
        _f = fso2_col
        _t = tso2_col
        constraints.append(CrossColumnRule(
            name="free_leq_total_so2",
            columns=[_f, _t],
            predicate=lambda df, fc=_f, tc=_t: (
                pd.to_numeric(df[fc], errors="coerce") <=
                pd.to_numeric(df[tc], errors="coerce")
            ),
            description="Free SO2 must be <= Total SO2",
        ))

    # Distribution hint
    if alc_col:
        constraints.append(DistributionHint(
            name="alcohol_dist",
            column=alc_col,
            distribution="right_skewed",
        ))

    if len(constraints) < 3:
        print(f"  [WARN] Only {len(constraints)} wine constraints matched "
              f"columns. Available columns: {cols}")

    return constraints


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "adult": load_adult,
    "german_credit": load_german_credit,
    "credit": load_german_credit,
    "insurance": load_insurance,
    "heart": load_heart,
    "heart_disease": load_heart,
    "diabetes": load_diabetes,
    "wine": load_wine,
    "wine_quality": load_wine,
}

# Canonical list (no aliases) for iterating all unique datasets
_CANONICAL_DATASETS = ["adult", "credit", "heart", "diabetes", "wine"]


def get_dataset(name: str) -> dict:
    """
    Load a single dataset by name.

    Parameters
    ----------
    name : str
        One of: adult, credit, german_credit, insurance,
        heart, heart_disease, diabetes, wine, wine_quality

    Returns
    -------
    dict with keys: name, task, target_col, df_train, df_test,
         columns, cat_columns, num_columns, ground_truth_constraints
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(_DATASET_LOADERS.keys())}"
        )
    return _DATASET_LOADERS[key]()


def get_all_datasets() -> list:
    """Load and return all benchmark datasets (no duplicates)."""
    datasets = []
    for key in _CANONICAL_DATASETS:
        try:
            ds = get_dataset(key)
            datasets.append(ds)
        except Exception as e:
            print(f"  [WARN] Failed to load {key}: {e}")
    return datasets


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name in _CANONICAL_DATASETS:
        try:
            ds = get_dataset(name)
        except Exception as e:
            print(f"\n[ERROR] Could not load {name}: {e}")
            continue

        print(f"\n{ds['name']}:")
        print(f"  Task: {ds['task']}, Target: {ds['target_col']}")
        print(f"  Train: {len(ds['df_train'])}, Test: {len(ds['df_test'])}")
        print(f"  Columns: {ds['columns'][:5]} ...")
        print(f"  Cat: {ds['cat_columns'][:3]}, Num: {ds['num_columns'][:3]}")
        print(f"  Ground-truth constraints ({len(ds['ground_truth_constraints'])}):")
        for c in ds["ground_truth_constraints"]:
            print(f"    {c}")

        # Check constraints on training data
        from constraints import check_constraints
        results = check_constraints(ds["df_train"],
                                    ds["ground_truth_constraints"])
        for cname, info in results.items():
            print(f"    {cname}: {info['violations']} violations "
                  f"(rate={info['rate']:.4f})")
