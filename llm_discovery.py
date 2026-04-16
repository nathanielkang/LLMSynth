"""
llm_discovery.py - LLM-augmented constraint discovery using Anthropic Claude.

Core function:
    discover_constraints(schema_info, sample_df, model)
        -> list[Constraint]

Also provides:
    mock_discover_constraints(dataset_name) for testing without API.

Uses python-dotenv to load the ANTHROPIC_API_KEY from .env file.
Includes rate limiting, error handling, and response caching.
"""

import json
import os
import re
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from constraints import (
    ValueRangeConstraint,
    FunctionalDependency,
    CrossColumnRule,
    DistributionHint,
)

# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------

def _load_api_key():
    """Load Anthropic API key from .env file using python-dotenv."""
    try:
        from dotenv import load_dotenv

        # Search multiple .env locations
        env_candidates = [
            # Project root (3 levels up from this file)
            Path(__file__).resolve().parent.parent.parent / ".env",
            # 2 levels up
            Path(__file__).resolve().parent.parent / ".env",
            # Same directory
            Path(__file__).resolve().parent / ".env",
        ]

        loaded = False
        for env_path in env_candidates:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"[LLM Discovery] Loaded .env from {env_path}")
                loaded = True
                break

        if not loaded:
            # Try default dotenv search
            load_dotenv()

    except ImportError:
        warnings.warn(
            "python-dotenv not installed. "
            "Set ANTHROPIC_API_KEY environment variable manually.",
            stacklevel=2,
        )

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        warnings.warn(
            "ANTHROPIC_API_KEY not found in environment or .env file. "
            "Checked paths: project root .env and parent directories.",
            stacklevel=2,
        )
    return key


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _get_cache_path(schema_hash: str) -> Path:
    """Return path to cached LLM response."""
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"llm_response_{schema_hash}.json"


def _compute_cache_key(schema_info: str, sample_text: str) -> str:
    """Compute a hash key for caching LLM responses."""
    content = schema_info + sample_text
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def _load_from_cache(cache_key: str):
    """Load cached LLM response if available."""
    path = _get_cache_path(cache_key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_to_cache(cache_key: str, response_data: dict):
    """Save LLM response to cache."""
    path = _get_cache_path(cache_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

def _build_schema_info(df: pd.DataFrame,
                       cat_columns: list = None,
                       num_columns: list = None) -> str:
    """
    Build a textual schema description from a DataFrame.
    """
    if cat_columns is None:
        cat_columns = df.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
    if num_columns is None:
        num_columns = [c for c in df.columns if c not in cat_columns]

    lines = []
    for col in df.columns:
        if col in cat_columns:
            unique_vals = df[col].astype(str).unique()
            if len(unique_vals) > 10:
                vals_str = ", ".join(sorted(unique_vals)[:10]) + ", ..."
            else:
                vals_str = ", ".join(sorted(unique_vals))
            lines.append(
                f"- {col}: categorical [{vals_str}] "
                f"({len(unique_vals)} unique values)"
            )
        else:
            col_data = pd.to_numeric(df[col], errors="coerce")
            col_min = col_data.min()
            col_max = col_data.max()
            col_mean = col_data.mean()
            lines.append(
                f"- {col}: numeric (min={col_min:.2f}, max={col_max:.2f}, "
                f"mean={col_mean:.2f})"
            )

    return "\n".join(lines)


def _build_sample_table(df: pd.DataFrame, n_rows: int = 12) -> str:
    """Format a sample of rows as a text table."""
    sample = df.head(n_rows)
    return sample.to_string(index=False)


def _build_prompt(schema_info: str, sample_table: str) -> str:
    """Construct the full prompt for Claude."""
    prompt = f"""You are a database expert and data quality analyst. Given the following table schema and a data sample, discover all data integrity constraints that the data should satisfy.

Schema:
{schema_info}

Sample rows (first 12 rows):
{sample_table}

Discover the following types of constraints:

1. **Value Range Constraints**: Valid ranges for numeric columns. Include minimum and/or maximum bounds.
2. **Functional Dependencies**: Columns where one column's value determines another's value (e.g., zip_code -> city).
3. **Cross-Column Rules**: Logical rules involving multiple columns (denial constraints). For example, "if column_A = X then column_B must be Y".
4. **Distribution Hints**: Expected distribution shapes for numeric columns (e.g., "roughly_normal", "right_skewed", "uniform", "bimodal").

Think step by step:
1. First, examine each numeric column's range and identify natural bounds.
2. Look for columns that might have deterministic relationships.
3. Consider domain knowledge about what this data represents.
4. Think about logical implications between categorical values.

Return your answer as JSON with this exact format:
{{
  "value_ranges": [
    {{"column": "col_name", "min": 0, "max": 100}},
    {{"column": "col_name", "min": 0, "max": null}}
  ],
  "functional_dependencies": [
    {{"determinant": ["col_a"], "dependent": "col_b", "description": "col_a determines col_b"}}
  ],
  "cross_column_rules": [
    {{"description": "human readable rule", "columns": ["col_a", "col_b"], "rule": "if col_a == X then col_b == Y", "condition_col": "col_a", "condition_val": "X", "result_col": "col_b", "result_val": "Y"}}
  ],
  "distribution_hints": [
    {{"column": "col_name", "distribution": "right_skewed"}}
  ]
}}

Important:
- For value_ranges, use null if there is no upper or lower bound.
- For cross_column_rules, provide machine-parseable rule descriptions.
- Only include constraints you are confident about based on the schema and data.
- Do NOT invent constraints that are not supported by the data.
- Return ONLY the JSON object, no additional text."""

    return prompt


# ---------------------------------------------------------------------------
# Parse LLM response into Constraint objects
# ---------------------------------------------------------------------------

def _parse_llm_response(response_data: dict) -> list:
    """
    Parse the structured JSON from Claude into Constraint objects.

    Parameters
    ----------
    response_data : dict
        Parsed JSON from Claude's response.

    Returns
    -------
    list of Constraint objects.
    """
    constraints = []

    # Value ranges
    for i, vr in enumerate(response_data.get("value_ranges", [])):
        col = vr.get("column", "")
        min_val = vr.get("min")
        max_val = vr.get("max")
        if col and (min_val is not None or max_val is not None):
            constraints.append(ValueRangeConstraint(
                name=f"llm_range_{col}_{i}",
                column=col,
                min_val=min_val,
                max_val=max_val,
            ))

    # Functional dependencies
    for i, fd in enumerate(response_data.get("functional_dependencies", [])):
        det = fd.get("determinant", [])
        dep = fd.get("dependent", "")
        if det and dep:
            constraints.append(FunctionalDependency(
                name=f"llm_fd_{dep}_{i}",
                determinant=det,
                dependent=dep,
            ))

    # Cross-column rules
    for i, rule in enumerate(response_data.get("cross_column_rules", [])):
        desc = rule.get("description", "")
        columns = rule.get("columns", [])
        cond_col = rule.get("condition_col", "")
        cond_val = rule.get("condition_val", "")
        res_col = rule.get("result_col", "")
        res_val = rule.get("result_val", "")

        if cond_col and res_col and cond_val and res_val:
            # Build a lambda for this specific rule
            # We need to capture values in a closure
            def make_predicate(cc, cv, rc, rv):
                def pred(df):
                    cond_match = df[cc].astype(str).str.strip() == str(cv)
                    res_match = df[rc].astype(str).str.strip() == str(rv)
                    # Rows violate if condition is met but result is not
                    return ~(cond_match & ~res_match)
                return pred

            constraints.append(CrossColumnRule(
                name=f"llm_rule_{i}",
                columns=columns if columns else [cond_col, res_col],
                predicate=make_predicate(cond_col, cond_val, res_col, res_val),
                description=desc,
            ))
        elif columns and desc:
            # Generic rule without machine-parseable parts - skip
            # (we can't enforce what we can't parse)
            pass

    # Distribution hints
    for i, dh in enumerate(response_data.get("distribution_hints", [])):
        col = dh.get("column", "")
        dist = dh.get("distribution", "")
        if col and dist:
            constraints.append(DistributionHint(
                name=f"llm_dist_{col}_{i}",
                column=col,
                distribution=dist,
            ))

    return constraints


def _extract_json_from_text(text: str) -> dict:
    """
    Extract a JSON object from Claude's response text.
    Handles cases where the JSON is wrapped in markdown code blocks.
    """
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON-like structure
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from LLM response")


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_constraints(
    df: pd.DataFrame,
    cat_columns: list = None,
    num_columns: list = None,
    model: str = "claude-sonnet-4-20250514",
    use_cache: bool = True,
    validate: bool = True,
    violation_threshold: float = 0.3,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list:
    """
    Use Claude to discover data constraints from schema and sample.

    Parameters
    ----------
    df : pd.DataFrame
        Training data to analyze.
    cat_columns : list, optional
        Categorical column names.
    num_columns : list, optional
        Numeric column names.
    model : str
        Claude model to use.
    use_cache : bool
        Cache LLM responses to avoid redundant API calls.
    validate : bool
        If True, validate discovered constraints against the data
        and discard those with violation rate > violation_threshold.
    violation_threshold : float
        Max allowed violation rate for a constraint to be kept.
    max_retries : int
        Number of retry attempts on API failure.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    list of Constraint objects.
    """
    # Build schema info and sample
    schema_info = _build_schema_info(df, cat_columns, num_columns)
    sample_table = _build_sample_table(df, n_rows=12)

    # Check cache
    cache_key = _compute_cache_key(schema_info, sample_table)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            print("[LLM Discovery] Using cached response "
                  f"(key={cache_key})")
            constraints = _parse_llm_response(cached)
            if validate:
                constraints = _validate_constraints(
                    constraints, df, violation_threshold
                )
            return constraints

    # Build prompt
    prompt = _build_prompt(schema_info, sample_table)

    # Call Claude API
    api_key = _load_api_key()
    if not api_key:
        print("[LLM Discovery] No API key -- will return empty "
              "(caller handles mock fallback)")
        return []

    try:
        import anthropic
    except ImportError:
        print("[LLM Discovery] anthropic package not installed -- "
              "will return empty (caller handles mock fallback)")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    response_text = None
    for attempt in range(max_retries):
        try:
            print(f"[LLM Discovery] Calling {model} "
                  f"(attempt {attempt + 1}/{max_retries}) ...")
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            response_text = response.content[0].text
            break
        except Exception as e:
            print(f"[LLM Discovery] API call failed: {e}")
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"[LLM Discovery] Retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print("[LLM Discovery] All retries exhausted -- "
                      "returning empty (caller handles mock fallback)")
                return []

    if response_text is None:
        return []

    # Parse response
    try:
        response_data = _extract_json_from_text(response_text)
    except ValueError as e:
        print(f"[LLM Discovery] Failed to parse response: {e}")
        print(f"[LLM Discovery] Raw response:\n{response_text[:500]}")
        return []

    # Cache the parsed response immediately after successful parse
    if use_cache:
        _save_to_cache(cache_key, response_data)
        print(f"[LLM Discovery] Response cached (key={cache_key})")

    # Parse into Constraint objects
    constraints = _parse_llm_response(response_data)
    print(f"[LLM Discovery] Discovered {len(constraints)} raw constraints")

    # Validate
    if validate:
        constraints = _validate_constraints(
            constraints, df, violation_threshold
        )

    return constraints


def _validate_constraints(constraints: list,
                          df: pd.DataFrame,
                          threshold: float = 0.3) -> list:
    """
    Validate constraints against the data.
    Discard constraints with violation rate > threshold.
    """
    validated = []
    for c in constraints:
        if isinstance(c, DistributionHint):
            # Always keep distribution hints
            validated.append(c)
            continue

        try:
            rate = c.violation_rate(df)
            if rate <= threshold:
                validated.append(c)
                print(f"  [OK] {c.name}: violation rate = {rate:.4f}")
            else:
                print(f"  [DROPPED] {c.name}: violation rate = {rate:.4f} "
                      f"> threshold {threshold}")
        except Exception as e:
            print(f"  [ERROR] {c.name}: {e} -- dropping")

    print(f"[LLM Discovery] Validated: {len(validated)} / "
          f"{len(constraints)} constraints kept")
    return validated


# ---------------------------------------------------------------------------
# Mock discovery (for testing without API)
# ---------------------------------------------------------------------------

def mock_discover_constraints(dataset_name: str) -> list:
    """
    Return predefined constraints that simulate LLM discovery.

    Useful for testing without making API calls.
    These constraints overlap with (but are not identical to) ground truth,
    simulating realistic LLM discovery accuracy.
    """
    name = dataset_name.lower().replace(" ", "_")

    if name == "adult":
        return [
            # Correct discoveries
            ValueRangeConstraint("mock_age_min", "age", min_val=17),
            ValueRangeConstraint("mock_hours_range", "hours_per_week",
                                 min_val=1, max_val=99),
            ValueRangeConstraint("mock_capgain", "capital_gain", min_val=0),
            ValueRangeConstraint("mock_caploss", "capital_loss", min_val=0),
            FunctionalDependency("mock_edu_fd", ["education"],
                                 "education_num"),
            # LLM might miss the husband->male rule
            # LLM might add an extra (spurious but valid) constraint
            ValueRangeConstraint("mock_age_max", "age", max_val=90),
            ValueRangeConstraint("mock_edunum_range", "education_num",
                                 min_val=1, max_val=16),
            DistributionHint("mock_age_dist", "age", "right_skewed"),
        ]

    elif name in ("german_credit", "germancredit", "credit"):
        return [
            ValueRangeConstraint("mock_duration", "duration", min_val=1),
            ValueRangeConstraint("mock_credit_amt", "credit_amount",
                                 min_val=1),
            ValueRangeConstraint("mock_age", "age", min_val=18),
            ValueRangeConstraint("mock_installment", "installment_commitment",
                                 min_val=1, max_val=4),
            # LLM might discover an additional range constraint
            ValueRangeConstraint("mock_residence", "residence_since",
                                 min_val=1, max_val=4),
            DistributionHint("mock_credit_dist", "credit_amount",
                             "right_skewed"),
        ]

    elif name == "insurance":
        return [
            ValueRangeConstraint("mock_age_range", "age",
                                 min_val=18, max_val=64),
            ValueRangeConstraint("mock_bmi", "bmi", min_val=0.1),
            ValueRangeConstraint("mock_children", "children", min_val=0),
            ValueRangeConstraint("mock_charges", "charges", min_val=0.01),
            DistributionHint("mock_charges_dist", "charges", "right_skewed"),
            DistributionHint("mock_bmi_dist", "bmi", "roughly_normal"),
        ]

    elif name == "heart":
        return [
            ValueRangeConstraint("mock_age_range", "age",
                                 min_val=28, max_val=80),
            ValueRangeConstraint("mock_bp_range",
                                 "resting_blood_pressure",
                                 min_val=90, max_val=200),
            ValueRangeConstraint("mock_chol_range",
                                 "serum_cholestoral",
                                 min_val=100, max_val=600),
            ValueRangeConstraint("mock_max_hr", "max_heart_rate",
                                 min_val=60, max_val=210),
            ValueRangeConstraint("mock_oldpeak", "oldpeak",
                                 min_val=0.0, max_val=7.0),
            ValueRangeConstraint("mock_vessels",
                                 "number_of_major_vessels",
                                 min_val=0, max_val=3),
            DistributionHint("mock_age_dist", "age", "roughly_normal"),
        ]

    elif name == "diabetes":
        return [
            ValueRangeConstraint("mock_target_range", "target",
                                 min_val=24, max_val=347),
            ValueRangeConstraint("mock_bmi_range", "bmi",
                                 min_val=-0.10, max_val=0.18),
            ValueRangeConstraint("mock_bp_range", "bp",
                                 min_val=-0.13, max_val=0.15),
            ValueRangeConstraint("mock_s1_range", "s1",
                                 min_val=-0.14, max_val=0.17),
            ValueRangeConstraint("mock_s5_range", "s5",
                                 min_val=-0.14, max_val=0.15),
            ValueRangeConstraint("mock_s6_range", "s6",
                                 min_val=-0.15, max_val=0.15),
            DistributionHint("mock_target_dist", "target",
                             "right_skewed"),
        ]

    elif name in ("winequality", "wine_quality", "wine"):
        return [
            ValueRangeConstraint("mock_fixed_acid", "fixed_acidity",
                                 min_val=3.5, max_val=16.5),
            ValueRangeConstraint("mock_volatile_acid",
                                 "volatile_acidity",
                                 min_val=0.08, max_val=1.70),
            ValueRangeConstraint("mock_citric_acid", "citric_acid",
                                 min_val=0.0, max_val=1.1),
            ValueRangeConstraint("mock_ph", "pH",
                                 min_val=2.7, max_val=4.2),
            ValueRangeConstraint("mock_alcohol", "alcohol",
                                 min_val=8.0, max_val=15.5),
            ValueRangeConstraint("mock_sulphates", "sulphates",
                                 min_val=0.2, max_val=2.2),
            ValueRangeConstraint("mock_residual_sugar",
                                 "residual_sugar", min_val=0.5),
            DistributionHint("mock_alcohol_dist", "alcohol",
                             "right_skewed"),
        ]

    else:
        print(f"[Mock] No predefined constraints for '{dataset_name}' "
              f"-- returning generic empty list")
        return []


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing mock constraint discovery ...")
    all_names = [
        "adult", "german_credit", "insurance",
        "heart", "diabetes", "wine",
    ]
    for ds_name in all_names:
        constraints = mock_discover_constraints(ds_name)
        print(f"\n{ds_name}: {len(constraints)} constraints discovered")
        for c in constraints:
            print(f"  {c}")
