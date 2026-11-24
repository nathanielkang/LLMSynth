"""
RelDDPM adapter for the LLMSynth experiment harness.

Wraps the official RelDDPM implementation (ruc-datalab/RelDDPM) vendored under
``third_party/RelDDPM``.  RelDDPM is *controllable*: when ``constraints`` are
passed to ``generate()``, we train a classifier controller on rows that satisfy
the supplied hard predicates (GT or ManualConstraints) versus synthetically
perturbed violations, then apply classifier guidance during reverse diffusion.

This adapter does **not** run LLM discovery; callers must pass explicit
constraint objects (same protocol as ManualConstraints / PostHocRepair).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

from baselines._paths import ensure_relddpm_path, relddpm_available
from constraints import (
    CrossColumnRule,
    DistributionHint,
    FunctionalDependency,
    ValueRangeConstraint,
)

ensure_relddpm_path()

if relddpm_available():
    import data_utils as du
    from ddpm import diffusion, modules, train
    from ddpm.resample import create_named_schedule_sampler
    from lib_oversampling.sample import get_cond_fn

    class _HarnessDataWrapper(du.DataWrapper):
        """DataWrapper with 2-D-safe ValsToBit for OpenML categoricals."""

        def transform(self, data):
            reorder_data = data[self.columns].values
            norm_data = []
            for i, col in enumerate(self.columns):
                col_data = reorder_data[:, i]
                if col in self.all_distinct_values.keys():
                    col_data = self.CatValsToNum(col, col_data).reshape(-1, 1)
                    flat = col_data.reshape(-1)
                    bit_values = np.zeros((flat.shape[0], self.col_dim[i]))
                    for j in range(flat.shape[0]):
                        val = int(np.asarray(flat[j]).item())
                        bits = self.col_dim[i]
                        bit_val = np.mod(
                            np.right_shift(val, list(reversed(np.arange(bits)))), 2
                        )
                        bit_values[j, :] = bit_val
                    norm_data.append(bit_values)
                elif col in self.num_normalizer.keys():
                    norm_data.append(
                        self.num_normalizer[col]
                        .transform(col_data.reshape(-1, 1))
                        .reshape(-1, 1)
                    )
            norm_data = np.concatenate(norm_data, axis=1)
            return norm_data.astype(np.float32)

else:
    du = None  # type: ignore
    _HarnessDataWrapper = None  # type: ignore


def _hard_constraints(constraints):
    return [
        c for c in (constraints or [])
        if not isinstance(c, DistributionHint)
    ]


def _row_satisfies_all(df: pd.DataFrame, constraints) -> pd.Series:
    if not constraints:
        return pd.Series(True, index=df.index)
    ok = pd.Series(True, index=df.index)
    for c in constraints:
        ok &= c.check(df)
    return ok


def _perturb_for_negative(df: pd.DataFrame, constraints, rng: np.random.Generator):
    """Create constraint-violating rows for controller negative class."""
    out = df.copy()
    hard = _hard_constraints(constraints)
    if not hard:
        # Flip a random numeric column if no constraints supplied.
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = rng.choice(num_cols)
            out[col] = out[col].max() * 10 + 1
        return out

    c = hard[rng.integers(0, len(hard))]
    if isinstance(c, ValueRangeConstraint) and c.column in out.columns:
        vals = pd.to_numeric(out[c.column], errors="coerce")
        if c.max_val is not None:
            out[c.column] = vals + (c.max_val - vals.min()) + 1
        elif c.min_val is not None:
            out[c.column] = vals - (vals.max() - c.min_val) - 1
        else:
            out[c.column] = vals * 10
    elif isinstance(c, CrossColumnRule):
        # Swap categorical values to break logical rule when possible.
        cols = [col for col in c.columns if col in out.columns]
        if len(cols) >= 2:
            out[cols[-1]] = out[cols[-1]].sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).values
    elif isinstance(c, FunctionalDependency):
        dep = c.dependent
        if dep in out.columns:
            out[dep] = out[dep].sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).values
    return out


class RelDDPMSynthesizer:
    """RelDDPM baseline — controllable diffusion with classifier guidance."""

    name = "RelDDPM"

    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 4096,
        lr: float = 1.8e-3,
        controller_lr: float = 1e-3,
        hidden_dim=None,
        n_timesteps: int = 1000,
        scale_factor: float = 8.0,
        verbose: bool = False,
        device: str | None = None,
    ):
        if not relddpm_available():
            raise ImportError(
                "RelDDPM source not found. Clone into "
                "2_Code/third_party/RelDDPM "
                "(git clone https://github.com/ruc-datalab/RelDDPM.git)."
            )

        self.epochs = int(epochs)
        self.batch_size = int(min(batch_size, 4096))
        self.lr = lr
        self.controller_lr = controller_lr
        self.hidden_dim = hidden_dim or [256, 512, 512, 256]
        self.n_timesteps = n_timesteps
        self.scale_factor = scale_factor
        self.verbose = verbose
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._data_wrapper = None
        self._diffuser = None
        self._controller = None
        self._col_order = None
        self._cat_columns: list = []
        self._num_columns: list = []
        self._train_constraints = []
        self._is_fitted = False

    def fit(
        self,
        df_train,
        cat_columns=None,
        num_columns=None,
        target_col=None,
        constraints=None,
    ):
        cat_columns = cat_columns or []
        num_columns = num_columns or []
        self._cat_columns = list(cat_columns)
        self._num_columns = list(num_columns)
        self._col_order = list(df_train.columns)
        self._train_constraints = _hard_constraints(constraints or [])

        df_prep = self._prepare_dataframe(df_train)

        self._data_wrapper = _HarnessDataWrapper(num_encoder="quantile")
        self._data_wrapper.fit(df_prep)

        train_x = self._data_wrapper.transform(df_prep)
        n_steps = max(self.epochs * max(1, len(train_x) // self.batch_size), 50)

        model = modules.MLPDiffusion(train_x.shape[1], self.hidden_dim, 0.0)
        model.to(self.device)
        diff_model = diffusion.GaussianDiffusion(
            train_x.shape[1],
            model,
            device=self.device,
            num_timesteps=self.n_timesteps,
        ).to(self.device)
        diff_model.train()

        ds = [torch.from_numpy(train_x).float()]
        dl = du.prepare_fast_dataloader(ds, batch_size=self.batch_size, shuffle=True)
        trainer = train.Trainer(
            diff_model, dl, self.lr, 0.0, n_steps,
            save_path=None, device=self.device,
        )
        if self.verbose:
            print(f"[RelDDPM] Training diffuser for {n_steps} steps ...")
        trainer.run_loop()

        diff_model.eval()
        diff_model.to(self.device)
        diff_model.variables_to_device(self.device)
        self._diffuser = diff_model

        # Controller: binary label 1 = satisfies provided hard constraints.
        self._train_controller(df_prep, train_x)
        self._is_fitted = True
        return self

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self._col_order].copy()
        for col in self._cat_columns:
            if col in out.columns:
                out[col] = out[col].astype(str).str.strip()
        for col in self._num_columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna().reset_index(drop=True)

    def _train_controller(self, df_train: pd.DataFrame, train_x: np.ndarray):
        rng = np.random.default_rng(42)
        pos_df = self._prepare_dataframe(df_train)
        neg_df = _perturb_for_negative(pos_df, self._train_constraints, rng)

        pos_x = self._data_wrapper.transform(pos_df)
        neg_x = self._data_wrapper.transform(neg_df)
        x_all = np.concatenate([pos_x, neg_x], axis=0)
        y_all = np.concatenate([
            np.ones(len(pos_x), dtype=np.float32),
            np.zeros(len(neg_x), dtype=np.float32),
        ])

        n_classes = 2
        controller = modules.MLPClassifier(
            x_all.shape[1],
            [256, 256],
            n_classes,
            0.0,
            t_in=True,
        ).to(self.device)

        x_tensor = torch.from_numpy(x_all).float()
        y_tensor = torch.from_numpy(y_all).float()
        ds = [x_tensor, y_tensor]
        dl = du.prepare_fast_dataloader(ds, batch_size=min(512, len(x_all)), shuffle=True)
        schedule_sampler = create_named_schedule_sampler(
            "uniform", self._diffuser.num_timesteps
        )
        opt = torch.optim.AdamW(
            controller.parameters(), lr=self.controller_lr, weight_decay=1e-5
        )

        steps = max(self.epochs * 5, 100)
        controller.train()
        self._diffuser.to(self.device)
        self._diffuser.variables_to_device(self.device)

        for step in range(steps):
            x, y = next(dl)
            x = x.to(self.device)
            y = y.to(self.device).squeeze()
            t, _ = schedule_sampler.sample(len(y), self.device)
            xt = self._diffuser.gaussian_q_sample(x, t)
            logits = controller(xt, t).squeeze()
            loss = torch.nn.functional.binary_cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if self.verbose and (step + 1) % max(steps // 5, 1) == 0:
                print(f"[RelDDPM] Controller step {step + 1}/{steps} loss={loss.item():.4f}")

        controller.eval()
        controller.to(self.device)
        self._controller = controller

    @torch.no_grad()
    def generate(self, n_samples: int, constraints=None):
        if not self._is_fitted:
            raise RuntimeError("RelDDPM not fitted. Call fit() first.")

        use_constraints = _hard_constraints(constraints)
        if use_constraints and use_constraints != self._train_constraints:
            warnings.warn(
                "RelDDPM generate() received constraints that differ from fit(); "
                "post-hoc repair still applies, but controller was trained on "
                "fit()-time predicates.",
                stacklevel=2,
            )

        self._controller.to(self.device)
        self._diffuser.to(self.device)
        self._diffuser.variables_to_device(self.device)

        cond_fn = get_cond_fn(
            self._controller,
            self.scale_factor,
            label=1.0,
            n_classes=2,
        )
        cond = torch.zeros(n_samples, 1)
        samples = self._diffuser.sample(
            n_samples, control_tools=[cond, cond_fn]
        )
        samples_np = samples.cpu().numpy()
        df_synth = self._data_wrapper.Reverse(samples_np)
        df_synth = df_synth[self._col_order]

        if use_constraints:
            df_synth = self._posthoc_repair(df_synth, use_constraints)

        return df_synth.reset_index(drop=True)

    def _posthoc_repair(self, df: pd.DataFrame, constraints):
        """Clamp ranges and apply simple FD / cross-column fixes."""
        for c in constraints:
            if isinstance(c, ValueRangeConstraint) and c.column in df.columns:
                vals = pd.to_numeric(df[c.column], errors="coerce")
                if c.min_val is not None:
                    vals = vals.clip(lower=c.min_val)
                if c.max_val is not None:
                    vals = vals.clip(upper=c.max_val)
                df[c.column] = vals
            elif isinstance(c, CrossColumnRule):
                mask = ~c.check(df)
                if mask.any() and len(c.columns) >= 2:
                    target = c.columns[-1]
                    if target in df.columns:
                        repl = df[target].sample(
                            n=int(mask.sum()), replace=True, random_state=42
                        ).values
                        df.loc[mask, target] = repl
        return df
