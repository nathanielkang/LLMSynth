"""
synthesizer.py - Tabular data synthesizers for LLMSynth experiments.

Synthesizer methods:
    1. VanillaTabDDPM:             Standard TabDDPM (no constraint awareness)
    2. ConstraintGuidedDiffusion:  TabDDPM + constraint enforcement during sampling
    3. PostHocRepair:              Generate with TabDDPM, then fix violations
    4. CTGANSynthesizer:           CTGAN baseline (via ctgan library)

Each synthesizer exposes:
    .fit(df_train, cat_columns=[], num_columns=[], target_col=None)
    .generate(n_samples, constraints=None) -> pd.DataFrame
"""

import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from constraints import (
    Constraint,
    ValueRangeConstraint,
    FunctionalDependency,
    CrossColumnRule,
    DistributionHint,
)


# ===================================================================
# Diffusion building blocks (reused from paper2, adapted)
# ===================================================================

def linear_beta_schedule(n_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linearly-spaced beta schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, n_timesteps,
                          dtype=torch.float32)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) *
            torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DenoisingMLP(nn.Module):
    """MLP for noise prediction: eps_theta(x_t, t)."""

    def __init__(self, input_dim, time_dim, hidden_dim=256,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)

        layers = []
        in_features = input_dim + time_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


class TabularDiffusionCore(nn.Module):
    """
    Core TabDDPM model (forward/reverse diffusion).

    Shared by VanillaTabDDPM and ConstraintGuidedDiffusion.
    """

    def __init__(self, input_dim, hidden_dim=256, n_layers=3,
                 n_timesteps=1000):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps

        betas = linear_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))

        time_dim = min(128, hidden_dim)
        self.denoiser = DenoisingMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    def forward(self, x, t):
        return self.denoiser(x, t)

    def compute_loss(self, x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = self.forward(x_t, t)
        loss = (noise - predicted_noise).pow(2).mean()
        return loss

    def train_model(self, X_train, epochs=50, batch_size=256,
                    lr=1e-3, verbose=True):
        """Train the diffusion model."""
        self.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        losses = []

        epoch_iter = tqdm(range(epochs), desc="TabDDPM training",
                          disable=not verbose, ascii=True)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            for (x_batch,) in loader:
                optimizer.zero_grad()
                loss = self.compute_loss(x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step()
            epoch_iter.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        return losses

    @torch.no_grad()
    def sample(self, n_samples, verbose=False):
        """Standard DDPM reverse sampling."""
        self.eval()
        x = torch.randn(n_samples, self.input_dim)

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Sampling",
                         disable=not verbose, ascii=True)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            predicted_noise = self.forward(x, t)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        return x.numpy()

    @torch.no_grad()
    def constrained_sample(self, n_samples,
                           range_constraints_scaled=None,
                           verbose=False):
        """Constraint-guided DDPM reverse sampling (late-stage guidance).

        Applies value-range clamping only in the last T_g denoising
        steps (default 200 of 1000), preserving the learned distribution
        during early/mid denoising while steering toward constraint
        satisfaction in the final steps.  This avoids compounding
        distortion from clamping at every step.

        Parameters
        ----------
        n_samples : int
        range_constraints_scaled : list of (col_idx, min_scaled, max_scaled)
            Value-range bounds in standardized (z-score) space.
        verbose : bool

        Returns
        -------
        numpy array of shape (n_samples, input_dim).
        """
        self.eval()
        if range_constraints_scaled is None:
            range_constraints_scaled = []

        # Late-stage guidance length T_g (manuscript: 200 of T=1000 steps).
        guidance_steps = 200

        x = torch.randn(n_samples, self.input_dim)

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Guided sampling",
                         disable=not verbose, ascii=True)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            predicted_noise = self.forward(x, t)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

            # === CONSTRAINT GUIDANCE (late-stage only) ===
            # Only apply clamping in the last `guidance_steps` timesteps.
            # t counts down from 999 to 0, so t < guidance_steps means
            # we are in the final steps of denoising.
            if t_val < guidance_steps:
                for col_idx, min_s, max_s in range_constraints_scaled:
                    if min_s is not None:
                        x[:, col_idx] = x[:, col_idx].clamp(
                            min=float(min_s))
                    if max_s is not None:
                        x[:, col_idx] = x[:, col_idx].clamp(
                            max=float(max_s))

        return x.numpy()


# ===================================================================
# Base synthesizer
# ===================================================================

class BaseSynthesizer:
    """Base class for all tabular synthesizers."""

    def __init__(self, name="BaseSynthesizer"):
        self.name = name
        self._columns = None
        self._cat_columns = []
        self._num_columns = []
        self._target_col = None
        self._label_encoders = {}
        self._scaler = None
        self._col_order = None
        self._is_fitted = False

    def _preprocess(self, df):
        """
        Encode categoricals and standardize numerics.
        Returns numpy array and stores encoders/scaler.
        """
        df = df.copy()
        self._col_order = list(df.columns)
        self._label_encoders = {}

        # Encode categorical columns
        for col in self._cat_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)

        # Standardize numeric columns
        values = df.values.astype(np.float32)
        self._scaler = StandardScaler()
        # Find numeric column indices (all after encoding)
        self._scaler.fit(values)
        values_scaled = self._scaler.transform(values)

        return values_scaled

    def _postprocess(self, X_synth):
        """
        Inverse-transform synthetic numpy array back to a DataFrame.
        """
        # Inverse scale
        X_inv = self._scaler.inverse_transform(X_synth)

        df = pd.DataFrame(X_inv, columns=self._col_order)

        # Round and decode categoricals
        for col in self._cat_columns:
            if col in df.columns and col in self._label_encoders:
                le = self._label_encoders[col]
                n_classes = len(le.classes_)
                # Clamp to valid range and round
                df[col] = df[col].round().clip(0, n_classes - 1).astype(int)
                try:
                    df[col] = le.inverse_transform(df[col])
                except ValueError:
                    # If some values are out of range, map to nearest
                    df[col] = df[col].clip(0, n_classes - 1)
                    df[col] = le.inverse_transform(df[col])

        # Round integer-like numeric columns
        for col in self._num_columns:
            if col in df.columns:
                # Check if original data was integer-like
                if col in self._col_order:
                    pass  # Keep as float for now

        return df

    def fit(self, df_train, cat_columns=None, num_columns=None,
            target_col=None):
        """Fit the synthesizer on training data."""
        raise NotImplementedError

    def generate(self, n_samples, constraints=None):
        """Generate synthetic data."""
        raise NotImplementedError


# ===================================================================
# 1. VanillaTabDDPM
# ===================================================================

class VanillaTabDDPM(BaseSynthesizer):
    """
    Standard TabDDPM without constraint awareness.

    Training: learns p(x) via DDPM.
    Sampling: standard reverse diffusion.
    """

    def __init__(self, hidden_dim=512, n_layers=3, n_timesteps=1000,
                 epochs=200, batch_size=4096, lr=1e-3, verbose=True):
        super().__init__(name="TabDDPM")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self._model = None

    def fit(self, df_train, cat_columns=None, num_columns=None,
            target_col=None):
        self._cat_columns = cat_columns or []
        self._num_columns = num_columns or []
        self._target_col = target_col
        self._columns = list(df_train.columns)

        X_train = self._preprocess(df_train)
        input_dim = X_train.shape[1]

        self._model = TabularDiffusionCore(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        self._model.train_model(
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=self.verbose,
        )
        self._is_fitted = True
        return self

    def generate(self, n_samples, constraints=None):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_synth = self._model.sample(n_samples, verbose=self.verbose)
        df_synth = self._postprocess(X_synth)
        return df_synth


# ===================================================================
# 2. ConstraintGuidedDiffusion (LLMSynth core)
# ===================================================================

class ConstraintGuidedDiffusion(BaseSynthesizer):
    """
    TabDDPM + constraint enforcement during and after sampling.

    Training: standard TabDDPM (no constraint awareness in training).
    Sampling:
        1. Late-stage guided reverse diffusion (range clamp in last T_g steps)
        2. Post-hoc value-range clamping in data space
        3. FD enforcement: nearest valid value (numeric) or mode (categorical)
        4. Cross-column rules: rejection resampling (max 3 retries; see gap doc)
    """

    def __init__(self, hidden_dim=512, n_layers=3, n_timesteps=1000,
                 epochs=200, batch_size=4096, lr=1e-3, verbose=True,
                 max_rejection_retries=3):
        super().__init__(name="LLMSynth")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.max_rejection_retries = max_rejection_retries
        self._model = None
        self._train_df = None  # Keep for FD lookup

    def fit(self, df_train, cat_columns=None, num_columns=None,
            target_col=None):
        self._cat_columns = cat_columns or []
        self._num_columns = num_columns or []
        self._target_col = target_col
        self._columns = list(df_train.columns)
        self._train_df = df_train.copy()

        X_train = self._preprocess(df_train)
        input_dim = X_train.shape[1]

        self._model = TabularDiffusionCore(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        self._model.train_model(
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=self.verbose,
        )
        self._is_fitted = True
        return self

    def generate(self, n_samples, constraints=None):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if constraints is None:
            constraints = []

        # Separate constraint types
        range_constraints = [c for c in constraints
                             if isinstance(c, ValueRangeConstraint)]
        fd_constraints = [c for c in constraints
                          if isinstance(c, FunctionalDependency)]
        rule_constraints = [c for c in constraints
                            if isinstance(c, CrossColumnRule)]

        # Convert range constraints to standardized (z-score) space
        self._scaled_ranges = self._constraints_to_scaled(range_constraints)

        # Step 1: LATE-STAGE guided reverse diffusion -- range constraints
        # are enforced only in the last T_g denoising steps, preserving
        # distribution quality while steering toward constraints.
        X_synth = self._model.constrained_sample(
            n_samples,
            range_constraints_scaled=self._scaled_ranges,
            verbose=self.verbose,
        )
        df_synth = self._postprocess(X_synth)

        # Step 2: Final value-range clamping in data space (belt-and-suspenders)
        df_synth = self._enforce_value_ranges(df_synth, range_constraints)

        # Step 3: Enforce functional dependencies (lookup from training)
        df_synth = self._enforce_fds(df_synth, fd_constraints)

        # Step 4: Cross-column rules (guided rejection sampling)
        df_synth = self._enforce_cross_column_rules(
            df_synth, rule_constraints
        )

        return df_synth

    def _constraints_to_scaled(self, range_constraints):
        """Convert ValueRangeConstraints to standardized (z-score) space.

        Uses the fitted StandardScaler to map data-space bounds to
        the scaled space that the diffusion model operates in.
        """
        if self._scaler is None or not range_constraints:
            return []

        scaled = []
        for c in range_constraints:
            col = c.column
            if col not in self._col_order:
                continue
            col_idx = self._col_order.index(col)

            mean_val = self._scaler.mean_[col_idx]
            scale_val = self._scaler.scale_[col_idx]
            if scale_val == 0:
                continue

            min_s = None
            max_s = None
            if c.min_val is not None:
                min_s = float((c.min_val - mean_val) / scale_val)
            if c.max_val is not None:
                max_s = float((c.max_val - mean_val) / scale_val)

            scaled.append((col_idx, min_s, max_s))

        return scaled

    def _enforce_value_ranges(self, df, constraints):
        """Clamp columns to valid ranges."""
        for c in constraints:
            col = c.column
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if c.min_val is not None:
                vals = vals.clip(lower=c.min_val)
            if c.max_val is not None:
                vals = vals.clip(upper=c.max_val)
            df[col] = vals
        return df

    def _enforce_fds(self, df, constraints):
        """
        Enforce functional dependencies using nearest valid value.

        For each FD (det -> dep):
          - Numeric dependent: pick the nearest valid value from training
            data for the given determinant group (preserves information).
          - Categorical dependent: use the mode (most common value).
        """
        if self._train_df is None:
            return df

        for c in constraints:
            det_cols = c.determinant
            dep_col = c.dependent

            missing = [col for col in det_cols + [dep_col]
                       if col not in df.columns or
                       col not in self._train_df.columns]
            if missing:
                continue

            dep_is_numeric = dep_col in self._num_columns

            if dep_is_numeric:
                # Numeric: pick nearest valid value per determinant group
                train_groups = self._train_df.groupby(
                    det_cols, sort=False
                )
                valid_map = {}
                for key, grp in train_groups:
                    vals = grp[dep_col].dropna().values.astype(float)
                    if len(vals) > 0:
                        valid_map[key] = np.sort(vals)

                new_dep = df[dep_col].copy().astype(float)
                for key, indices in df.groupby(det_cols).groups.items():
                    if key not in valid_map:
                        continue
                    arr = valid_map[key]
                    gen_vals = df.loc[indices, dep_col].values.astype(float)
                    # Vectorized nearest-value lookup via searchsorted
                    insert_pos = np.searchsorted(arr, gen_vals)
                    insert_pos = np.clip(insert_pos, 0, len(arr) - 1)
                    nearest = arr[insert_pos].copy()
                    left_pos = np.clip(insert_pos - 1, 0, len(arr) - 1)
                    left_vals = arr[left_pos]
                    use_left = (np.abs(gen_vals - left_vals)
                                < np.abs(gen_vals - nearest))
                    nearest[use_left] = left_vals[use_left]
                    new_dep.loc[indices] = nearest

                df[dep_col] = new_dep
            else:
                # Categorical: use mode (most common value)
                train_grouped = self._train_df.groupby(det_cols)[dep_col]
                mode_map = train_grouped.agg(
                    lambda x: x.value_counts().index[0]
                    if len(x) > 0 else np.nan
                )
                mapping_dict = (mode_map.to_dict()
                               if hasattr(mode_map, 'to_dict')
                               else dict(mode_map))
                if len(det_cols) == 1:
                    det_key = det_cols[0]
                    mapped = df[det_key].map(mapping_dict)
                else:
                    key_tuples = df[det_cols].apply(tuple, axis=1)
                    mapped = key_tuples.map(mapping_dict)

                mask = mapped.notna()
                if mask.any():
                    df.loc[mask, dep_col] = mapped.loc[mask].values

        return df

    def _enforce_cross_column_rules(self, df, constraints):
        """
        Enforce cross-column rules via rejection sampling.

        For violating rows: regenerate and replace (up to max_retries).
        If still violating after retries, keep the best attempt.
        """
        if not constraints:
            return df

        for retry in range(self.max_rejection_retries):
            # Check which rows violate any constraint
            violation_mask = pd.Series(False, index=df.index)
            for c in constraints:
                satisfied = c.check(df)
                violation_mask = violation_mask | (~satisfied)

            n_violations = violation_mask.sum()
            if n_violations == 0:
                break

            if retry == 0 and self.verbose:
                print(f"  [ConstraintGuided] {n_violations} rows violate "
                      f"cross-column rules, resampling ...")

            # Regenerate violating rows using GUIDED sampling
            violating_idx = df.index[violation_mask].tolist()
            n_resample = len(violating_idx)

            X_new = self._model.constrained_sample(
                n_resample,
                range_constraints_scaled=getattr(
                    self, '_scaled_ranges', []),
                verbose=False,
            )
            df_new = self._postprocess(X_new)

            # Replace violating rows
            for i, idx in enumerate(violating_idx):
                if i < len(df_new):
                    df.loc[idx] = df_new.iloc[i]

        if self.verbose:
            # Final violation count
            final_violations = 0
            for c in constraints:
                final_violations += c.violation_count(df)
            print(f"  [ConstraintGuided] After enforcement: "
                  f"{final_violations} total constraint violations remain")

        return df


# ===================================================================
# 3. PostHocRepair
# ===================================================================

class PostHocRepair(BaseSynthesizer):
    """
    Generate with standard TabDDPM, then repair violations afterwards.

    Repair strategies:
        - Value range: clamp to valid range
        - FD violations: replace dependent with mode for determinant group
        - Cross-column: resample the violating column from training dist
    """

    def __init__(self, hidden_dim=512, n_layers=3, n_timesteps=1000,
                 epochs=200, batch_size=4096, lr=1e-3, verbose=True):
        super().__init__(name="PostHocRepair")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self._model = None
        self._train_df = None

    def fit(self, df_train, cat_columns=None, num_columns=None,
            target_col=None):
        self._cat_columns = cat_columns or []
        self._num_columns = num_columns or []
        self._target_col = target_col
        self._columns = list(df_train.columns)
        self._train_df = df_train.copy()

        X_train = self._preprocess(df_train)
        input_dim = X_train.shape[1]

        self._model = TabularDiffusionCore(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        self._model.train_model(
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=self.verbose,
        )
        self._is_fitted = True
        return self

    def generate(self, n_samples, constraints=None):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if constraints is None:
            constraints = []

        # Step 1: Standard generation (no constraints)
        X_synth = self._model.sample(n_samples, verbose=self.verbose)
        df_synth = self._postprocess(X_synth)

        # Step 2: Post-hoc repair
        for c in constraints:
            if isinstance(c, DistributionHint):
                continue
            elif isinstance(c, ValueRangeConstraint):
                df_synth = self._repair_value_range(df_synth, c)
            elif isinstance(c, FunctionalDependency):
                df_synth = self._repair_fd(df_synth, c)
            elif isinstance(c, CrossColumnRule):
                df_synth = self._repair_cross_column(df_synth, c)

        return df_synth

    def _repair_value_range(self, df, constraint):
        """Clamp column to valid range."""
        col = constraint.column
        if col not in df.columns:
            return df
        vals = pd.to_numeric(df[col], errors="coerce")
        if constraint.min_val is not None:
            vals = vals.clip(lower=constraint.min_val)
        if constraint.max_val is not None:
            vals = vals.clip(upper=constraint.max_val)
        df[col] = vals
        return df

    def _repair_fd(self, df, constraint):
        """Replace dependent column with mode from training data."""
        if self._train_df is None:
            return df

        det_cols = constraint.determinant
        dep_col = constraint.dependent

        missing = [col for col in det_cols + [dep_col]
                   if col not in df.columns or
                   col not in self._train_df.columns]
        if missing:
            return df

        # Build mode mapping from training data
        train_grouped = self._train_df.groupby(det_cols)[dep_col]
        mode_map = train_grouped.agg(
            lambda x: x.value_counts().index[0]
            if len(x) > 0 else np.nan
        )

        mapping_dict = (mode_map.to_dict()
                       if hasattr(mode_map, 'to_dict')
                       else dict(mode_map))
        if len(det_cols) == 1:
            det_key = det_cols[0]
            mapped = df[det_key].map(mapping_dict)
        else:
            key_tuples = df[det_cols].apply(tuple, axis=1)
            mapped = key_tuples.map(mapping_dict)

        # Use mask-based assignment to avoid int64/float64 dtype conflicts
        mask = mapped.notna()
        if mask.any():
            df.loc[mask, dep_col] = mapped.loc[mask].values

        return df

    def _repair_cross_column(self, df, constraint):
        """
        For cross-column violations: resample the second column
        from training distribution (conditioned on first column if possible).
        """
        if self._train_df is None:
            return df

        satisfied = constraint.check(df)
        violating_mask = ~satisfied

        if violating_mask.sum() == 0:
            return df

        # For each violating row, replace the last column in
        # the constraint's column list with a random value from training
        cols = constraint.columns
        if len(cols) < 2:
            return df

        target_col = cols[-1]
        if target_col not in self._train_df.columns:
            return df

        # Sample replacement values from training distribution
        n_violations = violating_mask.sum()
        replacement_vals = self._train_df[target_col].sample(
            n=n_violations, replace=True, random_state=42
        ).values

        df.loc[violating_mask, target_col] = replacement_vals

        return df


# ===================================================================
# 4. CTGANSynthesizer
# ===================================================================

class CTGANSynthesizer(BaseSynthesizer):
    """
    CTGAN baseline synthesizer.

    Uses the ctgan library (Xu et al. 2019).
    Falls back gracefully if ctgan is not installed.
    """

    def __init__(self, epochs=50, batch_size=256, verbose=True):
        super().__init__(name="CTGAN")
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self._ctgan = None
        self._available = True

        try:
            from ctgan import CTGAN  # noqa: F401
        except ImportError:
            self._available = False
            warnings.warn(
                "ctgan not installed -- CTGANSynthesizer will be unavailable. "
                "Install via: pip install ctgan",
                stacklevel=2,
            )

    def fit(self, df_train, cat_columns=None, num_columns=None,
            target_col=None):
        if not self._available:
            self._is_fitted = False
            return self

        from ctgan import CTGAN

        self._cat_columns = cat_columns or []
        self._num_columns = num_columns or []
        self._target_col = target_col
        self._columns = list(df_train.columns)
        self._col_order = list(df_train.columns)

        # CTGAN works directly with DataFrames
        discrete_columns = self._cat_columns.copy()

        self._ctgan = CTGAN(
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(df_train)),
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"[CTGAN] Training for {self.epochs} epochs ...")

        self._ctgan.fit(df_train, discrete_columns=discrete_columns)
        self._is_fitted = True
        return self

    def generate(self, n_samples, constraints=None):
        if not self._available:
            raise RuntimeError(
                "CTGAN not available (ctgan not installed)."
            )
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df_synth = self._ctgan.sample(n_samples)

        # Ensure column order matches
        if self._col_order:
            missing = [c for c in self._col_order
                       if c not in df_synth.columns]
            if not missing:
                df_synth = df_synth[self._col_order]

        return df_synth


# ===================================================================
# Factory function
# ===================================================================

def get_synthesizer(method_name, **kwargs):
    """
    Get a synthesizer by name.

    Parameters
    ----------
    method_name : str
        One of: LLMSynth, TabDDPM, PostHocRepair, CTGAN, ManualConstraints
    **kwargs
        Passed to synthesizer constructor.

    Returns
    -------
    BaseSynthesizer instance.
    """
    registry = {
        "llmsynth": ConstraintGuidedDiffusion,
        "tabddpm": VanillaTabDDPM,
        "posthocrepair": PostHocRepair,
        "ctgan": CTGANSynthesizer,
        "manualconstraints": ConstraintGuidedDiffusion,
    }

    key = method_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in registry:
        raise ValueError(
            f"Unknown synthesizer '{method_name}'. "
            f"Choose from: {list(registry.keys())}"
        )

    synth = registry[key](**kwargs)

    # Adjust name for ManualConstraints
    if key == "manualconstraints":
        synth.name = "ManualConstraints"

    return synth


# ===================================================================
# Quick smoke test
# ===================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # Create dummy data
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.normal(50000, 15000, size=n).clip(0),
        "education": np.random.choice(["HS", "BS", "MS", "PhD"], size=n),
        "target": np.random.choice(["A", "B"], size=n),
    })

    # Test VanillaTabDDPM
    print("=== VanillaTabDDPM ===")
    synth = VanillaTabDDPM(hidden_dim=64, n_layers=2, n_timesteps=100,
                           epochs=5, verbose=True)
    synth.fit(df, cat_columns=["education", "target"],
              num_columns=["age", "income"])
    df_synth = synth.generate(50)
    print(f"Generated {len(df_synth)} rows")
    print(df_synth.head())

    # Test ConstraintGuidedDiffusion
    print("\n=== ConstraintGuidedDiffusion ===")
    constraints = [
        ValueRangeConstraint("age_range", "age", min_val=18, max_val=70),
        ValueRangeConstraint("income_pos", "income", min_val=0),
    ]
    synth2 = ConstraintGuidedDiffusion(
        hidden_dim=64, n_layers=2, n_timesteps=100,
        epochs=5, verbose=True
    )
    synth2.fit(df, cat_columns=["education", "target"],
               num_columns=["age", "income"])
    df_synth2 = synth2.generate(50, constraints=constraints)
    print(f"Generated {len(df_synth2)} rows")
    print(df_synth2.head())
    print(f"Age range: [{df_synth2['age'].min()}, {df_synth2['age'].max()}]")
