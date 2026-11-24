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

    def constrained_sample(self, n_samples,
                           range_constraints_scaled=None,
                           soft_ranges_scaled=None,
                           guidance_scale=0.5,
                           verbose=False):
        """Reverse sampling with hard-range clamps at every step.

        Soft-range guidance is added only on the last T_g steps
        (manuscript: T_g=200, s=0.5) as epsilon + s * grad C(x0_hat).
        Hard ranges are projected at every step, not only in the suffix.

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
        if soft_ranges_scaled is None:
            soft_ranges_scaled = []

        guidance_steps = 200
        x = torch.randn(n_samples, self.input_dim)
        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Guided sampling",
                         disable=not verbose, ascii=True)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = self.forward(x, t)
            alpha_b = self.alpha_bar[t_val]
            if t_val < guidance_steps and soft_ranges_scaled:
                predicted_noise = predicted_noise + guidance_scale * (
                    self._soft_range_grad(x, predicted_noise, alpha_b,
                                          soft_ranges_scaled)
                )
            alpha = self.alphas[t_val]
            beta = self.betas[t_val]
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)
            if t_val > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
            x = self._clamp_scaled(x, range_constraints_scaled)

        return x.detach().numpy()

    def _clamp_scaled(self, x, ranges):
        for col_idx, min_s, max_s in ranges:
            if min_s is not None:
                x[:, col_idx] = x[:, col_idx].clamp(min=float(min_s))
            if max_s is not None:
                x[:, col_idx] = x[:, col_idx].clamp(max=float(max_s))
        return x

    def _soft_range_grad(self, x, predicted_noise, alpha_bar, soft_ranges):
        """Gradient of the soft range penalty at the Tweedie estimate."""
        work = x.detach().requires_grad_(True)
        eps = predicted_noise.detach()
        denom = torch.sqrt(alpha_bar).clamp(min=1e-8)
        x0 = (work - torch.sqrt(1.0 - alpha_bar) * eps) / denom
        cost = work.new_zeros(())
        for col_idx, min_s, max_s in soft_ranges:
            column = x0[:, col_idx]
            if min_s is not None:
                cost = cost + torch.relu(float(min_s) - column).pow(2).sum()
            if max_s is not None:
                cost = cost + torch.relu(column - float(max_s)).pow(2).sum()
        if not cost.requires_grad or float(cost.detach()) == 0.0:
            return torch.zeros_like(predicted_noise)
        grad, = torch.autograd.grad(cost, work, retain_graph=False)
        return grad.detach()


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

        # Round integer-like numeric columns (incl. binary targets)
        for col in self._num_columns:
            if col in df.columns and hasattr(self, "_train_df") and self._train_df is not None:
                if col not in self._train_df.columns:
                    continue
                vals = pd.to_numeric(self._train_df[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                if np.allclose(vals, np.round(vals)):
                    df[col] = pd.to_numeric(df[col], errors="coerce").round()
                    nunique = vals.nunique()
                    if nunique <= 20:
                        # Snap to observed integer levels when cardinality is small
                        levels = np.sort(vals.unique())
                        arr = df[col].fillna(levels[0]).values.astype(float)
                        nearest = levels[
                            np.abs(arr.reshape(-1, 1) - levels.reshape(1, -1)).argmin(axis=1)
                        ]
                        df[col] = nearest

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
                 max_rejection_retries=25):
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

        constraints = self._with_roles(constraints)
        hard = [c for c in constraints if getattr(c, "role", "hard") == "hard"]
        soft = [c for c in constraints if getattr(c, "role", "") == "soft"]
        range_constraints = [c for c in hard if isinstance(c, ValueRangeConstraint)]
        soft_ranges = [c for c in soft if isinstance(c, ValueRangeConstraint)]
        fd_constraints = [c for c in hard if isinstance(c, FunctionalDependency)]
        rule_constraints = [c for c in hard if isinstance(c, CrossColumnRule)]
        self._last_range_constraints = range_constraints
        self._last_fd_constraints = fd_constraints
        self._hard_constraints = hard

        self._scaled_ranges = self._constraints_to_scaled(range_constraints)
        soft_scaled = self._constraints_to_scaled(soft_ranges)

        X_synth = self._model.constrained_sample(
            n_samples,
            range_constraints_scaled=self._scaled_ranges,
            soft_ranges_scaled=soft_scaled,
            guidance_scale=0.5,
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

        # Closure after FD lookup, then the ablation checkpoint, then the
        # output gate. Completed-candidate violations are recorded before
        # any discard. Emitted rows are those that pass every hard rule.
        df_synth = self._enforce_value_ranges(df_synth, range_constraints)
        df_synth = self._enforce_fds(df_synth, fd_constraints)
        precheck = self._hard_violation_mask(df_synth, hard)
        self.last_precheck_hard_rate = (
            float(precheck.mean()) if len(df_synth) else 0.0
        )
        df_synth = self._release_hard_clean(df_synth, hard, n_samples)
        return df_synth

    def _with_roles(self, constraints):
        if constraints and all(hasattr(c, "role") for c in constraints):
            return constraints
        if self._train_df is None:
            for constraint in constraints:
                if not hasattr(constraint, "role"):
                    constraint.role = "hard"
            return constraints
        from validation import validate_proposals
        return validate_proposals(list(constraints), self._train_df)

    def _hard_violation_mask(self, df, hard):
        mask = pd.Series(False, index=df.index)
        for constraint in hard:
            if isinstance(constraint, DistributionHint):
                continue
            satisfied = constraint.check(df)
            mask = mask | ~satisfied.reindex(df.index).fillna(False)
        return mask

    def _release_hard_clean(self, df, hard, n_samples):
        """Redraw hard-violating rows, then drop any row that still fails."""
        if not hard:
            self.last_release_shortfall = 0
            return df
        for _retry in range(self.max_rejection_retries):
            mask = self._hard_violation_mask(df, hard)
            if not bool(mask.any()):
                break
            n_bad = int(mask.sum())
            X_new = self._model.constrained_sample(
                n_bad,
                range_constraints_scaled=self._scaled_ranges,
                soft_ranges_scaled=[],
                verbose=False,
            )
            df_new = self._postprocess(X_new)
            df_new = self._enforce_value_ranges(
                df_new, self._last_range_constraints
            )
            df_new = self._enforce_fds(df_new, self._last_fd_constraints)
            df.loc[df.index[mask]] = df_new.iloc[:n_bad].to_numpy()
        mask = self._hard_violation_mask(df, hard)
        clean = df.loc[~mask].copy()
        self.last_release_shortfall = int(n_samples - len(clean))
        if self.verbose and self.last_release_shortfall:
            print(
                "  [ConstraintGuided] discarded "
                f"{int(mask.sum())} hard-violating rows"
            )
        return clean

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
        Enforce functional dependencies via deterministic modal lookup.

        For each FD (det -> dep), replace the dependent with the training-mode
        value for that determinant (paper: modal FD projection). Unseen
        determinants fall back to the global training mode.
        """
        if self._train_df is None:
            return df

        for c in constraints:
            det_cols = list(c.determinant)
            dep_col = c.dependent

            missing = [col for col in det_cols + [dep_col]
                       if col not in df.columns or
                       col not in self._train_df.columns]
            if missing:
                continue

            train = self._train_df.copy()
            for col in det_cols:
                train[col] = train[col].astype(str).str.strip()
            work = df.copy()
            for col in det_cols:
                work[col] = work[col].astype(str).str.strip()

            mode_map = (
                train.groupby(det_cols, sort=False)[dep_col]
                .agg(lambda x: x.value_counts().index[0] if len(x) else np.nan)
            )
            global_mode = (
                train[dep_col].value_counts().index[0]
                if len(train) > 0 else np.nan
            )

            if len(det_cols) == 1:
                mapped = work[det_cols[0]].map(mode_map)
            else:
                keys = work[det_cols].apply(tuple, axis=1)
                mapped = keys.map(
                    mode_map.to_dict() if hasattr(mode_map, "to_dict")
                    else dict(mode_map)
                )
            mapped = mapped.fillna(global_mode)

            # Integer-like numeric dependents: store as clean ints
            if dep_col in self._num_columns:
                train_vals = pd.to_numeric(train[dep_col], errors="coerce").dropna()
                if len(train_vals) and np.allclose(train_vals, np.round(train_vals)):
                    mapped = pd.to_numeric(mapped, errors="coerce").round()
                    mapped = mapped.astype(int)

            df[dep_col] = mapped.values
            # Keep determinant string cleanup consistent with map keys
            for col in det_cols:
                if col in self._cat_columns:
                    df[col] = work[col].values

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

            # Re-apply deterministic range/FD fixes after each resample batch
            df = self._enforce_value_ranges(
                df,
                [c for c in getattr(self, "_last_range_constraints", [])],
            )
            df = self._enforce_fds(
                df,
                [c for c in getattr(self, "_last_fd_constraints", [])],
            )

        # Last-resort deterministic repair for residual cross-column violations
        df = self._deterministic_cross_column_repair(df, constraints)

        if self.verbose:
            # Final violation count
            final_violations = 0
            for c in constraints:
                final_violations += c.violation_count(df)
            print(f"  [ConstraintGuided] After enforcement: "
                  f"{final_violations} total constraint violations remain")

        return df

    def _deterministic_cross_column_repair(self, df, constraints):
        """
        Repair remaining CrossColumnRule violations by copying a satisfying
        training row's dependent column values when possible.
        """
        if self._train_df is None or not constraints:
            return df

        for c in constraints:
            satisfied = c.check(df)
            if satisfied.all():
                continue
            viol_idx = df.index[~satisfied]
            cols = [col for col in c.columns if col in df.columns
                    and col in self._train_df.columns]
            if len(cols) < 2:
                continue
            # Prefer fixing the last column in the rule (typical consequent)
            target_col = cols[-1]
            train_ok = self._train_df.loc[c.check(self._train_df)]
            if len(train_ok) == 0:
                continue
            for idx in viol_idx:
                # Condition on other columns when possible
                cond_cols = [col for col in cols if col != target_col]
                row = df.loc[idx]
                subset = train_ok
                for col in cond_cols:
                    matched = subset[
                        subset[col].astype(str) == str(row[col])
                    ]
                    if len(matched) > 0:
                        subset = matched
                pick = subset.sample(1, random_state=0).iloc[0]
                df.at[idx, target_col] = pick[target_col]
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
        One of: LLMSynth, TabDDPM, RelDDPM, TabDiff, PostHocRepair, CTGAN,
        ManualConstraints
    **kwargs
        Passed to synthesizer constructor.

    Returns
    -------
    BaseSynthesizer instance.
    """
    from baselines.relddpm_adapter import RelDDPMSynthesizer
    from baselines.tabdiff_adapter import TabDiffSynthesizer

    registry = {
        "llmsynth": ConstraintGuidedDiffusion,
        "tabddpm": VanillaTabDDPM,
        "relddpm": RelDDPMSynthesizer,
        "tabdiff": TabDiffSynthesizer,
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
