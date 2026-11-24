"""
TabDiff adapter for the LLMSynth experiment harness.

TabDiff (MinkaiXu/TabDiff) targets mixed-type fidelity without semantic
constraints.  Full official training requires the TabDiff conda env and
processed ``data/`` layout; for harness integration we ship a **lightweight
mixed-type diffusion** that follows the same evaluation interface
(``fit`` / ``generate`` → ``DataFrame``) and uses feature-wise Gaussian noise
on encoded numerics plus ordinal categoricals.

This is an **unconstrained** baseline — expect non-zero hard constraint
violations.  For paper-faithful TabDiff numbers, use
``scripts/gcp/run_tabdiff_official.sh`` after ``conda env create -f tabdiff.yaml``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _linear_beta(n_timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


class _FeatureWiseDenoiser(nn.Module):
    """MLP noise predictor with per-feature output scaling (TabDiff-inspired)."""

    def __init__(self, input_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        layers: list[nn.Module] = []
        in_f = input_dim + 128
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_f, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            in_f = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.feature_log_scale = nn.Parameter(torch.zeros(input_dim))

    def _t_emb(self, t):
        half = 64
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < 128:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 128 - emb.shape[1], device=t.device)], dim=1)
        return self.time_embed(emb)

    def forward(self, x, t):
        return self.net(torch.cat([x, self._t_emb(t)], dim=-1)) * torch.exp(self.feature_log_scale)


class _MixedDiffusionCore(nn.Module):
    """Continuous-time-style mixed diffusion on encoded tabular vectors."""

    def __init__(self, input_dim, hidden_dim=256, n_layers=3, n_timesteps=1000):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        betas = _linear_beta(n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.denoiser = _FeatureWiseDenoiser(input_dim, hidden_dim, n_layers)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alpha_bar[t].unsqueeze(-1)
        so = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sa * x0 + so * noise, noise

    def compute_loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (b,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred = self.denoiser(xt, t)
        return (noise - pred).pow(2).mean()

    def train_model(self, x_train, epochs, batch_size, lr, verbose):
        self.train()
        loader = DataLoader(
            TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
        )
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        it = tqdm(range(epochs), desc="TabDiff training", disable=not verbose, ascii=True)
        for _ in it:
            eloss = 0.0
            nb = 0
            for (xb,) in loader:
                opt.zero_grad()
                loss = self.compute_loss(xb)
                loss.backward()
                opt.step()
                eloss += loss.item()
                nb += 1
            it.set_postfix(loss=f"{eloss / max(nb, 1):.4f}")

    @torch.no_grad()
    def sample(self, n_samples, verbose=False):
        self.eval()
        dev = next(self.parameters()).device
        x = torch.randn(n_samples, self.input_dim, device=dev)
        steps = tqdm(range(self.n_timesteps - 1, -1, -1), desc="TabDiff sampling", disable=not verbose, ascii=True)
        for t_val in steps:
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=dev)
            pred = self.denoiser(x, t)
            alpha = self.alphas[t_val]
            ab = self.alpha_bar[t_val]
            beta = self.betas[t_val]
            mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - ab)) * pred)
            if t_val > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.cpu().numpy()


class TabDiffSynthesizer:
    """TabDiff-style mixed-type synthesizer (harness adapter; unconstrained)."""

    name = "TabDiff"

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_timesteps: int = 1000,
        epochs: int = 200,
        batch_size: int = 4096,
        lr: float = 1e-3,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self._model = None
        self._scaler = None
        self._label_encoders = {}
        self._col_order = None
        self._cat_columns: list = []
        self._num_columns: list = []
        self._is_fitted = False

    def fit(self, df_train, cat_columns=None, num_columns=None, target_col=None):
        self._cat_columns = cat_columns or []
        self._num_columns = num_columns or []
        self._col_order = list(df_train.columns)
        df = df_train.copy()
        self._label_encoders = {}
        for col in self._cat_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)
        x = df.values.astype(np.float32)
        self._scaler = StandardScaler()
        x = self._scaler.fit_transform(x)
        self._model = _MixedDiffusionCore(
            input_dim=x.shape[1],
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        self._model.train_model(
            x,
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(x)),
            lr=self.lr,
            verbose=self.verbose,
        )
        self._is_fitted = True
        return self

    def generate(self, n_samples: int, constraints=None):
        if not self._is_fitted:
            raise RuntimeError("TabDiff not fitted. Call fit() first.")
        # TabDiff is fidelity-only; ignore constraints by design.
        xs = self._model.sample(n_samples, verbose=self.verbose)
        x_inv = self._scaler.inverse_transform(xs)
        df = pd.DataFrame(x_inv, columns=self._col_order)
        for col in self._cat_columns:
            if col in df.columns and col in self._label_encoders:
                le = self._label_encoders[col]
                n_cls = len(le.classes_)
                df[col] = df[col].round().clip(0, n_cls - 1).astype(int)
                df[col] = le.inverse_transform(df[col])
        return df.reset_index(drop=True)
