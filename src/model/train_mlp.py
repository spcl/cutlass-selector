#!/usr/bin/env python3
"""
MLP for CUTLASS SM90 BF16 GEMM kernel selection.

Supports three training objectives:
  --loss mse         : pointwise MSE on y_norm (default)
  --loss lambdarank  : NDCG-delta-weighted pairwise logistic
  --loss ranknet     : unweighted pairwise logistic (same pairs, no NDCG weight)

Feature sets:
  --feature-set full        : hardware-aware (default)
  --feature-set structural  : config/problem only, no hardware constants

Evaluation metrics are the same as train_xgb.py (harness.py).

Usage:
    python src/model/train_mlp.py --features artifacts/analysis/paper/features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from feature_manifest import category_levels_for
from features import CATEGORY_LEVELS, FEATURE_SETS, feature_columns
from harness import (
    NDCG_KS,
    base_shape_id,
    baselines,
    grouped_train_val_split,
    ndcg_per_group,
    regime_breakdown,
    select_and_regret,
    summarize,
)


# ----------------------------------------------------------------------
# Feature prep
# ----------------------------------------------------------------------
def _ohe(df: pd.DataFrame, cat_cols: list[str], category_levels: dict | None = None) -> np.ndarray:
    """One-hot encode categoricals using a frozen level order (default: CATEGORY_LEVELS)."""
    category_levels = CATEGORY_LEVELS if category_levels is None else category_levels
    parts = []
    for col in cat_cols:
        levels = category_levels[col]
        vals = df[col].astype(str)
        for lvl in levels:
            parts.append((vals == lvl).to_numpy(dtype="float32"))
    return np.stack(parts, axis=1)


def build_X(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
            category_levels: dict | None = None, scaler: StandardScaler | None = None):
    """Design matrix: standardized numeric features followed by one-hot categoricals.

    Fits a new StandardScaler unless one is passed; returns (X, scaler). category_levels
    fixes the one-hot column order (default: features.CATEGORY_LEVELS).
    """
    num = df[num_cols].to_numpy(dtype="float32")
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler()
        num = scaler.fit_transform(num)
    else:
        num = scaler.transform(num)
    cat = _ohe(df, cat_cols, category_levels)
    return np.concatenate([num, cat], axis=1), scaler


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class MLP(nn.Module):
    """Feed-forward scorer: Linear-ReLU-Dropout blocks of the given widths and a scalar output."""
    def __init__(self, n_in: int, hidden: tuple = (256, 128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score each row of x."""
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------
# Deployment export (torch.export)
# ----------------------------------------------------------------------
class ServingMLP(nn.Module):
    """Serving wrapper that bakes input nan-handling + standardization into the
    graph, so the exported artifact consumes the *raw* feature matrix directly
    (numeric columns first in NUMERIC_FEATURES order, then the one-hot block).

    This keeps the inference side (src/eval/propose.py) model- and
    scaler-agnostic: it just builds the feature matrix and calls the loaded
    program — no architecture or scaler code on the serving side. Standardization
    mirrors build_X exactly: nan_to_num(nan/posinf/neginf -> 0) then (x-mean)/scale.
    """

    def __init__(self, model: MLP, scaler_mean, scaler_scale, n_numeric: int):
        super().__init__()
        self.model = model
        self.n_numeric = n_numeric
        self.register_buffer("mean", torch.as_tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scaler_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize the numeric columns of raw features, then score them."""
        num = torch.nan_to_num(x[:, : self.n_numeric], nan=0.0, posinf=0.0, neginf=0.0)
        num = (num - self.mean) / self.scale
        return self.model(torch.cat([num, x[:, self.n_numeric :]], dim=1))


def export_serving_model(model: MLP, scaler, n_numeric: int, n_in: int, path) -> None:
    """Write the trained MLP (+ standardization) as a torch.export artifact with a
    dynamic candidate (batch) dimension. Served via torch.export.load(path).module()."""
    from torch.export import Dim, export, save

    wrapper = ServingMLP(model.cpu().eval(), scaler.mean_, scaler.scale_, n_numeric).eval()
    example = torch.zeros(8, n_in, dtype=torch.float32)
    ep = export(wrapper, (example,), dynamic_shapes=({0: Dim("n_candidates", min=1)},))
    save(ep, str(path))



def load_init_checkpoint(path: Path, device: str) -> dict:
    """Load a model_mlp.pt checkpoint for fine-tuning, checking it has the fields needed to rebuild the model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state", "n_in", "hidden", "dropout"):
        if key not in ckpt:
            raise ValueError(f"checkpoint {path} missing key {key!r}")
    return ckpt


def scaler_from_checkpoint(ckpt: dict) -> StandardScaler:
    """Rebuild the StandardScaler stored in a checkpoint."""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype="float64")
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype="float64")
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def init_mlp_from_checkpoint(
    ckpt: dict,
    n_in: int,
    hidden: tuple[int, ...],
    dropout: float,
    device: str,
) -> MLP:
    """Build an MLP initialized from a checkpoint.

    If the new input is wider (e.g. extra categorical columns for fused epilogues), the
    checkpoint weights fill the first input columns and the rest keep their fresh init.
    """
    ckpt_n = int(ckpt["n_in"])
    ckpt_hidden = tuple(int(h) for h in ckpt["hidden"])
    if ckpt_hidden != tuple(hidden):
        raise ValueError(f"checkpoint hidden={list(ckpt_hidden)} != --hidden {list(hidden)}")
    if float(ckpt["dropout"]) != float(dropout):
        print(f"warning: checkpoint dropout={ckpt['dropout']} != --dropout {dropout}")
    model = MLP(n_in, hidden, dropout).to(device)
    if ckpt_n == n_in:
        model.load_state_dict(ckpt["model_state"])
        return model
    if ckpt_n > n_in:
        raise ValueError(f"checkpoint n_in={ckpt_n} but target features have n_in={n_in}")
    # Fusion transfer: extra categorical columns (e.g. fusion_kind) append to the input.
    new_sd = model.state_dict()
    old_sd = ckpt["model_state"]
    w_key = "net.0.weight"
    new_sd[w_key][:, :ckpt_n] = old_sd[w_key]
    for key, val in old_sd.items():
        if key == w_key:
            continue
        if key in new_sd and new_sd[key].shape == val.shape:
            new_sd[key] = val
    model.load_state_dict(new_sd)
    print(f"partial init: copied first-layer weights for {ckpt_n}/{n_in} input dims")
    return model


# ----------------------------------------------------------------------
# MSE training (original path — unchanged)
# ----------------------------------------------------------------------
def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: tuple,
    dropout: float,
    seed: int,
    device: str,
    weight_decay: float = 0.0,
    verbose: bool = True,
    on_epoch_end=None,
    model: MLP | None = None,
) -> MLP:
    """Train an MLP with pointwise MSE on y_norm; returns the trained model.

    Pass model to continue training an existing network. on_epoch_end is called after
    every epoch (used to record the validation history).
    """
    torch.manual_seed(seed)
    if model is None:
        model = MLP(X.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = len(X_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            pred = model(X_t[b])
            loss = loss_fn(pred, y_t[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs}  mse={total / steps:.5f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(model, epoch, total / steps)

    return model


# ----------------------------------------------------------------------
# LambdaRank training
# ----------------------------------------------------------------------
class GroupDataset:
    """Pregroups training rows by group_id with optional row cap.

    When capping, the top-8 gain rows are always kept so needle configs
    always participate in pair generation.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                 max_rows: int, seed: int):
        rng = np.random.default_rng(seed)
        self.groups: list[tuple[np.ndarray, np.ndarray]] = []
        for gid in np.unique(group_ids):
            mask = group_ids == gid
            Xi, yi = X[mask], y[mask]
            if max_rows and len(Xi) > max_rows:
                n_needle = min(8, len(yi))
                top_idx = np.argsort(yi)[::-1][:n_needle]
                rest_pool = np.argsort(yi)[::-1][n_needle:]
                n_rest = max_rows - n_needle
                if len(rest_pool) > n_rest:
                    rest_pool = rng.choice(rest_pool, n_rest, replace=False)
                idx = np.concatenate([top_idx, rest_pool])
                Xi, yi = Xi[idx], yi[idx]
            self.groups.append((Xi, yi))
        self.rng = rng

    def __len__(self) -> int:
        return len(self.groups)

    def shuffled_batches(self, batch_groups: int):
        """Yield lists of batch_groups groups in a random order."""
        order = self.rng.permutation(len(self.groups))
        for i in range(0, len(order), batch_groups):
            yield [self.groups[j] for j in order[i : i + batch_groups]]


def _lambdarank_loss_one_group(
    scores: torch.Tensor,
    gains: torch.Tensor,   # raw y_norm values
    max_pairs: int,
    top_k: int,
    gain_mode: str = "linear",
    weighted: bool = True,   # False = RankNet: skip NDCG-delta weight
) -> tuple[torch.Tensor, bool]:
    """Pairwise logistic loss for one group; pair construction is fully GPU-side.

    weighted=True  (LambdaRank): loss = (softplus(sj−si) * delta_ndcg).mean()
    weighted=False (RankNet)   : loss =  softplus(sj−si).mean()

    Pair sampling uses torch.randperm (GPU) seeded by torch.manual_seed before training.
    Returns (loss, skipped). skipped=True when the group has no valid pairs.
    """
    n = len(scores)
    device = scores.device

    # ---- pair construction (GPU) — validity always on original y_norm ----
    gain_order = gains.argsort(descending=True)   # best config first

    # All upper-triangle pairs; indexing via gain_order gives (higher-gain, lower-gain) pairs
    aa, bb = torch.triu_indices(n, n, offset=1, device=device)
    i_idx = gain_order[aa]   # higher-gain config index
    j_idx = gain_order[bb]   # lower-gain config index

    valid = gains[i_idx] > gains[j_idx]
    i_idx, j_idx = i_idx[valid], j_idx[valid]

    if i_idx.numel() == 0:
        return torch.tensor(0.0, device=device), True

    if i_idx.numel() > max_pairs:
        # Needle guarantee: always include pairs involving top-k gain configs
        top_k_mask = torch.zeros(n, dtype=torch.bool, device=device)
        top_k_mask[gain_order[:top_k]] = True
        is_needle = top_k_mask[i_idx] | top_k_mask[j_idx]
        needle_i, needle_j = i_idx[is_needle], j_idx[is_needle]
        other_i, other_j = i_idx[~is_needle], j_idx[~is_needle]
        budget = max(0, max_pairs - needle_i.numel())
        if budget < other_i.numel():
            sel = torch.randperm(other_i.numel(), device=device)[:budget]
            other_i, other_j = other_i[sel], other_j[sel]
        i_idx = torch.cat([needle_i, other_i])
        j_idx = torch.cat([needle_j, other_j])

    si, sj = scores[i_idx], scores[j_idx]          # gradients flow here

    if not weighted:
        # RankNet: unweighted pairwise logistic — no NDCG weight
        return F.softplus(sj - si).mean(), False

    # LambdaRank: NDCG-delta-weighted pairwise logistic
    # Gain transform (no grad — gains is y_norm, a training label not a model param)
    if gain_mode == "exp":
        gains_w = torch.pow(2.0, gains) - 1.0
    else:
        gains_w = gains

    # Current ranks: sort by predicted score descending → rank 1 = highest score
    score_order = scores.detach().argsort(descending=True)
    ranks = torch.empty(n, dtype=torch.float32, device=device)
    ranks[score_order] = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(1.0 + ranks)  # no grad (ranks is detached)

    # IDCG using transformed gains; ideal order by transformed gain (monotonic = same as y_norm)
    ideal_order = gains_w.argsort(descending=True)
    ideal_pos = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    idcg = (gains_w[ideal_order] / torch.log2(1.0 + ideal_pos)).sum()
    if idcg.item() <= 0:
        return torch.tensor(0.0, device=device), True

    di, dj = discounts[i_idx], discounts[j_idx]    # no grad
    gi_w, gj_w = gains_w[i_idx], gains_w[j_idx]   # transformed gains, no grad

    # |ΔNDCG|: constant weight per pair
    delta_ndcg = ((gi_w - gj_w).abs() * (di - dj).abs()) / idcg

    # log(1 + exp(s_j - s_i)) for pairs where g_i > g_j: penalises wrong ordering
    loss = (F.softplus(sj - si) * delta_ndcg).mean()
    return loss, False


def _sanity_check(groups: list, n_check: int = 3) -> None:
    """Print per-group stats for the first n_check groups and total skip count."""
    print(f"\n-- LambdaRank sanity check ({n_check} groups) --")
    for k, (Xi, yi) in enumerate(groups[:n_check]):
        n = len(Xi)
        unique_gains = len(np.unique(yi))
        idcg = float(np.sum(np.sort(yi)[::-1] / np.log2(1.0 + np.arange(1, n + 1))))
        n_pairs = n * (n - 1) // 2
        status = "OK" if idcg > 0 and unique_gains > 1 else "SKIP"
        print(f"  group {k}: n={n:4d}  unique_gains={unique_gains:4d}  "
              f"potential_pairs={n_pairs:7,}  IDCG={idcg:.4f}  {status}")
    all_skipped = sum(1 for _, yi in groups if len(np.unique(yi)) <= 1)
    print(f"  groups with all-equal gains (will be skipped): {all_skipped}/{len(groups)}\n")


def _report_pair_stats(groups: list, max_pairs: int, top_k: int, sample: int = 50) -> None:
    """Report actual pairs per group after needle-guarantee + cap, on a sample of groups."""
    actual: list[int] = []
    for _, yi in groups[:sample]:
        n = len(yi)
        gains_np = yi.astype("float64")
        gain_order = np.argsort(gains_np)[::-1]
        aa, bb = np.triu_indices(n, k=1)
        i_idx = gain_order[aa]
        j_idx = gain_order[bb]
        n_valid = int((gains_np[i_idx] > gains_np[j_idx]).sum())

        if n_valid <= max_pairs:
            actual.append(n_valid)
            continue

        # Replicate needle+cap logic to get real count
        top_k_mask = np.zeros(n, dtype=bool)
        top_k_mask[gain_order[:top_k]] = True
        valid_mask = gains_np[i_idx] > gains_np[j_idx]
        vi, vj = i_idx[valid_mask], j_idx[valid_mask]
        is_needle = top_k_mask[vi] | top_k_mask[vj]
        n_needle = int(is_needle.sum())
        budget = max(0, max_pairs - n_needle)
        n_other = int((~is_needle).sum())
        actual.append(n_needle + min(budget, n_other))

    arr = np.array(actual)
    print(f"actual-pairs-per-group (sample={sample}, after needle+cap):  "
          f"median={int(np.median(arr)):,}  max={int(arr.max()):,}")


def train_lambdarank(
    dataset: GroupDataset,
    model: MLP,
    *,
    epochs: int,
    groups_per_batch: int,
    lr: float,
    max_pairs: int,
    top_k: int,
    device: str,
    seed: int,
    grad_clip: float,
    gain_mode: str,
    weighted: bool = True,
    weight_decay: float = 0.0,
    verbose: bool = True,
    on_epoch_end=None,
) -> MLP:
    """Train with a pairwise loss over pairs within each (shape, layout) group.

    weighted=True weights each pair by its NDCG change (LambdaRank), otherwise the loss is
    unweighted (RankNet). At most max_pairs pairs are sampled per group.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for batch in dataset.shuffled_batches(groups_per_batch):
            # Single forward pass over all rows in the batch
            X_cat = torch.from_numpy(np.concatenate([g[0] for g in batch])).to(device)
            y_cat = torch.from_numpy(np.concatenate([g[1] for g in batch])).to(device)
            scores = model(X_cat)

            loss = torch.zeros(1, device=device)
            active = 0
            start = 0
            for Xi, _ in batch:
                n = len(Xi)
                lg, skipped = _lambdarank_loss_one_group(
                    scores[start : start + n],
                    y_cat[start : start + n],
                    max_pairs, top_k, gain_mode, weighted,
                )
                if not skipped:
                    loss = loss + lg
                    active += 1
                start += n

            if active > 0:
                loss = loss / active
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total += loss.item()
                steps += 1

        if verbose:
            _loss_name = "lambdarank" if weighted else "ranknet"
            print(f"  epoch {epoch:2d}/{epochs}  {_loss_name}={total / max(steps, 1):.5f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(model, epoch, total / max(steps, 1))

    return model


# ----------------------------------------------------------------------
# Per-epoch validation
# ----------------------------------------------------------------------
class Validator:
    """Scores the model on the held-out validation shapes at the end of an epoch.

    Reports the run's own objective as a validation loss, plus harness NDCG -- the
    same NDCG definition train_xgb.py records -- so the two model families land on one
    comparable axis. Pair sampling for the ranking losses is reseeded before every
    pass, so an epoch-to-epoch change in validation loss is the model moving and not
    a different random set of pairs.
    """

    def __init__(self, val_df: pd.DataFrame, X_val: np.ndarray, args, device: str,
                 ndcg_ks=NDCG_KS):
        self.val_df = val_df
        self.X_val = X_val
        self.device = device
        self.args = args
        self.ndcg_ks = ndcg_ks
        self.rows: list[dict] = []
        self.y = torch.from_numpy(val_df["y_norm"].to_numpy(dtype="float32")).to(device)
        self.group_pos = [
            torch.from_numpy(np.asarray(pos)).to(device)
            for pos in val_df.groupby("group_id").indices.values()
        ]
        self.t0 = time.perf_counter()

    def _val_loss(self, scores: torch.Tensor) -> float:
        if self.args.loss == "mse":
            return float(F.mse_loss(scores, self.y).item())
        # Reseed so the sampled pairs are the same every epoch, then restore the
        # generators: seeding globally would otherwise restart the *training* loop's
        # dropout and pair sampling from a fixed seed once per epoch.
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.args.val_seed)
            weighted = self.args.loss == "lambdarank"
            total, active = 0.0, 0
            for pos in self.group_pos:
                lg, skipped = _lambdarank_loss_one_group(
                    scores[pos], self.y[pos],
                    self.args.max_pairs_per_group, 8, self.args.gain, weighted,
                )
                if not skipped:
                    total += float(lg.item())
                    active += 1
            return total / max(active, 1)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def __call__(self, model: MLP, epoch: int, train_loss: float) -> dict:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            scores = torch.from_numpy(predict(model, self.X_val, self.device)).to(self.device)
            val_loss = self._val_loss(scores)
        nd = ndcg_per_group(self.val_df, scores.cpu().numpy(), self.ndcg_ks).mean()
        if was_training:
            model.train()
        row = {
            "model": "mlp",
            "objective": self.args.loss,
            "feature_set": self.args.feature_set,
            "seed": self.args.seed,
            "epoch": epoch,
            "metric": self.args.loss,
            "train_metric": float(train_loss),
            "val_metric": val_loss,
            **{f"val_{k}": float(v) for k, v in nd.items()},
            "elapsed_s": time.perf_counter() - self.t0,
        }
        self.rows.append(row)
        return row


def _epoch_hook(validator, every: int):
    """Wraps a Validator so it only fires every `every` epochs (and on epoch 1)."""
    if validator is None or every <= 0:
        return None

    def hook(model, epoch: int, train_loss: float) -> None:
        if epoch % every and epoch != 1:
            return
        r = validator(model, epoch, train_loss)
        ndcg = "  ".join(f"{k}={r[k]:.4f}" for k in r if k.startswith("val_ndcg@"))
        print(f"       val_loss={r['val_metric']:.5f}  {ndcg}", flush=True)

    return hook


# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
def predict(model: MLP, X: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    """Score X in batches; returns a 1-D array."""
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out.append(model(X_t[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def fit_model(train, X_train, y_train, args, device, verbose: bool = True, on_epoch_end=None,
              init_model: MLP | None = None):
    """Train one model on these rows; returns (model, extra_meta)."""
    log = print if verbose else (lambda *a, **k: None)
    if args.loss == "mse":
        model = train_mlp(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=tuple(args.hidden),
            dropout=args.dropout,
            seed=args.seed,
            device=device,
            weight_decay=args.weight_decay,
            on_epoch_end=on_epoch_end,
            model=init_model,
        )
        extra_meta: dict = {"batch_size": args.batch_size}
    else:
        group_ids = train["group_id"].to_numpy()
        dataset = GroupDataset(
            X_train, y_train, group_ids,
            max_rows=args.max_rows_per_group,
            seed=args.seed,
        )
        if args.max_rows_per_group == 0:
            sizes = np.array([len(yi) for _, yi in dataset.groups])
            n_big = int((sizes * (sizes - 1) // 2 > args.max_pairs_per_group).sum())
            log(f"group-size stats (no cap): max={sizes.max():,}  "
                  f"median={int(np.median(sizes)):,}  "
                  f"groups with >{args.max_pairs_per_group:,} potential pairs: "
                  f"{n_big}/{len(sizes)}")
        if verbose:
            _sanity_check(dataset.groups)
        if verbose:
            _report_pair_stats(dataset.groups, args.max_pairs_per_group, top_k=8)
        if args.gain == "exp" and args.loss == "lambdarank":
            yi0 = dataset.groups[0][1].astype("float64")
            exp0 = np.power(2.0, yi0) - 1.0
            log(f"gain=exp sanity (group 0): "
                  f"y_norm [{yi0.min():.4f}, {yi0.max():.4f}]  "
                  f"2^y-1  [{exp0.min():.4f}, {exp0.max():.4f}]")
        if args.loss == "ranknet" and args.gain == "exp":
            log("note: --gain exp ignored for --loss ranknet (no NDCG delta weight)")
        _weighted = args.loss == "lambdarank"
        model = init_model
        if model is None:
            model = MLP(X_train.shape[1], tuple(args.hidden), args.dropout).to(device)
        model = train_lambdarank(
            dataset, model,
            epochs=args.epochs,
            groups_per_batch=args.groups_per_batch,
            lr=args.lr,
            max_pairs=args.max_pairs_per_group,
            top_k=8,
            device=device,
            seed=args.seed,
            grad_clip=args.grad_clip,
            gain_mode=args.gain,
            weighted=_weighted,
            weight_decay=args.weight_decay,
            on_epoch_end=on_epoch_end,
        )
        extra_meta = {
            "groups_per_batch": args.groups_per_batch,
            "max_pairs_per_group": args.max_pairs_per_group,
            "max_rows_per_group": args.max_rows_per_group,
            "grad_clip": args.grad_clip,
            "gain": args.gain if args.loss == "lambdarank" else "n/a",
        }
    return model, extra_meta


def cross_validate(train, num_cols, cat_cols, args, device, category_levels=None) -> dict:
    """Shape-grouped k-fold regret for THIS configuration (not a hyperparameter search).

    Replaces the true-oracle report when the dataset has no near-exhaustive groups. All
    four layouts of a shape share a fold (harness.base_shape_id), so nothing leaks across
    the split. Regret here is measured against the best *sampled* config in the group, not
    the oracle, so it is optimistic in absolute terms and only comparable between models
    scored on the same folds.
    """
    from sklearn.model_selection import GroupKFold

    groups = base_shape_id(train)
    gkf = GroupKFold(n_splits=args.cv_folds)
    per_fold = []
    for fold, (tr, va) in enumerate(gkf.split(train, groups=groups), 1):
        tr_df, va_df = train.iloc[tr].copy(), train.iloc[va].copy()
        X_tr, scaler = build_X(tr_df, num_cols, cat_cols, category_levels)
        X_va, _ = build_X(va_df, num_cols, cat_cols, category_levels, scaler)
        y_tr = tr_df["y_norm"].to_numpy(dtype="float32")
        torch.manual_seed(args.seed)
        model, _ = fit_model(tr_df, X_tr, y_tr, args, device, verbose=False)
        perg = select_and_regret(va_df, predict(model, X_va, device))
        print(f"  fold {fold}/{args.cv_folds}: held out {va_df['group_id'].nunique()} groups "
              f"({len(va_df):,} rows)")
        per_fold.append(summarize(perg, f"  fold{fold}"))
    keys = ("regret_mean", "regret_median", "regret_p95", "regret_max", "top1", "top5", "within5pct")
    agg = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    agg |= {f"{k}_sd": float(np.std([f[k] for f in per_fold])) for k in keys}
    agg["folds"] = args.cv_folds
    agg["regret_reference"] = "best sampled config in group (not oracle)"
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.parquet from features.py")
    ap.add_argument("--loss", choices=["lambdarank", "ranknet", "mse"], default="mse")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="full",
                    help="full = hardware-aware; structural = config/problem only")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="shape-grouped validation fraction for the training history "
                         "(0 = train on all shapes, no history)")
    ap.add_argument("--val-seed", type=int, default=42, help="seed for the validation split")
    ap.add_argument("--history-every", type=int, default=1,
                    help="record validation metrics every N epochs")
    ap.add_argument("--no-eval", "--skip-eval", dest="no_eval", action="store_true",
                    help="skip scoring the held-out eval groups. Their feature matrix is "
                         "millions of rows and is the peak memory cost of a run; validation "
                         "metrics are unaffected.")
    ap.add_argument("--epochs", type=int, default=144)
    ap.add_argument("--lr", type=float, default=0.0001191129211393104)
    # MSE-specific
    ap.add_argument("--batch-size", type=int, default=2048)
    # LambdaRank-specific
    ap.add_argument("--groups-per-batch", type=int, default=32)
    ap.add_argument("--max-pairs-per-group", type=int, default=32000)
    ap.add_argument("--max-rows-per-group", type=int, default=0,
                    help="row cap per group before pair generation (0 = no cap)")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max-norm for gradient clipping in LambdaRank (0 = disabled)")
    ap.add_argument("--gain", choices=["linear", "exp"], default="linear",
                    help="gain function for LambdaRank delta-NDCG: linear=y_norm, exp=2^y-1")
    # Shared
    ap.add_argument("--dropout", type=float, default=0.043937900354189416)
    ap.add_argument("--weight-decay", type=float, default=1e-07)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 512, 256])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="shape-grouped k-fold regret for this configuration "
                         "(0 = off). Use when the dataset has no true-oracle groups.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="fine-tune from a prior model_mlp.pt (e.g. BF16 pretrain)")
    ap.add_argument("--scaler-policy", choices=["fit", "source"], default="fit",
                    help="fit: StandardScaler on target train (default); "
                         "source: reuse the --init-checkpoint scaler")
    args = ap.parse_args()

    _gain_str = "n/a" if args.loss == "ranknet" else args.gain
    print(f"config: loss={args.loss}  gain={_gain_str}  lr={args.lr}  "
          f"epochs={args.epochs}  max-rows={args.max_rows_per_group}  grad-clip={args.grad_clip}  "
          f"weight-decay={args.weight_decay}")

    fpath = Path(args.features)
    suffix = "" if args.feature_set == "full" else f"_{args.feature_set}"
    outdir = Path(args.outdir) if args.outdir else fpath.parent / f"mlp_{args.loss}{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(str(fpath).rsplit(".", 1)[0] + ".manifest.json").read_text())
    num_cols, cat_cols = feature_columns(manifest, args.feature_set)
    print(f"feature set: {args.feature_set} "
          f"({len(num_cols)} numeric + {len(cat_cols)} categorical)")

    if args.no_eval:
        # Push the split filter and the column projection into the parquet read: the
        # eval rows and the provenance columns (notably the long `name` strings) are
        # dead weight here, and this is a run's peak memory.
        needed = list(dict.fromkeys(
            num_cols + cat_cols
            + ["split", "group_id", "M", "N", "K",
               "mean_tflops", "group_best_tflops", "y_norm", "rank_in_group"]
        ))
        df = pd.read_parquet(fpath, columns=needed, filters=[("split", "==", "train")])
    else:
        df = pd.read_parquet(fpath)
    category_levels = category_levels_for(manifest)
    for c in cat_cols:
        df[c] = pd.Categorical(df[c].astype(str), categories=category_levels[c])

    train = df[df["split"] == "train"].copy()
    ev = df[df["split"] == "eval"].copy() if not args.no_eval else df.iloc[0:0].copy()
    print(
        f"train={len(train):,} rows / {train['group_id'].nunique():,} groups   "
        f"eval={len(ev):,} rows / {ev['group_id'].nunique()} groups"
        + ("   (--no-eval)" if args.no_eval else "")
    )
    del df

    if args.val_frac > 0:
        fit_rows, val_rows = grouped_train_val_split(train, args.val_frac, args.val_seed)
        print(
            f"validation split: fit={len(fit_rows):,} rows / "
            f"{fit_rows['group_id'].nunique():,} groups / "
            f"{pd.unique(base_shape_id(fit_rows)).size} shapes   "
            f"val={len(val_rows):,} rows / {val_rows['group_id'].nunique():,} groups / "
            f"{pd.unique(base_shape_id(val_rows)).size} shapes"
        )
    else:
        fit_rows, val_rows = train, None

    init_ckpt = None
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(args.init_checkpoint)
        init_ckpt = load_init_checkpoint(args.init_checkpoint, "cpu")
        print(f"fine-tune init: {args.init_checkpoint}  scaler_policy={args.scaler_policy}")

    print("\nBuilding feature matrices ...")
    # Scaler is fit on the training rows only: standardizing with validation
    # statistics would leak the held-out shapes into every epoch's input.
    if init_ckpt is not None and args.scaler_policy == "source":
        scaler = scaler_from_checkpoint(init_ckpt)
        X_train, _ = build_X(fit_rows, num_cols, cat_cols, category_levels, scaler)
    else:
        X_train, scaler = build_X(fit_rows, num_cols, cat_cols, category_levels)
    X_val = (build_X(val_rows, num_cols, cat_cols, category_levels, scaler)[0]
             if val_rows is not None else None)
    X_eval = (build_X(ev, num_cols, cat_cols, category_levels, scaler)[0]
              if not ev.empty else None)
    y_train = fit_rows["y_norm"].to_numpy(dtype="float32")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    _gain_display = "n/a" if args.loss == "ranknet" else args.gain
    print(f"device={device}  loss={args.loss}  gain={_gain_display}  "
          f"input_dim={X_train.shape[1]}  hidden={args.hidden}")
    if args.loss in ("lambdarank", "ranknet"):
        print(f"seeded: weight_init=torch({args.seed})  "
              f"group_order=GroupDataset({args.seed})  "
              f"pair_sampling=train_lambdarank({args.seed})")
    print()

    torch.manual_seed(args.seed)

    cv_metrics = None
    if args.cv_folds:
        print(f"\n== Shape-grouped {args.cv_folds}-fold CV (regret vs best sampled config) ==")
        cv_metrics = cross_validate(train, num_cols, cat_cols, args, device, category_levels)
        print(f"  mean over folds: regret_mean={cv_metrics['regret_mean']:.4f}"
              f" (sd {cv_metrics['regret_mean_sd']:.4f})  top1={cv_metrics['top1']:.4f}")
        print("\n== Final fit on all training groups ==")

    init_model = None
    if init_ckpt is not None:
        init_model = init_mlp_from_checkpoint(
            init_ckpt, X_train.shape[1], tuple(args.hidden), args.dropout, device,
        )

    validator = Validator(val_rows, X_val, args, device) if val_rows is not None else None
    t0 = time.perf_counter()
    model, extra_meta = fit_model(
        fit_rows, X_train, y_train, args, device,
        on_epoch_end=_epoch_hook(validator, args.history_every),
        init_model=init_model,
    )
    train_seconds = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"trained {args.epochs} epochs in {train_seconds:.1f}s  "
          f"({n_params:,} parameters)")

    history = pd.DataFrame(validator.rows) if validator is not None else pd.DataFrame()
    if not history.empty:
        history.to_csv(outdir / "training_history_mlp.csv", index=False)
        best = history.loc[history["val_ndcg@10"].idxmax()]
        print(f"best val_ndcg@10={best['val_ndcg@10']:.4f} at epoch {int(best['epoch'])}")

    perg = None
    mlp_eval = None
    if X_eval is not None:
        print(f"\n== Eval on {ev['group_id'].nunique()} true-oracle groups ==")
        perg = select_and_regret(ev, predict(model, X_eval, device))
        mlp_eval = summarize(perg, f"MLP-{args.loss}")
    else:
        mlp_eval = {"label": f"MLP-{args.loss}", "groups": 0, "skipped": True}
        print("\n== No true-oracle eval groups in this dataset ==")
        print("   (load_eval found no near-exhaustive untagged groups; "
              "use --cv-folds N for a grouped-CV estimate)")

    mlp_metrics: dict = {
        "loss": args.loss,
        "feature_set": args.feature_set,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "gain": args.gain if args.loss == "lambdarank" else "n/a",
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "input_dim": int(X_train.shape[1]),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
        "train_seconds": train_seconds,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "scaler_policy": args.scaler_policy if args.init_checkpoint else "fit",
        **extra_meta,
        "eval": mlp_eval,
        "baselines": baselines(ev) if not ev.empty else None,
        "by_regime": regime_breakdown(perg) if perg is not None else None,
        "cv": cv_metrics,
    }
    if not history.empty:
        _best = history.loc[history["val_ndcg@10"].idxmax()]
        mlp_metrics["best_val_ndcg@10"] = float(_best["val_ndcg@10"])
        mlp_metrics["best_epoch"] = int(_best["epoch"])

    xgb_path = fpath.parent / "metrics.json"
    xgb_e = json.loads(xgb_path.read_text()).get("eval") or {} if xgb_path.exists() else {}
    if perg is not None and "regret_mean" in xgb_e:
        print(f"\n== Comparison: MLP-{args.loss} vs XGBoost ==")
        header = f"  {'metric':18}  {'XGB':>8}  {'MLP':>8}  {'delta':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        metric_keys = [
            ("regret_mean",   True),
            ("regret_median", True),
            ("regret_p95",    True),
            ("regret_max",    True),
            ("top1",          False),
            ("top5",          False),
            ("within5pct",    False),
        ]
        for key, lower_is_better in metric_keys:
            xv = xgb_e[key]
            mv = mlp_eval[key]
            delta = mv - xv
            better = delta < 0 if lower_is_better else delta > 0
            mark = " <" if better else "  "
            print(f"  {key:18}  {xv:8.4f}  {mv:8.4f}  {delta:+8.4f}{mark}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "n_in": int(X_train.shape[1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "feature_set": args.feature_set,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
        },
        outdir / "model_mlp.pt",
    )
    # Inference artifact for src/eval/propose.py: standardization baked in,
    # dynamic candidate dimension. The serving side needs no architecture/scaler code.
    export_serving_model(
        model, scaler, n_numeric=len(num_cols), n_in=int(X_train.shape[1]),
        path=outdir / "model_mlp.pt2",
    )
    if perg is not None:
        perg.to_csv(outdir / "eval_regret.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(mlp_metrics, indent=2))
    written = "model_mlp.pt  model_mlp.pt2  metrics.json" + ("  eval_regret.csv" if perg is not None else "")
    if not history.empty:
        written += f"  training_history_mlp.csv ({len(history)} epochs)"
    print(f"\nwrote {outdir}/{written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
