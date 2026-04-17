"""Compact single-file pertflow training and evaluation code for ablation studies."""

import argparse
import math
import os
import json
import warnings
from collections import defaultdict, namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from ema_pytorch import EMA
from scipy.optimize import linear_sum_assignment
from safetensors.torch import load_file, save_file
from scipy.stats import pearsonr
from torch import nn, tensor
from torch.nn import Module
from torch.utils.data import Dataset
from transformers.modeling_outputs import SequenceClassifierOutput

import wandb



def exists(v):
    """Return whether a value is not ``None``."""
    return v is not None


def default(v, d):
    """Return ``v`` when it exists, otherwise return the default ``d``."""
    return v if exists(v) else d


def identity(t):
    """Return the input unchanged."""
    return t


def append_dims(t, dims):
    """Append singleton dimensions to a tensor shape."""
    shape = t.shape
    ones = (1,) * dims
    return t.reshape(*shape, *ones)


def logit_normal_schedule(t, loc=0.0, scale=1.0):
    """Map uniform samples to a logit-normal time schedule."""
    logits = torch.logit(t, eps=1e-5)
    return 1.0 - torch.sigmoid(
        logits * scale + loc
    )  # sticking with 0 -> 1 convention of noise to data


def cosine_sim_loss(x, y):
    """Return mean cosine distance between two representation batches."""
    return 1.0 - F.cosine_similarity(x, y, dim=-1).mean()


LossBreakdown = namedtuple("LossBreakdown", ["total", "flow_loss", "repr_loss"])


@dataclass(frozen=True, order=True)
class ConditionKey:
    """Hashable identifier for a celltype and perturbation pairing."""

    celltype: str
    pert_idx: int


@dataclass
class PerturbationPool:
    """In-memory expression data plus indices used for OT-matched sampling."""

    expression: torch.Tensor
    obs: pd.DataFrame
    pert_indices: np.ndarray
    control_by_celltype: dict[str, np.ndarray]
    target_by_condition: dict[ConditionKey, np.ndarray]
    conditions_by_perturbation: dict[int, list[ConditionKey]]

    @property
    def condition_keys(self) -> list[ConditionKey]:
        """Return sorted perturbation conditions available in the pool."""
        return sorted(self.target_by_condition)

    @property
    def perturbation_ids(self) -> list[int]:
        """Return sorted perturbation ids represented in the pool."""
        return sorted(self.conditions_by_perturbation)


def choose_device(device_arg: str | None = None) -> str:
    """Choose an accelerator string from CLI input or available hardware."""
    if device_arg:
        return device_arg
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def autocast_context(device: str):
    """Return an autocast context for supported accelerator types."""
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_config_relative_path(path_value: str, config_path: Path) -> Path:
    """Resolve a config path relative to the directory containing that config."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def mse_loss(pred, target, reduction="mean"):
    """Small wrapper around ``torch.nn.functional.mse_loss``."""
    return F.mse_loss(pred, target, reduction=reduction)


def extract_perturbation_labels(obs: pd.DataFrame) -> list[str | None]:
    """Extract perturbation gene labels from supported AnnData obs schemas."""
    if "genotype" in obs.columns:
        labels: list[str | None] = []
        for value in obs["genotype"]:
            if pd.isna(value):
                labels.append(None)
                continue
            gene = str(value).strip()
            labels.append(None if not gene or gene.lower() == "ctrl" else gene)
        return labels

    if "condition" in obs.columns:
        return parse_perturbation_labels(obs["condition"])

    raise KeyError("Expected obs['condition'] or obs['genotype'] for perturbation labels")


def parse_perturbation_labels(conditions: pd.Series) -> list[str | None]:
    """Parse ``ctrl`` and ``ctrl+GENE`` condition strings into gene labels."""
    pert_labels: list[str | None] = []

    for value in conditions:
        if pd.isna(value):
            pert_labels.append(None)
            continue

        condition = str(value).strip()
        if condition == "ctrl":
            pert_labels.append(None)
            continue

        if condition.startswith("ctrl+"):
            gene = condition.split("+", 1)[1].strip()
            pert_labels.append(gene or None)
            continue

        raise ValueError(f"Unsupported condition value: {condition!r}")

    return pert_labels


def load_pert_embeddings(cache_path, pert_genes) -> tuple[dict[str, int], torch.Tensor]:
    """Load perturbation embeddings and validate coverage for observed genes."""
    cache_path = Path(cache_path)

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {cache_path}, got {type(payload)!r}")

    if "gene_to_idx" not in payload or "embeddings" not in payload:
        raise KeyError(
            f"Perturbation cache {cache_path} must contain 'gene_to_idx' and 'embeddings'"
        )

    gene_to_idx = {str(gene): int(idx) for gene, idx in payload["gene_to_idx"].items()}
    embeddings = payload["embeddings"]

    if isinstance(embeddings, np.ndarray):
        embedding_matrix = torch.from_numpy(embeddings).float()
    elif torch.is_tensor(embeddings):
        embedding_matrix = embeddings.float().cpu()
    else:
        embedding_matrix = torch.tensor(embeddings, dtype=torch.float32)

    if embedding_matrix.ndim != 2:
        raise ValueError(
            f"Expected perturbation embedding matrix to be 2D, got {embedding_matrix.shape}"
        )

    expected_rows = max(gene_to_idx.values(), default=0) + 1
    if embedding_matrix.shape[0] < expected_rows:
        raise ValueError(
            f"Embedding matrix in {cache_path} has {embedding_matrix.shape[0]} rows, "
            f"but mapping requires at least {expected_rows}"
        )

    expected_genes = {gene for gene in pert_genes if gene is not None}
    missing_genes = sorted(expected_genes - set(gene_to_idx))
    if missing_genes:
        raise KeyError(
            f"Missing perturbation embeddings for genes: {', '.join(missing_genes)}"
        )

    return gene_to_idx, embedding_matrix


def build_condition_index(
    obs: pd.DataFrame, pert_indices: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[ConditionKey, np.ndarray]]:
    """Index control cells and perturbation targets by celltype-aware condition."""
    if "celltype" not in obs.columns:
        raise KeyError("Training requires obs['celltype'] to restrict control pools")

    control_by_celltype: dict[str, list[int]] = defaultdict(list)
    target_by_condition: dict[ConditionKey, list[int]] = defaultdict(list)

    celltypes = obs["celltype"].astype(str).to_numpy()
    for idx, (celltype, pert_idx) in enumerate(zip(celltypes, pert_indices.tolist(), strict=False)):
        pert_idx = int(pert_idx)
        if pert_idx == 0:
            control_by_celltype[celltype].append(idx)
            continue
        target_by_condition[ConditionKey(celltype=celltype, pert_idx=pert_idx)].append(idx)

    control_arrays = {
        key: np.asarray(indices, dtype=np.int64) for key, indices in control_by_celltype.items()
    }
    filtered_targets = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in target_by_condition.items()
        if key.celltype in control_arrays
    }
    return control_arrays, filtered_targets


def build_pool(expression: np.ndarray, obs: pd.DataFrame, pert_indices: np.ndarray) -> PerturbationPool:
    """Materialize a ``PerturbationPool`` from arrays and metadata."""
    local_obs = obs.reset_index(drop=True).copy()
    local_pert_indices = np.asarray(pert_indices, dtype=np.int64)
    control_by_celltype, target_by_condition = build_condition_index(local_obs, local_pert_indices)
    conditions_by_perturbation: dict[int, list[ConditionKey]] = defaultdict(list)

    for condition_key in sorted(target_by_condition):
        conditions_by_perturbation[condition_key.pert_idx].append(condition_key)

    return PerturbationPool(
        expression=torch.as_tensor(np.asarray(expression, dtype=np.float32)),
        obs=local_obs,
        pert_indices=local_pert_indices,
        control_by_celltype=control_by_celltype,
        target_by_condition=target_by_condition,
        conditions_by_perturbation=dict(conditions_by_perturbation),
    )


def split_indices_by_group(
    obs: pd.DataFrame,
    pert_indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split cells into train and validation sets while preserving group structure."""
    rng = np.random.default_rng(seed)
    grouped_indices: dict[tuple[str, str, int], list[int]] = defaultdict(list)

    celltypes = obs["celltype"].astype(str).to_numpy()
    for idx, (celltype, pert_idx) in enumerate(zip(celltypes, pert_indices.tolist(), strict=False)):
        pert_idx = int(pert_idx)
        group_key = ("ctrl" if pert_idx == 0 else "pert", celltype, pert_idx)
        grouped_indices[group_key].append(idx)

    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for indices in grouped_indices.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)

        n_val = 0
        if val_fraction > 0.0 and len(shuffled) > 1:
            n_val = int(round(len(shuffled) * val_fraction))
            n_val = min(max(n_val, 1), len(shuffled) - 1)

        if n_val > 0:
            val_indices.append(shuffled[:n_val])
            train_indices.append(shuffled[n_val:])
        else:
            train_indices.append(shuffled)

    train_idx = np.concatenate(train_indices) if train_indices else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_indices) if val_indices else np.empty(0, dtype=np.int64)
    return train_idx, val_idx


def build_train_val_pools(
    expression: np.ndarray,
    obs: pd.DataFrame,
    pert_indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[PerturbationPool, PerturbationPool | None]:
    """Construct train and optional validation pools from one annotated dataset."""
    train_idx, val_idx = split_indices_by_group(obs, pert_indices, val_fraction, seed)
    train_pool = build_pool(expression[train_idx], obs.iloc[train_idx], pert_indices[train_idx])
    val_pool = None

    if len(val_idx) > 0:
        candidate_pool = build_pool(expression[val_idx], obs.iloc[val_idx], pert_indices[val_idx])
        if candidate_pool.condition_keys:
            val_pool = candidate_pool

    return train_pool, val_pool


def load_conditioned_adata(
    data_path: Path,
    pert_cache_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, int], torch.Tensor]:
    """Load dense expression values and perturbation indices from an h5ad file."""
    adata = sc.read_h5ad(data_path)
    obs = adata.obs.reset_index(drop=True).copy()
    pert_labels = extract_perturbation_labels(obs)
    gene_to_idx, pert_embedding_matrix = load_pert_embeddings(pert_cache_path, pert_labels)
    pert_indices = np.asarray(
        [0 if gene is None else gene_to_idx[gene] for gene in pert_labels],
        dtype=np.int64,
    )
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    expression = np.asarray(matrix, dtype=np.float32)
    return expression, obs, pert_indices, gene_to_idx, pert_embedding_matrix


def sample_indices(indices: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample indices with replacement only when the pool is too small."""
    replace = len(indices) < size
    return rng.choice(indices, size=size, replace=replace)


def sample_condition_key(pool: PerturbationPool, rng: np.random.Generator) -> ConditionKey:
    """Sample a perturbation condition uniformly over perturbation ids."""
    if not pool.conditions_by_perturbation:
        raise ValueError("No perturbation conditions are available for sampling")

    pert_idx = int(rng.choice(pool.perturbation_ids))
    candidates = pool.conditions_by_perturbation[pert_idx]
    return candidates[int(rng.integers(len(candidates)))]


def require_ot():
    """Import POT lazily and raise a clear installation error if it is missing."""
    try:
        import ot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python Optimal Transport (`pot`) is required for OT matching. "
            "Install it with `uv add pot` or `pip install pot`."
        ) from exc

    return ot


def sanitize_transport_plan(
    plan: torch.Tensor | np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an OT plan to a finite nonnegative tensor and normalize its mass."""
    if not torch.is_tensor(plan):
        plan = torch.as_tensor(plan, device=device, dtype=dtype)
    else:
        plan = plan.to(device=device, dtype=dtype)

    plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    total_mass = float(plan.sum().item())
    if total_mass > 0:
        plan = plan / total_mass
    return plan


def transport_plan_has_mass(plan: torch.Tensor) -> bool:
    """Return whether a sanitized transport plan contains positive mass."""
    return bool(plan.ndim == 2 and float(plan.sum().item()) > 0 and torch.any(plan > 0))


def deterministic_transport_plan(cost: torch.Tensor) -> torch.Tensor:
    """Build a deterministic nonempty pairing when OT solvers fail numerically."""
    n_source, n_target = cost.shape
    plan = torch.zeros_like(cost)
    if n_source == 0 or n_target == 0:
        return plan

    if n_source == n_target:
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        row_ind = torch.as_tensor(row_ind, device=cost.device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind, device=cost.device, dtype=torch.long)
        plan[row_ind, col_ind] = 1.0 / len(row_ind)
        return plan

    target_idx = cost.argmin(dim=1)
    source_idx = torch.arange(n_source, device=cost.device)
    plan[source_idx, target_idx] = 1.0 / n_source
    return plan


def compute_transport_plan(
    source_repr: torch.Tensor,
    target_repr: torch.Tensor,
    solver: str = "sinkhorn",
    reg: float = 0.05,
):
    """Compute an entropic OT plan between control and target representations."""
    if solver != "sinkhorn":
        raise ValueError(f"Unsupported OT solver: {solver!r}")
    if reg <= 0:
        raise ValueError(f"OT regularization must be positive, got {reg}")

    ot = require_ot()
    source_repr = source_repr.float()
    target_repr = target_repr.float()
    cost = torch.cdist(source_repr, target_repr, p=2).pow(2)
    # Normalize the cost to a bounded [0, 1] range. Without this, the squared-
    # distance scale grows with the latent representation as training
    # progresses, and a fixed entropic `reg` eventually underflows Sinkhorn.
    cost = cost - cost.min()
    cost_max = cost.max()
    if float(cost_max.item()) > 0:
        cost = cost / cost_max

    a = torch.full(
        (source_repr.shape[0],),
        1.0 / max(source_repr.shape[0], 1),
        device=cost.device,
        dtype=cost.dtype,
    )
    b = torch.full(
        (target_repr.shape[0],),
        1.0 / max(target_repr.shape[0], 1),
        device=cost.device,
        dtype=cost.dtype,
    )

    # Log-domain Sinkhorn is numerically stable against the exp(-cost/reg)
    # underflow that plain Sinkhorn hits when cost/reg gets large.
    plan = sanitize_transport_plan(
        ot.sinkhorn(a, b, cost, reg=reg, method="sinkhorn_log", numItermax=1000, warn=False),
        device=cost.device,
        dtype=cost.dtype,
    )
    if transport_plan_has_mass(plan):
        return plan

    plan = sanitize_transport_plan(
        ot.bregman.sinkhorn_stabilized(
            a,
            b,
            cost,
            reg=reg,
            numItermax=2000,
            warn=False,
        ),
        device=cost.device,
        dtype=cost.dtype,
    )
    if transport_plan_has_mass(plan):
        return plan

    warnings.warn(
        "OT solvers returned an empty transport plan; "
        "falling back to deterministic nearest-neighbor pairing for this batch.",
        RuntimeWarning,
        stacklevel=2,
    )
    return deterministic_transport_plan(cost)


def sample_hard_pairs(
    transport_plan: torch.Tensor,
    num_pairs: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw hard source-target pairs from the nonzero mass of an OT plan."""
    if transport_plan.ndim != 2:
        raise ValueError(
            f"Expected a 2D transport plan, got shape {tuple(transport_plan.shape)}"
        )

    flat_plan = transport_plan.reshape(-1).clamp(min=0)
    positive_mask = flat_plan > 0
    if not torch.any(positive_mask):
        raise ValueError("Transport plan has no positive mass to sample from")

    positive_indices = torch.nonzero(positive_mask, as_tuple=False).squeeze(-1)
    weights = flat_plan[positive_mask]
    weights = weights / weights.sum()
    sampled_positions = torch.multinomial(
        weights,
        num_samples=num_pairs,
        replacement=True,
        generator=generator,
    )
    flat_indices = positive_indices[sampled_positions]
    n_target = transport_plan.shape[1]
    source_idx = torch.div(flat_indices, n_target, rounding_mode="floor")
    target_idx = flat_indices % n_target
    return source_idx.long(), target_idx.long()


class SCRNADataset(Dataset):
    """Minimal dataset wrapper for expression, target, and perturbation tensors."""

    def __init__(
        self,
        source_expression=None,
        targets=None,
        pert_indices=None,
        tokens=None,
    ):
        base_expression = source_expression if source_expression is not None else tokens
        if base_expression is None:
            raise ValueError("SCRNADataset requires `source_expression` or `tokens`")

        self.source_expression = torch.as_tensor(base_expression, dtype=torch.float32)
        self.targets = (
            None if targets is None else torch.as_tensor(targets, dtype=torch.float32)
        )
        self.pert_indices = (
            None if pert_indices is None else torch.as_tensor(pert_indices, dtype=torch.long)
        )

    def __len__(self):
        return int(self.source_expression.shape[0])

    def __getitem__(self, idx):
        item = {
            "source_expr": self.source_expression[idx],
            "input_ids": self.source_expression[idx],
        }

        if self.targets is not None:
            item["labels"] = self.targets[idx]
        if self.pert_indices is not None:
            item["pert_idx"] = self.pert_indices[idx]
        return item


class ValueEncoder(Module):
    """Encode either continuous expression values or discrete bins per gene."""

    def __init__(self, nbins, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(max(nbins + 2, 2), embed_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        if torch.is_floating_point(x):
            return self.value_mlp(x.unsqueeze(-1))
        return self.token_embedding(x.long())


class ExprDecoder(Module):
    """Decode genewise hidden states back into expression values."""

    def __init__(self, embed_dim):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.ffn(x).squeeze(-1)


class Attention(Module):
    """Multi-head self-attention over the gene dimension."""

    def __init__(
        self,
        dim,
        nheads=4,
        dim_head=32,
        flash=True,
    ):
        super().__init__()

        self.scale = dim_head**-0.5
        self.nheads = nheads
        hidden_dim = dim_head * nheads

        self.flash = flash

        self.norm = torch.nn.modules.normalization.RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.nheads), qkv
        )

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            q = q * self.scale
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class EncoderBlock(Module):
    """Residual attention block used in the expression encoder."""

    def __init__(self, dim, nheads, dim_head, ff_mult=4):
        super().__init__()
        self.attn = Attention(dim, nheads, dim_head)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class ConditionEncoder(nn.Module):
    """
    Combines sample encodings and perturbation embeddings into a single
    global conditioning vector, projected back to the model's hidden dimension.
    """

    def __init__(self, dim: int, pert_dim: int, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.pert_dim = pert_dim

        # MLP to learn non-linear interactions between cell state and perturbation
        self.mlp = nn.Sequential(
            nn.Linear(dim + pert_dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(
        self, sample_repr: torch.Tensor, pert_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample_repr: (b, n_genes, dim) - Gene-level encodings
            pert_repr: (b, pert_dim) - ESM C perturbation embeddings
        Returns:
            cond_vector: (b, n_genes, dim) - Global context broadcasted to all genes
        """
        b, n_genes, _ = sample_repr.shape

        # 1. Pool the sample encoding to a single global vector (b, dim)
        # Using mean pooling as a simple permutation-invariant aggregator
        pooled_sample = sample_repr.mean(dim=1)

        # 2. Concatenate sample and perturbation: (b, dim + pert_dim)
        combined = torch.cat([pooled_sample, pert_repr], dim=-1)

        # 3. Project back to the target hidden dimension: (b, dim)
        global_cond = self.mlp(combined)

        # 4. Broadcast the global cell condition to match sequence length: (b, n_genes, dim)
        cond_vector = repeat(global_cond, "b d -> b n d", n=n_genes)

        return cond_vector


class SinusoidalPosEmb(Module):
    """Standard sinusoidal embedding for scalar diffusion or flow times."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb  # (..., 1) * (half_dim,) -> (..., half_dim)
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMBlock(Module):
    """Feed-forward block modulated by a time-dependent FiLM projection."""

    def __init__(self, dim, time_dim, ff_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x, t):
        h = self.norm(x)
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        # if t was (b, time_dim), scale/shift are (b, dim) — need unsqueeze
        # if t was (b, n, time_dim), scale/shift are (b, n, dim) — already matches h
        if scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        h = h * (1 + scale) + shift
        return x + self.ff(h)


class FlowHead(Module):
    """Genewise velocity predictor conditioned on time and encoded context."""

    def __init__(self, dim, depth=4, time_dim=None):
        super().__init__()
        self.dim = dim
        time_dim = default(time_dim, dim * 4)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.x_proj = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([FiLMBlock(dim, time_dim) for _ in range(depth)])
        self.out = nn.Linear(dim, 1)

    def forward(self, x, times, cond, return_hiddens=False):
        """Predict per-gene flow vectors, optionally returning hidden states."""
        # x: (b, n_genes), cond: (b, n_genes, dim), times: (b,)
        h = self.x_proj(x.unsqueeze(-1)) + cond
        t = self.time_mlp(times)
        hiddens = []
        for block in self.blocks:
            h = block(h, t)
            if return_hiddens:
                hiddens.append(h)
        out = self.out(h).squeeze(-1)
        if return_hiddens:
            return out, hiddens
        return out


class DirectPathFlow(Module):
    """Simple rectified-flow objective on straight paths between source and target."""

    def __init__(
        self,
        model: Module,
        times_cond_kwarg="times",
        max_timesteps=100,
        loss_fn=mse_loss,
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.max_timesteps = max_timesteps
        self.loss_fn = loss_fn

    @torch.no_grad()
    def sample(self, source, cond, steps=16):
        """Integrate the learned vector field from a source state."""
        x = source.clone()
        delta = 1.0 / steps
        times = torch.linspace(0.0, 1.0, steps + 1, device=source.device)[:-1]

        for time in times:
            time_batch = repeat(time, "-> b", b=source.shape[0])
            time_kwarg = (
                {self.times_cond_kwarg: time_batch}
                if exists(self.times_cond_kwarg)
                else dict()
            )
            pred_flow = self.model(x, cond=cond, **time_kwarg)
            x = x + delta * pred_flow

        return x

    def forward(self, source, target, cond, times=None, loss_reduction="mean"):
        """Compute direct path flow loss at random interpolation times."""
        batch = source.shape[0]
        device = source.device

        if times is None:
            times = torch.rand(batch, device=device)
            times = times * (1.0 - self.max_timesteps**-1)

        padded_times = append_dims(times, source.ndim - 1)
        x_t = source.lerp(target, padded_times)
        velocity_target = target - source

        time_kwarg = (
            {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict()
        )
        pred_flow = self.model(x_t, cond=cond, **time_kwarg)
        return self.loss_fn(pred_flow, velocity_target, reduction=loss_reduction)


class SelfFlow(Module):
    """Source→target flow matching with an EMA teacher and representation alignment.

    Drop-in replacement for ``DirectPathFlow``: same ``(source, target, cond, times)``
    forward and ``(source, cond, steps)`` sample interface. Supervises the velocity
    ``target - source`` along the straight path ``source.lerp(target, t)`` while
    additionally aligning a student hidden state to an EMA-teacher hidden state
    computed at a cleaner (max-of-two) timestep, with per-patch dual-time masking
    on the student input.
    """

    def __init__(
        self,
        model: Module,
        teacher_model: Module | None = None,
        times_cond_kwarg="times",
        max_timesteps=100,
        loss_fn=mse_loss,
        repr_loss_fn=cosine_sim_loss,
        repr_loss_weight=1.0,
        mask_ratio=0.5,
        patch_size: int = 1,
        student_align_layer=-2,
        teacher_align_layer=-1,
        schedule_fn=logit_normal_schedule,
        eps=1e-5,
        ema_kwargs=None,
    ):
        super().__init__()
        self.model = model

        ema_kwargs = default(
            ema_kwargs,
            dict(beta=0.999, update_every=1, include_online_model=False),
        )
        self.teacher_model = default(teacher_model, EMA(model, **ema_kwargs))
        if not isinstance(self.teacher_model, EMA):
            raise TypeError("teacher model must be an instance of ema_pytorch.EMA")

        self.times_cond_kwarg = times_cond_kwarg
        self.max_timesteps = max_timesteps

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.schedule_fn = schedule_fn
        self.eps = eps

        self.student_align_layer = student_align_layer
        self.teacher_align_layer = teacher_align_layer

        dim = model.dim
        self.projector = nn.Sequential(
            nn.RMSNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

        self.has_repr_loss = repr_loss_weight > 0.0

        self.loss_fn = loss_fn
        self.repr_loss_fn = repr_loss_fn
        self.repr_loss_weight = repr_loss_weight

        self.register_buffer("zero", tensor(0.0), persistent=False)

    def post_training_step_update(self):
        """Update the EMA teacher after an optimizer step."""
        self.teacher_model.update()

    @torch.no_grad()
    def sample(self, source, cond, steps=16, model="teacher"):
        """Integrate the learned vector field from a source state."""
        if model not in ("student", "teacher"):
            raise ValueError(f"Unknown sample model {model!r}; expected 'student' or 'teacher'")
        if not 1 <= steps <= self.max_timesteps:
            raise ValueError(f"steps must be in [1, {self.max_timesteps}], got {steps}")

        selected_model = self.teacher_model if model == "teacher" else self.model

        x = source.clone()
        delta = 1.0 / steps
        times = torch.linspace(0.0, 1.0, steps + 1, device=source.device)[:-1]

        for t in times:
            t_batch = repeat(t, "-> b", b=source.shape[0])
            time_kwarg = (
                {self.times_cond_kwarg: t_batch}
                if exists(self.times_cond_kwarg)
                else dict()
            )
            out = selected_model(x, cond=cond, **time_kwarg)
            pred = out[0] if isinstance(out, tuple) else out
            x = x + delta * pred

        return x

    def forward(
        self,
        source,
        target,
        cond,
        times=None,
        loss_reduction="mean",
        return_loss_breakdown=False,
    ):
        """Compute the source→target SelfFlow objective and alignment loss."""
        shape, device = source.shape, source.device
        batch, seq_len = shape[0], shape[1]

        patch_size = self.patch_size
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if seq_len % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} must evenly divide sequence length {seq_len}"
            )
        num_patches = seq_len // patch_size

        # Two time samples per example: explicit `times` override the student sample
        # so callers (and tests) can control the student interpolation exactly.
        if times is None:
            student_time = self.schedule_fn(torch.rand(batch, device=device))
        else:
            student_time = times
        teacher_time = self.schedule_fn(torch.rand(batch, device=device))

        # Teacher always sees a state at least as clean as the student's.
        times_clean_teacher = torch.maximum(teacher_time, student_time)

        flow = target - source

        teacher_time_patch = repeat(teacher_time, "b -> b n", n=num_patches)
        student_time_patch = repeat(student_time, "b -> b n", n=num_patches)

        mask = torch.rand((batch, num_patches), device=device) < self.mask_ratio
        times_clean_student_patch = torch.where(
            mask, student_time_patch, teacher_time_patch
        )

        times_student_elem = repeat(
            times_clean_student_patch, "b n -> b (n p)", p=patch_size
        )
        student_input = source.lerp(target, times_student_elem)

        time_kwarg = (
            {self.times_cond_kwarg: times_clean_student_patch}
            if exists(self.times_cond_kwarg)
            else dict()
        )

        pred, student_hiddens = self.model(
            student_input, cond=cond, return_hiddens=True, **time_kwarg
        )

        flow_loss = self.loss_fn(pred, flow, reduction=loss_reduction)

        repr_loss = self.zero
        if self.has_repr_loss:
            self.teacher_model.eval()
            with torch.no_grad():
                time_kwarg_teacher = (
                    {self.times_cond_kwarg: times_clean_teacher}
                    if exists(self.times_cond_kwarg)
                    else dict()
                )
                times_clean_teacher_padded = append_dims(
                    times_clean_teacher, source.ndim - 1
                )
                teacher_input = source.lerp(target, times_clean_teacher_padded)

                _, teacher_hiddens = self.teacher_model(
                    teacher_input, cond=cond, return_hiddens=True, **time_kwarg_teacher
                )

            student_repr = student_hiddens[self.student_align_layer]
            student_pred_teacher = self.projector(student_repr)
            teacher_repr = teacher_hiddens[self.teacher_align_layer]
            repr_loss = self.repr_loss_fn(student_pred_teacher, teacher_repr)

        total_loss = flow_loss + repr_loss * self.repr_loss_weight

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, flow_loss, repr_loss)


class PerturbationModel(Module):
    """End-to-end perturbation predictor with MLP or flow-based decoding heads."""

    def __init__(
        self,
        dim,
        nheads,
        dim_head,
        nlayers,
        nbins,
        head_type="flow",
        flow_mode="self",
        pert_dim=0,
        pert_embedding_matrix=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_type = head_type
        self.flow_mode = flow_mode
        pert_embedding_tensor = None
        if pert_embedding_matrix is not None:
            pert_embedding_tensor = torch.as_tensor(
                pert_embedding_matrix, dtype=torch.float32
            )

            inferred_pert_dim = int(pert_embedding_tensor.shape[1])
            if pert_dim == 0:
                pert_dim = inferred_pert_dim
            elif pert_dim != inferred_pert_dim:
                raise ValueError(
                    f"pert_dim={pert_dim} does not match embedding width {inferred_pert_dim}"
                )

        self.pert_dim = pert_dim

        self.encoder = nn.Sequential(
            ValueEncoder(nbins, dim),
            *[EncoderBlock(dim, nheads, dim_head) for _ in range(nlayers)],
        )

        self.pert_embedding = None
        self.condition_encoder = None
        if self.pert_dim > 0:
            self.condition_encoder = ConditionEncoder(dim, pert_dim)
            if pert_embedding_tensor is not None:
                self.pert_embedding = nn.Embedding.from_pretrained(
                    pert_embedding_tensor, freeze=True
                )

        if head_type == "flow":
            if flow_mode == "self":
                self.head = SelfFlow(FlowHead(dim), times_cond_kwarg="times")
            elif flow_mode == "direct":
                self.head = DirectPathFlow(FlowHead(dim), times_cond_kwarg="times")
            else:
                raise ValueError(
                    f"unknown flow_mode: {flow_mode!r} (expected 'self' or 'direct')"
                )
        elif head_type == "mlp":
            self.head = ExprDecoder(dim)
        else:
            raise ValueError(f"unknown head_type: {head_type!r} (expected 'flow' or 'mlp')")

    def encode_expression(self, expression: torch.Tensor) -> torch.Tensor:
        """Encode a batch of gene-expression vectors into per-gene hidden states."""
        return self.encoder(expression)

    def pooled_representation(self, expression: torch.Tensor) -> torch.Tensor:
        """Pool per-gene encoder features into one vector per cell."""
        return self.encode_expression(expression).mean(dim=1)

    def get_match_representation(self, expression: torch.Tensor) -> torch.Tensor:
        """Return the representation space used for OT matching."""
        return self.pooled_representation(expression)

    def lookup_perturbation(
        self,
        pert_idx: torch.Tensor | None = None,
        pert_repr: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Resolve a perturbation embedding from explicit vectors or indices."""
        if pert_repr is not None:
            return pert_repr

        if pert_idx is None:
            return None

        if self.pert_embedding is None:
            raise ValueError(
                "pert_idx was provided, but this model was constructed without "
                "a perturbation embedding matrix"
            )
        return self.pert_embedding(pert_idx)

    def build_condition(
        self,
        source_expression: torch.Tensor | None = None,
        source_encoding: torch.Tensor | None = None,
        pert_idx: torch.Tensor | None = None,
        pert_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the conditioning tensor consumed by the prediction head."""
        encoded = source_encoding
        if encoded is None:
            if source_expression is None:
                raise ValueError("Either source_expression or source_encoding must be provided")
            encoded = self.encode_expression(source_expression.float())

        pert_representation = self.lookup_perturbation(pert_idx=pert_idx, pert_repr=pert_repr)
        if pert_representation is None or self.condition_encoder is None:
            return encoded
        return self.condition_encoder(encoded, pert_representation)

    def forward(
        self,
        input_ids=None,
        source_expr=None,
        labels=None,
        pert_idx=None,
        pert_repr=None,
        steps=16,
        times=None,
        **kwargs,
    ):
        """Run training loss computation or sampling for the selected head type."""
        source_expression = source_expr if source_expr is not None else input_ids
        if source_expression is None:
            raise ValueError("Expected `source_expr` or `input_ids`")

        source_expression = source_expression.float()
        expr_encoding = self.encode_expression(source_expression)
        cond = self.build_condition(
            source_encoding=expr_encoding,
            pert_idx=pert_idx,
            pert_repr=pert_repr,
        )

        target_expression = None if labels is None else labels.float()

        if self.head_type == "mlp":
            preds = self.head(cond)
            loss = F.mse_loss(preds, target_expression) if target_expression is not None else None
            return SequenceClassifierOutput(loss=loss, logits=preds)

        loss = None
        if target_expression is not None:
            loss = self.head(
                source=source_expression,
                target=target_expression,
                cond=cond,
                times=times,
            )

        if target_expression is not None and self.training:
            if self.flow_mode == "self":
                self.head.post_training_step_update()
            return SequenceClassifierOutput(loss=loss, logits=None)

        sampled = self.head.sample(source=source_expression, cond=cond, steps=steps)
        return SequenceClassifierOutput(loss=loss, logits=sampled)


def tokenizer(data, nbins):
    """Discretize continuous expression values into evenly spaced bins."""
    bins = np.linspace(data.min(), data.max(), nbins)
    binned = np.digitize(data, bins)

    return binned.astype(int)


def pairwise_squared_distances(x, y):
    """Compute squared Euclidean distances between rows of two arrays."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def rbf_mmd(x, y):
    """Estimate an RBF-kernel MMD between two sample sets."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    sigma_sample = np.concatenate([x[:512], y[:512]], axis=0)
    sigma_dist = pairwise_squared_distances(sigma_sample, sigma_sample)
    positive_distances = sigma_dist[sigma_dist > 0]
    sigma = np.sqrt(np.median(positive_distances)) if positive_distances.size else 1.0
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0

    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * pairwise_squared_distances(x, x))
    k_yy = np.exp(-gamma * pairwise_squared_distances(y, y))
    k_xy = np.exp(-gamma * pairwise_squared_distances(x, y))
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for constant inputs."""
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y)[0])


def compute_pair_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute pairwise and distributional metrics for predicted perturbation effects."""
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    return {
        "pair_mse": float(np.mean((preds - targets) ** 2)),
        "pair_mae": float(np.mean(np.abs(preds - targets))),
        "pair_pearson": float(
            np.nanmean([safe_pearson(targets[i], preds[i]) for i in range(len(preds))])
        ),
        "dist_mmd": rbf_mmd(preds, targets),
        "pair_cell_mean_mae": float(
            np.mean(np.abs(preds.mean(axis=1) - targets.mean(axis=1)))
        ),
    }


def load_model(
    config: dict,
    checkpoint_path: Path,
    device: str,
    pert_embedding_matrix: torch.Tensor | None = None,
) -> PerturbationModel:
    """Reconstruct a trained ``PerturbationModel`` from config and weights."""
    model = PerturbationModel(
        dim=config["d_model"],
        nheads=config["nheads"],
        dim_head=config["dim_head"],
        nlayers=config["num_layers"],
        nbins=config["nbins"],
        head_type=config["head_type"],
        flow_mode=config.get("flow_mode", "direct"),
        pert_dim=config.get(
            "pert_dim",
            0 if pert_embedding_matrix is None else int(pert_embedding_matrix.shape[1]),
        ),
        pert_embedding_matrix=pert_embedding_matrix,
    )
    state_dict = load_file(str(checkpoint_path))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def build_ot_paired_batch(
    model: PerturbationModel,
    pool: PerturbationPool,
    condition_key: ConditionKey,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    ot_match_space: str,
    ot_solver: str,
    ot_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a control-target batch and pair it using OT in the chosen space."""
    control_indices = sample_indices(pool.control_by_celltype[condition_key.celltype], batch_size, rng)
    target_indices = sample_indices(pool.target_by_condition[condition_key], batch_size, rng)

    source_batch = pool.expression[control_indices].to(device)
    target_batch = pool.expression[target_indices].to(device)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        if ot_match_space == "latent":
            source_repr = model.get_match_representation(source_batch)
            target_repr = model.get_match_representation(target_batch)
        elif ot_match_space == "expression":
            source_repr = source_batch
            target_repr = target_batch
        else:
            raise ValueError(
                f"Unknown OT match space: {ot_match_space!r} (expected 'latent' or 'expression')"
            )

        transport_plan = compute_transport_plan(
            source_repr,
            target_repr,
            solver=ot_solver,
            reg=ot_reg,
        )
        source_match_idx, target_match_idx = sample_hard_pairs(
            transport_plan,
            num_pairs=batch_size,
        )

    if model_was_training:
        model.train()

    matched_source = source_batch[source_match_idx]
    matched_target = target_batch[target_match_idx]
    pert_batch = torch.full(
        (batch_size,),
        condition_key.pert_idx,
        device=device,
        dtype=torch.long,
    )
    return matched_source, matched_target, pert_batch, transport_plan


def evaluate_model(
    model: PerturbationModel,
    pool: PerturbationPool | None,
    device: str,
    batch_size: int,
    steps: int,
    ot_match_space: str,
    ot_solver: str,
    ot_reg: float,
    max_conditions: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Evaluate a model on OT-paired validation batches across conditions."""
    if pool is None or not pool.condition_keys:
        return {}, {}

    rng = np.random.default_rng(seed)
    model_was_training = model.training
    model.eval()

    condition_keys = pool.condition_keys
    if max_conditions is not None and max_conditions < len(condition_keys):
        selected = rng.choice(len(condition_keys), size=max_conditions, replace=False)
        condition_keys = [condition_keys[int(idx)] for idx in np.sort(selected)]

    aggregate_lists: dict[str, list[float]] = defaultdict(list)
    per_condition: dict[str, dict[str, float]] = {}

    for condition_key in condition_keys:
        source_batch, target_batch, pert_batch, _ = build_ot_paired_batch(
            model=model,
            pool=pool,
            condition_key=condition_key,
            batch_size=batch_size,
            device=device,
            rng=rng,
            ot_match_space=ot_match_space,
            ot_solver=ot_solver,
            ot_reg=ot_reg,
        )

        with torch.no_grad(), autocast_context(device):
            preds = model(
                source_expr=source_batch,
                pert_idx=pert_batch,
                steps=steps,
            ).logits

        metric_dict = compute_pair_metrics(
            preds.float().cpu().numpy(),
            target_batch.float().cpu().numpy(),
        )
        per_condition[f"{condition_key.celltype}|pert_{condition_key.pert_idx}"] = metric_dict

        for metric_name, metric_value in metric_dict.items():
            aggregate_lists[metric_name].append(metric_value)

    if model_was_training:
        model.train()

    aggregate = {
        metric_name: float(np.nanmean(values))
        for metric_name, values in aggregate_lists.items()
        if values
    }
    aggregate["n_conditions"] = float(len(per_condition))
    return aggregate, per_condition


def save_checkpoint(
    model: PerturbationModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    global_step: int,
    metrics: dict[str, float],
):
    """Persist model weights, optimizer state, and scalar metrics for one step."""
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(checkpoint_dir / "model.safetensors"))
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    with open(checkpoint_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    return checkpoint_dir


def train_one_epoch(
    model: PerturbationModel,
    pool: PerturbationPool,
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    steps_per_epoch: int,
    ot_match_space: str,
    ot_solver: str,
    ot_reg: float,
    flow_steps: int,
    rng: np.random.Generator,
    log_every: int,
    epoch_index: int,
    num_epochs: int,
) -> float:
    """Run one epoch of OT-paired perturbation training."""
    model.train()
    running_loss = 0.0

    for step in range(steps_per_epoch):
        condition_key = sample_condition_key(pool, rng)
        source_batch, target_batch, pert_batch, _ = build_ot_paired_batch(
            model=model,
            pool=pool,
            condition_key=condition_key,
            batch_size=batch_size,
            device=device,
            rng=rng,
            ot_match_space=ot_match_space,
            ot_solver=ot_solver,
            ot_reg=ot_reg,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss = model(
                source_expr=source_batch,
                labels=target_batch,
                pert_idx=pert_batch,
                steps=flow_steps,
            ).loss

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        if (step + 1) % max(log_every, 1) == 0 or step == 0:
            avg_so_far = running_loss / (step + 1)
            print(
                f"Epoch {epoch_index + 1}/{num_epochs} "
                f"step {step + 1}/{steps_per_epoch} train_loss={avg_so_far:.4f}"
            )

    return running_loss / max(steps_per_epoch, 1)


def parse_train_args():
    """Parse command-line arguments for the single-file training entry point."""
    root = Path(__file__).resolve().parent
    default_pert_cache = Path(
        os.getenv("PERT_CACHE_PATH", root / "esmc_pert_embeddings.pt")
    )

    parser = argparse.ArgumentParser(
        description="Train pertflow with OT-matched control-to-perturbed batches."
    )
    parser.add_argument("--data-path", type=Path, default=root / "train_val.h5ad")
    parser.add_argument("--pert-cache-path", type=Path, default=default_pert_cache)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--head-type",
        choices=("flow", "mlp"),
        default=os.getenv("HEAD_TYPE", "flow"),
    )
    parser.add_argument(
        "--flow-mode",
        choices=("self", "direct"),
        default=os.getenv("FLOW_MODE", "self"),
    )
    parser.add_argument("--ot-solver", choices=("sinkhorn",), default="sinkhorn")
    parser.add_argument(
        "--ot-match-space",
        choices=("latent", "expression"),
        default="latent",
    )
    parser.add_argument("--ot-reg", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-max-conditions", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default="pertflow")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.getenv("WANDB_MODE", "online"),
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    """Train pertflow end to end and periodically evaluate on validation conditions."""
    args = parse_train_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.device)
    print(f"Using {device} device")

    output_dir = args.output_dir
    if output_dir is None:
        suffix = (
            f"{args.head_type}-{args.flow_mode}"
            if args.head_type == "flow"
            else args.head_type
        )
        output_dir = Path(f"pert-model-ot-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expression, obs, pert_indices, gene_to_idx, pert_embedding_matrix = load_conditioned_adata(
        args.data_path,
        args.pert_cache_path,
    )
    print(
        f"Loaded {expression.shape[0]} cells, {expression.shape[1]} genes, "
        f"{len(gene_to_idx)} perturbation embeddings"
    )

    train_pool, val_pool = build_train_val_pools(
        expression=expression,
        obs=obs,
        pert_indices=pert_indices,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if not train_pool.condition_keys:
        raise ValueError("No valid training conditions remain after celltype-restricted pairing")

    steps_per_epoch = default(
        args.steps_per_epoch,
        max(
            len(train_pool.perturbation_ids),
            math.ceil(sum(len(v) for v in train_pool.target_by_condition.values()) / args.batch_size),
        ),
    )

    config = {
        "head_type": args.head_type,
        "flow_mode": args.flow_mode,
        "d_model": args.d_model,
        "nheads": args.nheads,
        "dim_head": args.dim_head,
        "num_layers": args.num_layers,
        "nbins": args.nbins,
        "expr_space": "continuous",
        "pert_dim": int(pert_embedding_matrix.shape[1]),
        "pert_cache_path": str(args.pert_cache_path.resolve()),
        "pairing_key": "celltype",
        "ot_solver": args.ot_solver,
        "ot_match_space": args.ot_match_space,
        "ot_reg": args.ot_reg,
        "pair_sampling": "hard",
        "perturbation_sampling": "uniform",
        "flow_steps": args.flow_steps,
        "data_path": str(args.data_path.resolve()),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    model = PerturbationModel(
        dim=args.d_model,
        nheads=args.nheads,
        dim_head=args.dim_head,
        nlayers=args.num_layers,
        nbins=args.nbins,
        head_type=args.head_type,
        flow_mode=args.flow_mode,
        pert_dim=int(pert_embedding_matrix.shape[1]),
        pert_embedding_matrix=pert_embedding_matrix,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    torch.set_float32_matmul_precision("high")

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=output_dir.name,
        mode=args.wandb_mode,
        config={
            **config,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": args.learning_rate,
        },
    )

    rng = np.random.default_rng(args.seed)
    best_pair_mse = float("inf")
    global_step = 0

    for epoch_idx in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            pool=train_pool,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            steps_per_epoch=steps_per_epoch,
            ot_match_space=args.ot_match_space,
            ot_solver=args.ot_solver,
            ot_reg=args.ot_reg,
            flow_steps=args.flow_steps,
            rng=rng,
            log_every=args.log_every,
            epoch_index=epoch_idx,
            num_epochs=args.epochs,
        )
        global_step += steps_per_epoch

        val_metrics, _ = evaluate_model(
            model=model,
            pool=val_pool,
            device=device,
            batch_size=args.batch_size,
            steps=args.flow_steps,
            ot_match_space=args.ot_match_space,
            ot_solver=args.ot_solver,
            ot_reg=args.ot_reg,
            max_conditions=args.eval_max_conditions,
            seed=args.seed + epoch_idx,
        )

        epoch_metrics = {"train_loss": train_loss, **val_metrics}
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=output_dir,
            global_step=global_step,
            metrics=epoch_metrics,
        )

        pair_mse = val_metrics.get("pair_mse")
        if pair_mse is not None and pair_mse < best_pair_mse:
            best_pair_mse = pair_mse
            save_file(model.state_dict(), str(output_dir / "best_model.safetensors"))

        message = f"Epoch {epoch_idx + 1}/{args.epochs} train_loss={train_loss:.4f}"
        if val_metrics:
            message += (
                f" val_pair_mse={val_metrics.get('pair_mse', float('nan')):.4f}"
                f" val_pair_mae={val_metrics.get('pair_mae', float('nan')):.4f}"
                f" val_dist_mmd={val_metrics.get('dist_mmd', float('nan')):.4f}"
            )
        print(message)

        if wandb_run is not None:
            wandb.log({"epoch": epoch_idx + 1, **epoch_metrics}, step=global_step)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Parameters: {num_parameters}")

    if wandb_run is not None:
        wandb_run.summary["num_parameters"] = num_parameters
        wandb.finish()


if __name__ == "__main__":
    main()
