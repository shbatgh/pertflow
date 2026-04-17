from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, namedtuple

import torch
import scanpy as sc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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

def tokenizer(data, nbins):
    """Discretize continuous expression values into evenly spaced bins."""
    bins = np.linspace(data.min(), data.max(), nbins)
    binned = np.digitize(data, bins)

    return binned.astype(int)

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
