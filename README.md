# PertFlow

PertFlow is research code for **single-cell perturbation prediction**.

The main idea is:
- start from a **control / wildtype** cell expression profile,
- condition on the perturbation with a **frozen ESMC embedding** of the perturbed gene,
- form **control → perturbed** training pairs with **celltype-restricted optimal transport (OT)**,
- predict the perturbed state with either an **MLP** or a **flow-matching** head.

The main implementation lives in **`pertflow/model/pertflow.py`**.

## Main paths

- `pertflow/model/pertflow.py` — `PerturbationModel`, OT-paired batch assembly, training loop, evaluation helpers
- `pertflow/utils/dataset.py` — load `.h5ad`, parse perturbations, build train/val pools
- `pertflow/utils/ot.py` — Sinkhorn transport plans and hard-pair sampling
- `pertflow/model/flow.py` — `RectifiedFlow`, `SelfFlow`, `FlowHead`
- `pertflow/model/encoder.py` — expression encoder and perturbation conditioning
- `pertflow/utils/compute_esmc_embeddings.py` — precompute perturbation embedding cache

> Use concrete module paths like `pertflow.model.pertflow` and `pertflow.utils.dataset`. The package-level exports are still minimal.

## Data assumptions

The active path expects an `.h5ad` with:
- `obs['celltype']`
- either `obs['condition']` or `obs['genotype']`

Supported perturbation labels:
- `ctrl`
- `ctrl+GENE`
- direct gene labels in `genotype`

The current loader converts `adata.X` to a dense `float32` array, so this is not a memory-light pipeline.

## Run the main model

```bash
uv run python -m pertflow.model.pertflow \
  --data-path pertflow/data/train_val.h5ad \
  --pert-cache-path pertflow/data/esmc_pert_embeddings.pt \
  --output-dir training/pert-model-ot-flow-self \
  --head-type flow \
  --flow-mode self \
  --wandb-mode offline
```

Useful ablations:

```bash
# MLP baseline
uv run python -m pertflow.model.pertflow \
  --data-path pertflow/data/train_val.h5ad \
  --pert-cache-path pertflow/data/esmc_pert_embeddings.pt \
  --head-type mlp
```

```bash
# Direct flow matching
uv run python -m pertflow.model.pertflow \
  --data-path pertflow/data/train_val.h5ad \
  --pert-cache-path pertflow/data/esmc_pert_embeddings.pt \
  --head-type flow \
  --flow-mode direct
```

## Example: dataset path

This is the main data-loading path used by training.

```python
from pathlib import Path
from pertflow.utils.dataset import load_conditioned_adata, build_train_val_pools

expression, obs, pert_indices, gene_to_idx, pert_embedding_matrix = load_conditioned_adata(
    Path("pertflow/data/train_val.h5ad"),
    Path("pertflow/data/esmc_pert_embeddings.pt"),
)

train_pool, val_pool = build_train_val_pools(
    expression=expression,
    obs=obs,
    pert_indices=pert_indices,
    val_fraction=0.2,
    seed=42,
)
```

Key objects:
- `expression` — dense cell × gene matrix
- `pert_indices` — integer perturbation IDs
- `pert_embedding_matrix` — cached ESMC embedding table
- `train_pool` / `val_pool` — control and perturbed pools used for OT sampling

## Example: main model

```python
from pertflow.model.pertflow import PerturbationModel

model = PerturbationModel(
    dim=128,
    nheads=8,
    dim_head=16,
    nlayers=3,
    nbins=100,
    head_type="flow",      # or "mlp"
    flow_mode="self",      # or "direct"
    pert_dim=int(pert_embedding_matrix.shape[1]),
    pert_embedding_matrix=pert_embedding_matrix,
)
```

Training-style call:

```python
output = model(
    source_expr=source_batch,
    labels=target_batch,
    pert_idx=pert_batch,
    steps=16,
)
loss = output.loss
```

Inference-style call:

```python
preds = model(
    source_expr=source_batch,
    pert_idx=pert_batch,
    steps=16,
).logits
```

## Example: OT matching

High-level batch assembly used by the trainer:

```python
from pertflow.model.pertflow import build_ot_paired_batch

source_batch, target_batch, pert_batch, transport_plan = build_ot_paired_batch(
    model=model,
    pool=train_pool,
    condition_key=train_pool.condition_keys[0],
    batch_size=64,
    device="cuda",
    rng=rng,
    ot_match_space="latent",    # or "expression"
    ot_solver="sinkhorn",
    ot_reg=0.05,
)
```

Low-level OT utilities:

```python
from pertflow.utils.ot import compute_transport_plan, sample_hard_pairs

plan = compute_transport_plan(source_repr, target_repr, solver="sinkhorn", reg=0.05)
source_idx, target_idx = sample_hard_pairs(plan, num_pairs=64)
```

## Example: flow-matching heads

The flow heads live in `pertflow/model/flow.py`.

```python
from pertflow.model.flow import FlowHead, RectifiedFlow, SelfFlow

vector_field = FlowHead(dim=128)

direct_head = RectifiedFlow(vector_field, times_cond_kwarg="times")
self_head = SelfFlow(FlowHead(dim=128), times_cond_kwarg="times")
```

In `PerturbationModel`, these correspond to:
- `head_type="flow", flow_mode="direct"` → `RectifiedFlow`
- `head_type="flow", flow_mode="self"` → `SelfFlow`

## Metrics

Current evaluation reports:
- `pair_mse`
- `pair_mae`
- `pair_pearson`
- `dist_mmd`
- `pair_cell_mean_mae`

## If only reading a few files

Read in this order:
1. `pertflow/model/pertflow.py`
2. `pertflow/utils/dataset.py`
3. `pertflow/utils/ot.py`
4. `pertflow/model/flow.py`
