"""Compact single-file pertflow training and evaluation code for ablation studies."""

import argparse
import math
import os
import json
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module
from safetensors.torch import load_file, save_file
from transformers.modeling_outputs import SequenceClassifierOutput
import wandb

from pertflow.model.flow import FlowHead, SelfFlow, RectifiedFlow
from pertflow.model.encoder import ValueEncoder, ConditionEncoder, EncoderBlock
from pertflow.model.downstream import ExprDecoder

from pertflow.utils.dataset import PerturbationPool, ConditionKey, build_train_val_pools, load_conditioned_adata, sample_condition_key, sample_indices
from pertflow.utils.ot import compute_transport_plan, sample_hard_pairs
from pertflow.utils.metrics import compute_pair_metrics
from pertflow.utils.custom_utils import choose_device, autocast_context, default



LossBreakdown = namedtuple("LossBreakdown", ["total", "flow_loss", "repr_loss"])


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
                self.head = RectifiedFlow(FlowHead(dim), times_cond_kwarg="times")
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
        os.getenv("PERT_CACHE_PATH", root / "../data/esmc_pert_embeddings.pt")
    )

    parser = argparse.ArgumentParser(
        description="Train pertflow with OT-matched control-to-perturbed batches."
    )
    parser.add_argument("--data-path", type=Path, default=root / "../data/train_val.h5ad")
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
        output_dir = Path(f"training/pert-model-ot-{suffix}")
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
