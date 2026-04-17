import argparse
import json
from pathlib import Path

from pertflow import (
    build_pool,
    choose_device,
    evaluate_model,
    load_conditioned_adata,
    load_model,
    resolve_config_relative_path,
)


def find_latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found in {run_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pertflow with OT-matched control-to-perturbed validation batches."
    )
    parser.add_argument("--run-dir", default="pert-model-ot-flow-direct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default="train_val.h5ad")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        run_dir = checkpoint_dir.parent
    else:
        run_dir = Path(args.run_dir)
        checkpoint_dir = find_latest_checkpoint(run_dir)

    config_path = run_dir / "config.json"
    with open(config_path) as handle:
        config = json.load(handle)

    device = choose_device()
    print(f"Using {device} device")
    print(f"Loading config from {config_path}")
    print(f"Loading weights from {checkpoint_dir / 'model.safetensors'}")

    pert_cache_path = resolve_config_relative_path(config["pert_cache_path"], config_path)
    expression, obs, pert_indices, _, pert_embedding_matrix = load_conditioned_adata(
        Path(args.data),
        pert_cache_path,
    )
    pool = build_pool(expression, obs, pert_indices)
    if not pool.condition_keys:
        raise ValueError("Evaluation data contains no valid celltype-restricted perturbation conditions")

    model = load_model(
        config=config,
        checkpoint_path=checkpoint_dir / "model.safetensors",
        device=device,
        pert_embedding_matrix=pert_embedding_matrix,
    )

    aggregate_metrics, per_condition = evaluate_model(
        model=model,
        pool=pool,
        device=device,
        batch_size=args.batch_size,
        steps=args.steps or config.get("flow_steps", 16),
        ot_match_space=config.get("ot_match_space", "latent"),
        ot_solver=config.get("ot_solver", "sinkhorn"),
        ot_reg=config.get("ot_reg", 0.05),
        max_conditions=args.max_conditions,
        seed=args.seed,
    )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_dir),
        "data": str(args.data),
        "aggregate": aggregate_metrics,
        "per_condition": per_condition,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
