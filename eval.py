# ADD DISTRIBUTIONAL METRICS 
# ADD  SOMETHING WHERE IT GENERATES X CELLS AND CALCULATES MMD/KL OR SOMETHING LIKE THAT BETWEEN THE TWO DISTRIBUTIONS
import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from pertflow import PerturbationModel, SCRNADataset, tokenizer


def find_latest_checkpoint(run_dir: Path) -> Path:
    """Pick the most recent `checkpoint-*` subdirectory by step number."""
    candidates = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found in {run_dir}")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="pert-model-flow",
        help="Trainer output_dir containing config.json and checkpoint-*/model.safetensors",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Specific checkpoint dir (e.g. pert-model-9/checkpoint-52). Defaults to latest in --run-dir.",
    )
    ap.add_argument("--data", default="test_ood_context.h5ad")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        # derive run_dir from the checkpoint's parent so config.json matches weights
        run_dir = ckpt_dir.parent
    else:
        run_dir = Path(args.run_dir)
        ckpt_dir = find_latest_checkpoint(run_dir)
    config_path = run_dir / "config.json"

    print(f"Loading config from {config_path}")
    print(f"Loading weights from {ckpt_dir / 'model.safetensors'}")

    with open(config_path) as f:
        cfg = json.load(f)

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = PerturbationModel(
        cfg["d_model"],
        cfg["nheads"],
        cfg["dim_head"],
        cfg["num_layers"],
        cfg["nbins"],
        head_type=cfg.get("head_type", "flow"),
    )
    state_dict = load_file(str(ckpt_dir / "model.safetensors"))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    ad_obj = sc.read_h5ad(args.data)
    raw = ad_obj.X.toarray()
    tokens = tokenizer(raw, cfg["nbins"])
    ds = SCRNADataset(tokens=tokens, targets=raw)
    dl = DataLoader(ds, batch_size=args.batch_size)

    all_preds = []
    all_targets = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids)
            all_preds.append(out.logits.float().cpu().numpy())
            all_targets.append(labels.float().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Global metrics
    print(f"MSE:  {mean_squared_error(targets, preds):.6f}")
    print(f"MAE:  {mean_absolute_error(targets, preds):.6f}")
    print(f"R²:   {r2_score(targets, preds):.6f}")

    # Per-cell correlation (how well each cell's profile is reconstructed)
    cell_pearson = [pearsonr(targets[i], preds[i])[0] for i in range(len(targets))]
    print(
        f"\nPer-cell Pearson r:  {np.nanmean(cell_pearson):.4f} ± {np.nanstd(cell_pearson):.4f}"
    )

    # Per-gene correlation (how well each gene's variation across cells is captured)
    gene_pearson = []
    for g in range(targets.shape[1]):
        if targets[:, g].std() > 0:
            gene_pearson.append(pearsonr(targets[:, g], preds[:, g])[0])
    print(
        f"Per-gene Pearson r:  {np.nanmean(gene_pearson):.4f} ± {np.nanstd(gene_pearson):.4f}"
    )
    print(
        f"  Genes with r > 0.6: {sum(1 for r in gene_pearson if r > 0.6)}/{len(gene_pearson)}"
    )

    # Fraction of variance unexplained per gene
    gene_fvu = []
    for g in range(targets.shape[1]):
        var = targets[:, g].var()
        if var > 0:
            gene_fvu.append(((targets[:, g] - preds[:, g]) ** 2).mean() / var)
    print(f"Per-gene FVU (median): {np.nanmedian(gene_fvu):.4f}")


if __name__ == "__main__":
    main()
