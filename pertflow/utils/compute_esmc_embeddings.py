import argparse
import gc
import time
from pathlib import Path
from urllib.parse import quote

import anndata as ad
import numpy as np
import requests
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


ENSEMBL_BASE_URL = "https://rest.ensembl.org"


def parse_args():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Pre-compute ESMC embeddings for perturbation genes."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "train_val.h5ad",
        help="h5ad file whose obs['condition'] column defines perturbation genes.",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Optional explicit list of gene symbols. Overrides --data-path parsing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "esmc_pert_embeddings.pt",
        help="Output .pt cache path.",
    )
    parser.add_argument(
        "--model-name",
        default="esmc_300m",
        help="ESMC checkpoint name to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for Ensembl REST calls.",
    )
    return parser.parse_args()


def parse_condition_value(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    condition = str(value).strip()
    if not condition or condition == "ctrl":
        return None

    if condition.startswith("ctrl+"):
        gene = condition.split("+", 1)[1].strip()
        return gene or None

    raise ValueError(f"Unsupported condition value: {condition!r}")


def genes_from_h5ad(data_path: Path) -> list[str]:
    adata = ad.read_h5ad(data_path, backed="r")
    try:
        if "condition" not in adata.obs:
            raise KeyError(f"{data_path} is missing required obs['condition'] metadata")

        pert_genes = {
            gene
            for gene in (parse_condition_value(value) for value in adata.obs["condition"])
            if gene is not None
        }
        if not pert_genes:
            raise ValueError(f"No perturbation genes found in {data_path}")
        return sorted(pert_genes)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def choose_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def strip_version(identifier: str | None) -> str | None:
    if not identifier:
        return None
    return identifier.split(".", 1)[0]


def request_json(session: requests.Session, endpoint: str, timeout: float, params=None):
    url = f"{ENSEMBL_BASE_URL}{endpoint}"

    for attempt in range(3):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
            return response.json()
        time.sleep(2**attempt)

    response.raise_for_status()
    return response.json()


def resolve_sequence_identifier(gene_lookup: dict) -> tuple[str, str]:
    transcripts = gene_lookup.get("Transcript") or []
    canonical_transcript = strip_version(gene_lookup.get("canonical_transcript"))

    if canonical_transcript:
        for transcript in transcripts:
            transcript_id = strip_version(transcript.get("id"))
            if transcript_id != canonical_transcript:
                continue
            translation = transcript.get("Translation") or {}
            translation_id = strip_version(translation.get("id"))
            if translation_id:
                return "translation", translation_id
            return "transcript", canonical_transcript

        return "transcript", canonical_transcript

    for transcript in transcripts:
        if transcript.get("biotype") != "protein_coding":
            continue
        translation = transcript.get("Translation") or {}
        translation_id = strip_version(translation.get("id"))
        if translation_id:
            return "translation", translation_id
        transcript_id = strip_version(transcript.get("id"))
        if transcript_id:
            return "transcript", transcript_id

    raise ValueError(
        f"Could not find a protein-coding canonical transcript for {gene_lookup.get('display_name')}"
    )


def fetch_protein_sequence(session: requests.Session, gene_symbol: str, timeout: float) -> str:
    lookup = request_json(
        session,
        f"/lookup/symbol/homo_sapiens/{quote(gene_symbol)}",
        timeout=timeout,
        params={"expand": 1},
    )
    identifier_type, identifier = resolve_sequence_identifier(lookup)

    sequence_params = {"type": "protein"} if identifier_type == "transcript" else None
    sequence_payload = request_json(
        session,
        f"/sequence/id/{identifier}",
        timeout=timeout,
        params=sequence_params,
    )

    sequence = sequence_payload.get("seq", "").replace(" ", "").replace("\n", "").strip()
    if not sequence:
        raise ValueError(f"Ensembl returned an empty protein sequence for {gene_symbol}")
    return sequence


def compute_embedding_matrix(
    gene_symbols: list[str],
    model_name: str,
    device: str,
    timeout: float,
) -> tuple[dict[str, int], np.ndarray]:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_symbols, start=1)}
    embeddings = np.zeros((len(gene_symbols) + 1, 960), dtype=np.float32)

    try:
        print(f"Loading ESMC model {model_name} on {device}")
        client = ESMC.from_pretrained(model_name).to(device)
        client.eval()

        for gene in gene_symbols:
            print(f"Fetching protein sequence for {gene}")
            sequence = fetch_protein_sequence(session, gene, timeout=timeout)
            protein = ESMProtein(sequence=sequence)

            with torch.inference_mode():
                protein_tensor = client.encode(protein)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                pooled = logits_output.embeddings.mean(dim=1).squeeze(0)

            embeddings[gene_to_idx[gene]] = pooled.detach().cpu().numpy().astype(
                np.float32, copy=False
            )

            del protein_tensor, logits_output, pooled
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        session.close()
        if "client" in locals():
            del client
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return gene_to_idx, embeddings


def main():
    args = parse_args()

    if args.genes:
        gene_symbols = sorted({gene.strip() for gene in args.genes if gene.strip()})
    else:
        gene_symbols = genes_from_h5ad(args.data_path)

    if not gene_symbols:
        raise ValueError("No perturbation genes were provided")

    device = choose_device(args.device)
    print(f"Computing ESMC embeddings for {len(gene_symbols)} genes")

    gene_to_idx, embeddings = compute_embedding_matrix(
        gene_symbols=gene_symbols,
        model_name=args.model_name,
        device=device,
        timeout=args.request_timeout,
    )

    payload = {
        "gene_to_idx": gene_to_idx,
        "embeddings": embeddings,
        "model_name": args.model_name,
        "genes": gene_symbols,
        "source_data_path": str(args.data_path),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved perturbation embedding cache to {args.output}")


if __name__ == "__main__":
    main()
