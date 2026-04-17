import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

from pertflow.utils.dataset import PerturbationPool, ConditionKey

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
