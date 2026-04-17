import numpy as np
from scipy.stats import pearsonr

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
