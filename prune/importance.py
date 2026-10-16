
import numpy as np


def magnitude_importance(weights, mask):
    """
    Importance based on absolute weight magnitude.

    Pruned connections receive infinite scores
    so they are never selected for pruning again.
    """
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if weights.shape != mask.shape:
        raise ValueError("Weights and mask shapes must match")

    scores = np.abs(weights)

    return np.where(mask, scores, np.inf)


def taylor_importance(weights, gradients, mask):
    """
    First-order Taylor sensitivity: abs(weight * gradient).

    Only active connections are eligible for pruning.
    """
    weights = np.asarray(weights, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if weights.shape != gradients.shape:
        raise ValueError("Weights and gradients must match")

    if weights.shape != mask.shape:
        raise ValueError("Weights and mask shapes must match")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite")

    if not np.all(np.isfinite(gradients)):
        raise ValueError("Gradients must be finite")

    scores = np.abs(weights * gradients)

    return np.where(mask, scores, np.inf)
