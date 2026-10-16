
import numpy as np

from engine.tensor import Tensor


def softmax_cross_entropy(logits, targets):
    """
    Numerically stable softmax cross-entropy.

    logits: Tensor of shape (batch_size, num_classes)
    targets: integer class labels of shape (batch_size,)

    Returns:
        Scalar Tensor containing mean cross-entropy loss.
    """

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    if logits.data.ndim != 2:
        raise ValueError(
            "logits must have shape (batch_size, num_classes)"
        )

    targets = np.asarray(targets)

    batch_size, num_classes = logits.data.shape

    if batch_size == 0 or num_classes == 0:
        raise ValueError("Logits cannot have empty dimensions")

    if targets.shape != (batch_size,):
        raise ValueError("Targets must have shape (batch_size,)")

    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("Targets must contain integer class labels")

    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("Target class index out of range")

    if not np.all(np.isfinite(logits.data)):
        raise ValueError("Logits must be finite")

    # Step 1: Subtract the maximum for numerical stability.
    shifted = logits.data - np.max(
        logits.data,
        axis=1,
        keepdims=True
    )

    # Step 2: Calculate normalized probabilities.
    exp_shifted = np.exp(shifted)

    probabilities = exp_shifted / np.sum(
        exp_shifted,
        axis=1,
        keepdims=True
    )

    # Step 3: Calculate log-sum-exp.
    log_sum_exp = np.log(
        np.sum(exp_shifted, axis=1)
    )

    # Step 4: Negative log-likelihood per example.
    correct_logits = shifted[
        np.arange(batch_size),
        targets
    ]

    losses = log_sum_exp - correct_logits

    # Step 5: Average loss across the batch.
    mean_loss = np.mean(losses)

    result = Tensor(
        mean_loss,
        requires_grad=logits.requires_grad,
        parents=(logits,),
        operation="softmax_cross_entropy"
    )

    # Step 6: Custom backward function.
    def _backward():
        if result.grad is None:
            return

        if logits.requires_grad:
            if logits.grad is None:
                logits.grad = np.zeros_like(logits.data)

            gradient = probabilities.copy()

            gradient[
                np.arange(batch_size),
                targets
            ] -= 1.0

            gradient /= batch_size

            logits.grad += result.grad * gradient

    result._backward = _backward

    return result
