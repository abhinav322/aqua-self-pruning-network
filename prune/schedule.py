
import numpy as np


def cubic_sparsity(
    epoch,
    start_epoch,
    end_epoch,
    final_sparsity
):
    """
    Cubic pruning schedule.

    Before start_epoch: zero sparsity.
    Between start and end: cubic sparsity ramp.
    After end_epoch: final target sparsity.
    """

    if not 0.0 <= final_sparsity <= 1.0:
        raise ValueError(
            "Final sparsity must be between 0 and 1"
        )

    if start_epoch < 0:
        raise ValueError(
            "Start epoch cannot be negative"
        )

    if end_epoch <= start_epoch:
        raise ValueError(
            "End epoch must be greater than start epoch"
        )

    if not np.isfinite(epoch):
        raise ValueError("Epoch must be finite")

    if epoch <= start_epoch:
        return 0.0

    if epoch >= end_epoch:
        return float(final_sparsity)

    progress = (
        (epoch - start_epoch)
        / (end_epoch - start_epoch)
    )

    sparsity = final_sparsity * (
        1.0 - (1.0 - progress) ** 3
    )

    return float(sparsity)
