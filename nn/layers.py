
import numpy as np

from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, rng):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive")

        # He initialization.
        scale = np.sqrt(2.0 / in_features)

        weights = rng.normal(
            loc=0.0,
            scale=scale,
            size=(in_features, out_features)
        )

        self.weight = Tensor(
            weights,
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

        # Initially all connections are active.
        self.mask = np.ones_like(
            self.weight.data,
            dtype=bool
        )

    def forward(self, x):
        # The mask is a constant Tensor.
        mask_tensor = Tensor(
            self.mask.astype(np.float64),
            requires_grad=False
        )

        effective_weight = self.weight * mask_tensor

        return (x @ effective_weight) + self.bias

    def set_mask(self, new_mask):
        new_mask = np.asarray(new_mask)

        if new_mask.shape != self.weight.data.shape:
            raise ValueError("Mask shape mismatch")

        if not np.all(
            (new_mask == 0) | (new_mask == 1)
        ):
            raise ValueError("Mask values must be 0 or 1")

        self.mask = new_mask.astype(bool).copy()

    def sparsity(self):
        return 1.0 - np.mean(self.mask)

    def parameters(self):
        return [self.weight, self.bias]
