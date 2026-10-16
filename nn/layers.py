
import numpy as np

from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, rng):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive")

        # He initialization for ReLU-based networks.
        scale = np.sqrt(2.0 / in_features)

        weights = rng.normal(
            loc=0.0,
            scale=scale,
            size=(in_features, out_features)
        )

        biases = np.zeros(out_features)

        self.weight = Tensor(
            weights,
            requires_grad=True
        )

        self.bias = Tensor(
            biases,
            requires_grad=True
        )

    def forward(self, x):
        # x: (batch_size, in_features)
        # W: (in_features, out_features)
        # b: (out_features,)
        return (x @ self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]
