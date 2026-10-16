
import numpy as np


class SparseLinear:
    """
    Sparse-aware inference using active connections.

    Stores active input indices and weights for
    each output neuron.
    """

    def __init__(self, weight, bias, mask):
        weight = np.asarray(weight, dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        mask = np.asarray(mask)

        if weight.ndim != 2:
            raise ValueError("Weight must be a 2D matrix")

        if mask.shape != weight.shape:
            raise ValueError("Mask shape mismatch")

        if bias.shape != (weight.shape[1],):
            raise ValueError("Bias shape mismatch")

        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Mask must contain only 0 or 1")

        self.input_dim = weight.shape[0]
        self.output_dim = weight.shape[1]
        self.bias = bias.copy()

        self.indices = []
        self.values = []

        for output_index in range(self.output_dim):
            active_indices = np.flatnonzero(
                mask[:, output_index]
            )

            active_values = weight[
                active_indices,
                output_index
            ].copy()

            self.indices.append(active_indices)
            self.values.append(active_values)

        self.nnz = sum(
            len(values) for values in self.values
        )

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError("Input must be 2D")

        if x.shape[1] != self.input_dim:
            raise ValueError("Input feature count mismatch")

        batch_size = x.shape[0]

        output = np.broadcast_to(
            self.bias,
            (batch_size, self.output_dim)
        ).copy()

        for output_index in range(self.output_dim):
            indices = self.indices[output_index]
            values = self.values[output_index]

            if len(indices) == 0:
                continue

            output[:, output_index] += (
                x[:, indices] @ values
            )

        return output


class SparseMLP:
    """
    Inference-only sparse representation of our MLP.
    """

    def __init__(self, model):
        self.layer1 = SparseLinear(
            model.layer1.weight.data,
            model.layer1.bias.data,
            model.layer1.mask
        )

        self.layer2 = SparseLinear(
            model.layer2.weight.data,
            model.layer2.bias.data,
            model.layer2.mask
        )

    def forward(self, x):
        hidden = self.layer1.forward(x)

        hidden = np.maximum(hidden, 0.0)

        return self.layer2.forward(hidden)

    def active_weights(self):
        return self.layer1.nnz + self.layer2.nnz
