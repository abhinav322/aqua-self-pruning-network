
import numpy as np


class GlobalPruner:
    """
    Globally prune Linear layer weights.

    Biases are excluded.
    Pruning is permanent: no regrowth.
    """

    def __init__(self, layers, optimizer):
        self.layers = list(layers)
        self.optimizer = optimizer

        if not self.layers:
            raise ValueError("At least one layer is required")

        self.parameter_indices = []

        # Match each layer's weight to its Adam parameter.
        for layer in self.layers:
            matches = [
                i for i, parameter in enumerate(
                    optimizer.parameters
                )
                if parameter is layer.weight
            ]

            if len(matches) != 1:
                raise ValueError(
                    "Each weight must appear exactly once "
                    "in the optimizer"
                )

            self.parameter_indices.append(matches[0])

        self.total_weights = sum(
            layer.weight.data.size
            for layer in self.layers
        )

    def achieved_sparsity(self):
        active = sum(
            np.count_nonzero(layer.mask)
            for layer in self.layers
        )

        return 1.0 - active / self.total_weights

    def prune_to(self, target_sparsity, scores):
        """
        Prune globally to the requested sparsity.

        scores: one NumPy score array per Linear layer.
        Smaller scores are removed first.
        """

        if not np.isfinite(target_sparsity):
            raise ValueError("Sparsity must be finite")

        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("Sparsity must be between 0 and 1")

        if len(scores) != len(self.layers):
            raise ValueError("One score array per layer required")

        # Validate everything before modifying masks.
        candidate_scores = []

        for layer, score in zip(self.layers, scores):
            score = np.asarray(score, dtype=np.float64)

            if score.shape != layer.weight.data.shape:
                raise ValueError("Score shape mismatch")

            if np.any(np.isnan(score)):
                raise ValueError("NaN importance scores")

            if not np.all(np.isfinite(score[layer.mask])):
                raise ValueError(
                    "Active scores must be finite"
                )

            candidate_scores.append(score)

        # Number of connections that should be pruned.
        target_removed = int(
            np.floor(
                target_sparsity * self.total_weights
                + 1e-10
            )
        )

        current_removed = self.total_weights - sum(
            np.count_nonzero(layer.mask)
            for layer in self.layers
        )

        # Never reactivate pruned weights.
        additional = max(0, target_removed - current_removed)

        if additional == 0:
            return self.achieved_sparsity()

        # Collect global indices of active connections.
        candidates = []

        for layer_index, (layer, score) in enumerate(
            zip(self.layers, candidate_scores)
        ):
            for flat_index in np.flatnonzero(layer.mask):
                candidates.append((
                    float(score.flat[flat_index]),
                    layer_index,
                    int(flat_index)
                ))

        # Stable, deterministic order for tied scores.
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )

        new_masks = [
            layer.mask.copy()
            for layer in self.layers
        ]

        # Remove the least-important active connections.
        for _, layer_index, flat_index in candidates[:additional]:
            new_masks[layer_index].flat[flat_index] = False

        # Synchronize forward masks and Adam masks.
        for layer, parameter_index, new_mask in zip(
            self.layers,
            self.parameter_indices,
            new_masks
        ):
            layer.set_mask(new_mask)

            self.optimizer.set_mask(
                parameter_index,
                new_mask
            )

        return self.achieved_sparsity()
