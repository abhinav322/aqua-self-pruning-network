
import numpy as np
import pytest

from nn.model import MLP
from nn.optimizers import Adam
from prune.controller import GlobalPruner


def make_pruner():
    model = MLP(
        input_dim=2,
        hidden_dim=4,
        output_dim=2,
        seed=42
    )

    optimizer = Adam(model.parameters())

    pruner = GlobalPruner(
        [model.layer1, model.layer2],
        optimizer
    )

    return model, optimizer, pruner


def make_scores(model):
    # Both layers have 8 weights.
    # Scores are distinct to make ordering predictable.
    return [
        np.arange(1, 9, dtype=float).reshape(2, 4),
        np.arange(9, 17, dtype=float).reshape(4, 2)
    ]


def test_initial_sparsity():
    model, optimizer, pruner = make_pruner()

    assert pruner.total_weights == 16
    assert pruner.achieved_sparsity() == 0.0


def test_global_pruning():
    model, optimizer, pruner = make_pruner()

    achieved = pruner.prune_to(
        0.5,
        make_scores(model)
    )

    assert achieved == 0.5

    # All eight smallest scores belong to layer 1.
    assert np.count_nonzero(model.layer1.mask) == 0
    assert np.count_nonzero(model.layer2.mask) == 8


def test_gradual_pruning_does_not_overshoot():
    model, optimizer, pruner = make_pruner()

    scores = make_scores(model)

    pruner.prune_to(0.25, scores)

    assert pruner.achieved_sparsity() == 0.25

    pruner.prune_to(0.5, scores)

    assert pruner.achieved_sparsity() == 0.5


def test_pruning_never_regrows():
    model, optimizer, pruner = make_pruner()

    scores = make_scores(model)

    pruner.prune_to(0.75, scores)

    before = [
        layer.mask.copy()
        for layer in [model.layer1, model.layer2]
    ]

    pruner.prune_to(0.25, scores)

    for layer, original in zip(
        [model.layer1, model.layer2],
        before
    ):
        np.testing.assert_array_equal(
            layer.mask,
            original
        )


def test_optimizer_masks_are_synchronized():
    model, optimizer, pruner = make_pruner()

    pruner.prune_to(
        0.5,
        make_scores(model)
    )

    for layer, index in zip(
        pruner.layers,
        pruner.parameter_indices
    ):
        np.testing.assert_array_equal(
            layer.mask,
            optimizer.masks[index]
        )

        np.testing.assert_array_equal(
            layer.weight.data[~layer.mask],
            0.0
        )


def test_invalid_scores_rejected():
    model, optimizer, pruner = make_pruner()

    with pytest.raises(ValueError):
        pruner.prune_to(
            0.5,
            [np.ones((2, 4))]
        )

    assert pruner.achieved_sparsity() == 0.0
