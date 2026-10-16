
import numpy as np

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy
from nn.model import MLP
from nn.optimizers import Adam
from prune.controller import GlobalPruner
from prune.importance import magnitude_importance


def make_system():
    model = MLP(
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        seed=42
    )

    optimizer = Adam(
        model.parameters(),
        learning_rate=0.01
    )

    pruner = GlobalPruner(
        [model.layer1, model.layer2],
        optimizer
    )

    return model, optimizer, pruner


def train_one_step(model, optimizer):
    x = Tensor([
        [0.1, 0.2],
        [1.0, -0.5],
        [-0.8, 0.3],
        [0.7, 0.9]
    ])

    y = np.array([0, 1, 0, 1])

    optimizer.zero_grad()

    logits = model.forward(x)
    loss = softmax_cross_entropy(logits, y)

    loss.backward()
    optimizer.step()

    return float(loss.data)


def prune_half(model, pruner):
    scores = [
        magnitude_importance(
            layer.weight.data,
            layer.mask
        )
        for layer in pruner.layers
    ]

    return pruner.prune_to(0.5, scores)


def test_training_continues_after_pruning():
    model, optimizer, pruner = make_system()

    train_one_step(model, optimizer)

    achieved = prune_half(model, pruner)

    assert achieved == 0.5

    for _ in range(10):
        loss = train_one_step(model, optimizer)
        assert np.isfinite(loss)

    assert pruner.achieved_sparsity() == 0.5


def test_pruned_weights_stay_zero_after_training():
    model, optimizer, pruner = make_system()

    train_one_step(model, optimizer)
    prune_half(model, pruner)

    original_masks = [
        layer.mask.copy()
        for layer in pruner.layers
    ]

    for _ in range(20):
        train_one_step(model, optimizer)

    for layer, original_mask in zip(
        pruner.layers,
        original_masks
    ):
        np.testing.assert_array_equal(
            layer.mask,
            original_mask
        )

        np.testing.assert_array_equal(
            layer.weight.data[~layer.mask],
            0.0
        )


def test_optimizer_state_stays_masked():
    model, optimizer, pruner = make_system()

    train_one_step(model, optimizer)
    prune_half(model, pruner)

    for _ in range(10):
        train_one_step(model, optimizer)

    for layer, index in zip(
        pruner.layers,
        pruner.parameter_indices
    ):
        np.testing.assert_array_equal(
            layer.mask,
            optimizer.masks[index]
        )

        np.testing.assert_array_equal(
            optimizer.m[index][~layer.mask],
            0.0
        )

        np.testing.assert_array_equal(
            optimizer.v[index][~layer.mask],
            0.0
        )


def test_revived_connection_resets_adam_state():
    model, optimizer, pruner = make_system()

    layer = model.layer1
    parameter_index = pruner.parameter_indices[0]
    position = (0, 0)

    # Populate Adam's moments with a known gradient.
    optimizer.zero_grad()
    layer.weight.grad = np.ones_like(layer.weight.data)
    optimizer.step()

    assert optimizer.m[parameter_index][position] != 0.0
    assert optimizer.v[parameter_index][position] != 0.0

    # Prune one connection.
    pruned_mask = layer.mask.copy()
    pruned_mask[position] = False

    layer.set_mask(pruned_mask)
    optimizer.set_mask(parameter_index, pruned_mask)

    assert layer.weight.data[position] == 0.0
    assert optimizer.m[parameter_index][position] == 0.0
    assert optimizer.v[parameter_index][position] == 0.0

    # Even an accidentally supplied gradient must not
    # update an inactive connection.
    optimizer.zero_grad()
    layer.weight.grad[position] = 10.0
    optimizer.step()

    assert layer.weight.data[position] == 0.0
    assert optimizer.m[parameter_index][position] == 0.0
    assert optimizer.v[parameter_index][position] == 0.0

    # Reactivate the connection through the mask APIs.
    revived_mask = layer.mask.copy()
    revived_mask[position] = True

    layer.set_mask(revived_mask)
    optimizer.set_mask(parameter_index, revived_mask)

    # Old momentum must not survive reactivation.
    assert layer.weight.data[position] == 0.0
    assert optimizer.m[parameter_index][position] == 0.0
    assert optimizer.v[parameter_index][position] == 0.0

    # With zero new gradient, the revived weight
    # must remain zero.
    optimizer.zero_grad()
    optimizer.step()

    assert layer.weight.data[position] == 0.0

    # A fresh gradient should allow the revived
    # weight to update again.
    optimizer.zero_grad()
    layer.weight.grad[position] = 1.0
    optimizer.step()

    assert layer.weight.data[position] < 0.0
    assert optimizer.m[parameter_index][position] > 0.0
    assert optimizer.v[parameter_index][position] > 0.0
