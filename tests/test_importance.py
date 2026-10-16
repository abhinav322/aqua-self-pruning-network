
import numpy as np
import pytest

from prune.importance import (
    magnitude_importance,
    taylor_importance
)


def test_magnitude_scores():
    weights = np.array([0.8, 0.1, -0.4])
    mask = np.array([1, 1, 1])

    scores = magnitude_importance(weights, mask)

    np.testing.assert_allclose(
        scores,
        [0.8, 0.1, 0.4]
    )


def test_taylor_scores():
    weights = np.array([0.8, 0.1, 0.4])
    gradients = np.array([0.01, 2.0, 0.1])
    mask = np.array([1, 1, 1])

    scores = taylor_importance(
        weights,
        gradients,
        mask
    )

    np.testing.assert_allclose(
        scores,
        [0.008, 0.2, 0.04]
    )


def test_pruned_connections_are_excluded():
    weights = np.array([0.8, 0.1, 0.4])
    gradients = np.array([0.01, 2.0, 0.1])
    mask = np.array([1, 0, 1])

    magnitude = magnitude_importance(weights, mask)
    taylor = taylor_importance(weights, gradients, mask)

    assert np.isinf(magnitude[1])
    assert np.isinf(taylor[1])


def test_different_methods_rank_differently():
    weights = np.array([0.8, 0.1, 0.4])
    gradients = np.array([0.01, 2.0, 0.1])
    mask = np.ones(3, dtype=bool)

    magnitude = magnitude_importance(weights, mask)
    taylor = taylor_importance(weights, gradients, mask)

    assert np.argmin(magnitude) == 1
    assert np.argmin(taylor) == 0


def test_invalid_gradient_shape():
    weights = np.array([0.8, 0.1, 0.4])
    gradients = np.array([0.1, 0.2])
    mask = np.ones(3, dtype=bool)

    with pytest.raises(ValueError):
        taylor_importance(weights, gradients, mask)
