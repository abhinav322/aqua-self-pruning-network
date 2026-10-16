
import numpy as np
import pytest

from prune.schedule import cubic_sparsity


def test_before_pruning_starts():
    result = cubic_sparsity(
        epoch=10,
        start_epoch=20,
        end_epoch=120,
        final_sparsity=0.9
    )

    assert result == 0.0


def test_at_start_epoch():
    result = cubic_sparsity(
        epoch=20,
        start_epoch=20,
        end_epoch=120,
        final_sparsity=0.9
    )

    assert result == 0.0


def test_midpoint():
    result = cubic_sparsity(
        epoch=70,
        start_epoch=20,
        end_epoch=120,
        final_sparsity=0.9
    )

    # 0.9 * (1 - (1 - 0.5)^3)
    expected = 0.7875

    np.testing.assert_allclose(
        result,
        expected
    )


def test_final_sparsity():
    result = cubic_sparsity(
        epoch=120,
        start_epoch=20,
        end_epoch=120,
        final_sparsity=0.9
    )

    assert result == 0.9


def test_after_pruning_finishes():
    result = cubic_sparsity(
        epoch=150,
        start_epoch=20,
        end_epoch=120,
        final_sparsity=0.9
    )

    assert result == 0.9


def test_schedule_monotonic():
    values = [
        cubic_sparsity(
            epoch=e,
            start_epoch=20,
            end_epoch=120,
            final_sparsity=0.9
        )
        for e in range(151)
    ]

    assert np.all(np.diff(values) >= -1e-12)

    assert all(0.0 <= value <= 0.9 for value in values)


def test_invalid_epoch_range():
    with pytest.raises(ValueError):
        cubic_sparsity(
            epoch=50,
            start_epoch=100,
            end_epoch=20,
            final_sparsity=0.9
        )
