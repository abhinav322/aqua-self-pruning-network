
import numpy as np


def generate_two_moons(
    n_samples=1000,
    noise=0.12,
    seed=42
):
    rng = np.random.default_rng(seed)

    n_first = n_samples // 2
    n_second = n_samples - n_first

    # Generate angles for the two semicircles.
    theta1 = rng.uniform(0, np.pi, n_first)
    theta2 = rng.uniform(0, np.pi, n_second)

    # First moon: upper semicircle.
    moon1 = np.column_stack([
        np.cos(theta1),
        np.sin(theta1)
    ])

    # Second moon: shifted lower semicircle.
    moon2 = np.column_stack([
        1.0 - np.cos(theta2),
        0.5 - np.sin(theta2)
    ])

    X = np.vstack([moon1, moon2])

    # Assign class labels.
    y = np.concatenate([
        np.zeros(n_first, dtype=np.int64),
        np.ones(n_second, dtype=np.int64)
    ])

    # Add random noise.
    X += rng.normal(
        0.0,
        noise,
        size=X.shape
    )

    # Randomize example order.
    indices = rng.permutation(n_samples)

    return X[indices], y[indices]


def train_test_split(X, y, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))

    test_size = int(len(X) * test_fraction)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )
