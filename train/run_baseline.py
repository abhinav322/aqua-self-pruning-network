
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy
from nn.model import MLP
from nn.optimizers import Adam
from train.dataset import (
    generate_two_moons,
    train_test_split
)


SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.01


def evaluate(model, X, y):
    logits = model.forward(Tensor(X))

    loss = softmax_cross_entropy(logits, y)

    predictions = np.argmax(
        logits.data,
        axis=1
    )

    accuracy = np.mean(predictions == y)

    return float(loss.data), float(accuracy)


def train():
    X, y = generate_two_moons(
        n_samples=1000,
        noise=0.12,
        seed=SEED
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, seed=SEED
    )

    model = MLP(
        input_dim=2,
        hidden_dim=32,
        output_dim=2,
        seed=SEED
    )

    optimizer = Adam(
        model.parameters(),
        learning_rate=LEARNING_RATE
    )

    rng = np.random.default_rng(SEED)

    history = []

    for epoch in range(1, EPOCHS + 1):

        # Shuffle examples each epoch.
        indices = rng.permutation(len(X_train))

        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Reset gradients from the previous batch.
            optimizer.zero_grad()

            # Forward pass.
            inputs = Tensor(X_batch)

            logits = model.forward(inputs)

            loss = softmax_cross_entropy(
                logits,
                y_batch
            )

            # Backward pass.
            loss.backward()

            # Update parameters.
            optimizer.step()

        # Evaluate after the entire epoch.
        train_loss, train_accuracy = evaluate(
            model, X_train, y_train
        )

        test_loss, test_accuracy = evaluate(
            model, X_test, y_test
        )

        if not np.all(
            np.isfinite([
                train_loss,
                test_loss
            ])
        ):
            raise RuntimeError("Non-finite loss detected")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2%} | "
                f"Test Acc: {test_accuracy:.2%}"
            )

    # Save experimental results.
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    # Plot training and test loss.
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training loss"
    )

    plt.plot(
        epochs,
        [row["test_loss"] for row in history],
        label="Test loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Baseline MLP: Learning Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=160
    )

    plt.close()

    # Plot accuracy.
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Training accuracy"
    )

    plt.plot(
        epochs,
        [row["test_accuracy"] for row in history],
        label="Test accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline MLP: Accuracy Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=160
    )

    plt.close()

    print("\nFinal results:")
    print(f"Training accuracy: {history[-1]['train_accuracy']:.2%}")
    print(f"Test accuracy: {history[-1]['test_accuracy']:.2%}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    train()
