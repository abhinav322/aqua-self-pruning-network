import os
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
from prune.controller import GlobalPruner
from prune.schedule import cubic_sparsity
from prune.importance import (
    magnitude_importance,
    taylor_importance,
)
from train.dataset import (
    generate_two_moons,
    train_test_split,
)


SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.01

TARGET_SPARSITY = float(
    os.environ.get("AQUA_TARGET_SPARSITY", "0.75")
)
PRUNE_START = 20
PRUNE_END = 140
PRUNE_EVERY = 5

METHOD = os.environ.get(
    "AQUA_METHOD", "taylor"
).lower()

if METHOD not in {"taylor", "magnitude"}:
    raise ValueError("Method must be taylor or magnitude")


def prepare_data():
    X, y = generate_two_moons(
        n_samples=1000,
        noise=0.12,
        seed=SEED,
    )

    # Reserve the final test set first.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_fraction=0.2, seed=SEED
    )

    # Split the remaining 800 into 640 train,
    # 160 validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_fraction=0.2, seed=SEED + 1
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y):
    logits = model.forward(Tensor(X))

    loss = softmax_cross_entropy(logits, y)

    predictions = np.argmax(logits.data, axis=1)

    accuracy = np.mean(predictions == y)

    return float(loss.data), float(accuracy)


def calculate_importance(model, X_train, y_train, method):
    layers = [model.layer1, model.layer2]

    if method == "magnitude":
        return [
            magnitude_importance(
                layer.weight.data,
                layer.mask
            )
            for layer in layers
        ]

    if method != "taylor":
        raise ValueError("Unknown importance method")

    # Accumulate weighted sensitivity scores.
    totals = [
        np.zeros_like(layer.weight.data)
        for layer in layers
    ]

    total_examples = 0

    for start in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_train[start:start + BATCH_SIZE]
        y_batch = y_train[start:start + BATCH_SIZE]

        # This is a gradient measurement,
        # NOT an optimizer update.
        for parameter in model.parameters():
            parameter.zero_grad()

        logits = model.forward(Tensor(X_batch))

        loss = softmax_cross_entropy(
            logits, y_batch
        )

        loss.backward()

        batch_count = len(X_batch)

        for i, layer in enumerate(layers):
            scores = taylor_importance(
                layer.weight.data,
                layer.weight.grad,
                layer.mask
            )

            # Convert inactive infinite scores to zero
            # before accumulating batch sensitivities.
            active_scores = np.where(
                layer.mask, scores, 0.0
            )

            totals[i] += batch_count * active_scores

        total_examples += batch_count

    if total_examples == 0:
        raise ValueError("Cannot score an empty dataset")

    return [
        np.where(
            layer.mask,
            total / total_examples,
            np.inf
        )
        for layer, total in zip(layers, totals)
    ]


def train():
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = prepare_data()

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

    pruner = GlobalPruner(
        [model.layer1, model.layer2],
        optimizer
    )

    rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(1, EPOCHS + 1):
        indices = rng.permutation(len(X_train))

        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()

            logits = model.forward(Tensor(X_batch))

            loss = softmax_cross_entropy(
                logits, y_batch
            )

            loss.backward()
            optimizer.step()

        # Prune only after all training updates
        # for this epoch are finished.
        should_prune = (
            PRUNE_START < epoch <= PRUNE_END
            and (
                (epoch - PRUNE_START) % PRUNE_EVERY == 0
                or epoch == PRUNE_END
            )
        )

        if should_prune:
            target = cubic_sparsity(
                epoch=epoch,
                start_epoch=PRUNE_START,
                end_epoch=PRUNE_END,
                final_sparsity=TARGET_SPARSITY
            )

            scores = calculate_importance(
                model, X_train, y_train, METHOD
            )

            achieved = pruner.prune_to(target, scores)

            print(
                f"Pruning at epoch {epoch}: "
                f"target={target:.2%}, "
                f"achieved={achieved:.2%}"
            )

        train_loss, train_acc = evaluate(
            model, X_train, y_train
        )

        val_loss, val_acc = evaluate(
            model, X_val, y_val
        )

        metrics = [
            train_loss, train_acc,
            val_loss, val_acc
        ]

        if not np.all(np.isfinite(metrics)):
            raise RuntimeError(
                "Non-finite training metrics detected"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "sparsity": pruner.achieved_sparsity()
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%} | "
                f"Sparsity: {pruner.achieved_sparsity():.2%}"
            )

    # Verify the final sparsity budget.
    expected_removed = int(
        np.floor(
            TARGET_SPARSITY * pruner.total_weights
            + 1e-10
        )
    )

    actual_removed = sum(
        np.count_nonzero(~layer.mask)
        for layer in pruner.layers
    )

    assert actual_removed == expected_removed, (
        f"Expected {expected_removed} removed weights, "
        f"got {actual_removed}"
    )

    # Evaluate the untouched test set only after
    # training and pruning are complete.
    test_loss, test_acc = evaluate(
        model, X_test, y_test
    )

    output_dir = Path(
        f"results/pruned_{METHOD}_{int(TARGET_SPARSITY * 100)}"
    )
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

    # Save actual model weights and binary masks.
    np.savez(
        output_dir / "model.npz",
        w1=model.layer1.weight.data,
        b1=model.layer1.bias.data,
        mask1=model.layer1.mask,
        w2=model.layer2.weight.data,
        b2=model.layer2.bias.data,
        mask2=model.layer2.mask,
    )

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Train loss"
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Taylor Pruning: Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [row["sparsity"] * 100 for row in history]
    )
    plt.xlabel("Epoch")
    plt.ylabel("Weight sparsity (%)")
    plt.title("Achieved Sparsity During Training")
    plt.tight_layout()
    plt.savefig(output_dir / "sparsity_curve.png", dpi=160)
    plt.close()

    print("\nFINAL RESULTS")
    print(f"Method: {METHOD}")
    print(f"Train accuracy: {history[-1]['train_accuracy']:.2%}")
    print(f"Validation accuracy: {history[-1]['val_accuracy']:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Achieved sparsity: {pruner.achieved_sparsity():.2%}")
    print(f"Removed weights: {actual_removed}/{pruner.total_weights}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    train()
