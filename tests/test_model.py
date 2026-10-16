
import numpy as np

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy
from nn.model import MLP


def test_model_forward_shape():
    model = MLP(
        input_dim=2,
        hidden_dim=32,
        output_dim=2,
        seed=42
    )

    x = Tensor(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]
    )

    logits = model.forward(x)

    assert logits.data.shape == (3, 2)


def test_model_backward_gradients():
    model = MLP(seed=42)

    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    targets = np.array([0, 1, 0])

    logits = model.forward(x)

    loss = softmax_cross_entropy(
        logits,
        targets
    )

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None

        assert parameter.grad.shape == parameter.data.shape

        assert np.all(np.isfinite(parameter.grad))


def test_model_reproducible_initialization():
    model1 = MLP(seed=42)
    model2 = MLP(seed=42)

    for p1, p2 in zip(
        model1.parameters(),
        model2.parameters()
    ):
        np.testing.assert_array_equal(
            p1.data,
            p2.data
        )
