
import numpy as np

from nn.layers import Linear


class MLP:
    def __init__(
        self,
        input_dim=2,
        hidden_dim=32,
        output_dim=2,
        seed=42
    ):
        rng = np.random.default_rng(seed)

        self.layer1 = Linear(
            input_dim,
            hidden_dim,
            rng
        )

        self.layer2 = Linear(
            hidden_dim,
            output_dim,
            rng
        )

    def forward(self, x):
        hidden = self.layer1.forward(x).relu()

        logits = self.layer2.forward(hidden)

        return logits

    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
        )
