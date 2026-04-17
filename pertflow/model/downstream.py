from torch import nn
from torch.nn import Module

class ExprDecoder(Module):
    """Decode genewise hidden states back into expression values."""

    def __init__(self, embed_dim):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.ffn(x).squeeze(-1)