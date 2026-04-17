from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module

class ValueEncoder(Module):
    """Encode either continuous expression values or discrete bins per gene."""

    def __init__(self, nbins, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(max(nbins + 2, 2), embed_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        if torch.is_floating_point(x):
            return self.value_mlp(x.unsqueeze(-1))
        return self.token_embedding(x.long())


class Attention(Module):
    """Multi-head self-attention over the gene dimension."""

    def __init__(
        self,
        dim,
        nheads=4,
        dim_head=32,
        flash=True,
    ):
        super().__init__()

        self.scale = dim_head**-0.5
        self.nheads = nheads
        hidden_dim = dim_head * nheads

        self.flash = flash

        self.norm = torch.nn.modules.normalization.RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.nheads), qkv
        )

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            q = q * self.scale
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class EncoderBlock(Module):
    """Residual attention block used in the expression encoder."""

    def __init__(self, dim, nheads, dim_head, ff_mult=4):
        super().__init__()
        self.attn = Attention(dim, nheads, dim_head)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class ConditionEncoder(nn.Module):
    """
    Combines sample encodings and perturbation embeddings into a single
    global conditioning vector, projected back to the model's hidden dimension.
    """

    def __init__(self, dim: int, pert_dim: int, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.pert_dim = pert_dim

        # MLP to learn non-linear interactions between cell state and perturbation
        self.mlp = nn.Sequential(
            nn.Linear(dim + pert_dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(
        self, sample_repr: torch.Tensor, pert_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample_repr: (b, n_genes, dim) - Gene-level encodings
            pert_repr: (b, pert_dim) - ESM C perturbation embeddings
        Returns:
            cond_vector: (b, n_genes, dim) - Global context broadcasted to all genes
        """
        b, n_genes, _ = sample_repr.shape

        # 1. Pool the sample encoding to a single global vector (b, dim)
        # Using mean pooling as a simple permutation-invariant aggregator
        pooled_sample = sample_repr.mean(dim=1)

        # 2. Concatenate sample and perturbation: (b, dim + pert_dim)
        combined = torch.cat([pooled_sample, pert_repr], dim=-1)

        # 3. Project back to the target hidden dimension: (b, dim)
        global_cond = self.mlp(combined)

        # 4. Broadcast the global cell condition to match sequence length: (b, n_genes, dim)
        cond_vector = repeat(global_cond, "b d -> b n d", n=n_genes)

        return cond_vector
