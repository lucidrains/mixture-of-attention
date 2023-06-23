import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from mixture_of_attention.mixture_of_attention import MixtureOfAutoregressiveAttention

from mixture_of_attention.rotary_emb import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

# classes

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

# main class

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        seq_len,
        local_attn_window_size,
        num_routed_queries,
        num_routed_key_values,
        num_experts,
        cosine_sim_routing = True,
        routed_window_size = None,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_triton = True,
        routed_rotary_emb = True
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.rotary_emb = RotaryEmbedding(dim_head) if routed_rotary_emb else None

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MixtureOfAutoregressiveAttention(
                    dim = dim,
                    local_attn_window_size = local_attn_window_size,
                    routed_window_size = routed_window_size,
                    num_routed_queries = num_routed_queries,
                    num_routed_key_values = num_routed_key_values,
                    cosine_sim_routing = cosine_sim_routing,
                    num_experts = num_experts,
                    dim_head = dim_head,
                    heads = heads,
                    use_triton = use_triton
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(x.shape[-2], device = self.device))

        rotary_emb = None
        if exists(self.rotary_emb):
            rotary_emb = self.rotary_emb(x.shape[1])

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = rotary_emb) + x

            x = ff(x) + x

        return self.to_logits(x)
