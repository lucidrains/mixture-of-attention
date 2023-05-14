import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, pack, unpack

from mixture_of_attention.attend import Attend
from colt5_attention import CoordinateDescentRouter

# helpers

def exists(val):
    return val is not None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        groups = 1, # defines number of experts
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.groups = groups
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.attend = Attend(
            dropout = dropout,
            causal = causal,
            flash = flash
        )

        # taking advantage of convolutional groups to process experts in parallel

        self.to_q = nn.Conv1d(dim * groups, dim_inner * groups, 1, bias = False, groups = groups)
        self.to_kv = nn.Conv1d(dim * groups, dim_inner * 2 * groups, 1, bias = False, groups = groups)
        self.to_out = nn.Conv1d(dim_inner * groups, dim * groups, 1, bias = False, groups = groups)

    def forward(
        self,
        x,
        mask = None
    ):
        """
        einops
        b - batch
        g - groups
        n - sequence
        d - feature dimension
        """
        g, h = self.groups, self.heads

        one_expert = x.ndim == 3

        if one_expert:
            assert g == 1
            x = rearrange(x, 'b n d -> b 1 n d')

        assert x.ndim == 4
        assert x.shape[1] == g

        # fold the groups into the feature dimension to be processed in one go by grouped convolutions

        x = rearrange(x, 'b g n d -> b (g d) n')

        # queries, keys, values

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        # split out heads and merge groups into batches

        q, k, v = map(lambda t: rearrange(t, 'b (g h d) n -> (b g) h n d', h = h, g = g), (q, k, v))

        # attention

        out = self.attend(q, k, v, mask = mask)

        # combine heads out

        out = rearrange(out, '(b g) h n d -> b (g h d) n', g = g)

        out = self.to_out(out)

        out = rearrange(out, 'b (g d) n -> b g n d', g = g)

        if one_expert:
            out = rearrange(out, 'b 1 n d -> b n d')

        return out

# class

class MixtureOfAttention(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
