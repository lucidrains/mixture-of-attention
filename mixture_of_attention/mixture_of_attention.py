import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

from mixture_of_attention.attend import Attend
from colt5_attention import CoordinateDescentRouter

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
        dim_context = None,
        heads = 8,
        causal = False,
        groups = 1, # defines number of experts
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.groups = groups

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(
            dropout = dropout,
            causal = causal,
            flash = flash
        )

        # null key / value, to protect against a row that is all masked out

        self.null_kv = nn.Parameter(torch.randn(2, groups, heads, 1, dim_head))

        # taking advantage of convolutional groups to process experts in parallel

        self.to_q = nn.Conv1d(dim * groups, dim_inner * groups, 1, bias = False, groups = groups)
        self.to_kv = nn.Conv1d(dim_context * groups, dim_inner * 2 * groups, 1, bias = False, groups = groups)
        self.to_out = nn.Conv1d(dim_inner * groups, dim * groups, 1, bias = False, groups = groups)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        """
        einops
        b - batch
        g - groups
        n - sequence
        d - feature dimension
        """
        b, g, h = x.shape[0], self.groups, self.heads

        one_expert = x.ndim == 3

        if one_expert:
            assert g == 1
            x = rearrange(x, 'b n d -> b 1 n d')

        assert x.ndim == 4
        assert x.shape[1] == g

        # fold the groups into the feature dimension to be processed in one go by grouped convolutions

        x = rearrange(x, 'b g n d -> b (g d) n')

        # handle context for cross attention

        if exists(context):
            context_one_expert = context.ndim == 3

            if context_one_expert:
                assert g == 1
                context = rearrange(context, 'b n d -> b 1 n d')

            assert context.ndim == 4
            assert context.shape[1] == g

            context = rearrange(context, 'b g n d -> b (g d) n')

        context = default(context, x)

        # take care of mask

        if exists(mask):
            if mask.ndim == 2:
                mask = repeat(mask, 'b n -> (b g) n', g = g)
            elif mask.ndim == 3:
                mask = rearrange(mask, 'b g n -> (b g) n')

            mask = F.pad(mask, (1, 0), value = True)

        # queries, keys, values

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))

        # split out heads and merge groups into batches

        q, k, v = map(lambda t: rearrange(t, 'b (g h d) n -> (b g) h n d', h = h, g = g), (q, k, v))

        # concat null key / values, to protect against a row having all masked out elements and save a lot of headache

        nk, nv = map(lambda t: repeat(t, 'g h 1 d -> (b g) h 1 d', b = b), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

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
