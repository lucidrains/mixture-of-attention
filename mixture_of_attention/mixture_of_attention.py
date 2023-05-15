import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, reduce, pack, unpack

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
        mask = None,
        queries_scale = None,
        keys_scale = None,
        values_scale = None
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

        q, k, v = map(lambda t: rearrange(t, 'b (g h d) n -> b g h n d', h = h, g = g), (q, k, v))

        # give gradients to routed keys / values via normalized scores from the router, if passed in

        if exists(queries_scale):
            q = q * queries_scale

        if exists(keys_scale):
            k = k * keys_scale

        if exists(values_scale):
            v = v * values_scale

        # merge group into batch

        q, k, v = map(lambda t: rearrange(t, 'b g ... -> (b g) ...'), (q, k, v))

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

class MixtureOfAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_routed_queries,
        num_routed_key_values,
        dim_context = None,
        num_experts = 2,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_triton = True,
        flash_attn = True,
        **kwargs
    ):
        super().__init__()
        dim_context = default(dim_context, dim)
        self.num_routed_queries = num_routed_queries
        self.num_routed_key_values = num_routed_key_values

        self.null_routed_token = nn.Parameter(torch.randn(1, 1, dim))

        self.query_router = CoordinateDescentRouter(
            dim,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        self.key_value_router = CoordinateDescentRouter(
            dim_context,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        self.attn = Attention(
            dim = dim,
            dim_context = dim_context,
            dim_head = dim_head,
            heads = heads,
            groups = num_experts,
            dropout = dropout,
            flash = flash_attn
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        num_routed_queries = None,
        num_routed_key_values = None
    ):
        num_routed_queries = default(num_routed_queries, self.num_routed_queries)
        num_routed_key_values = default(num_routed_key_values, self.num_routed_key_values)

        if not exists(context):
            # self attention if context and context mask not passed in
            context = x
            context_mask = mask

        num_queries = x.shape[-2]
        num_key_values = context.shape[-2]

        need_route_queries = num_routed_queries < num_queries
        need_route_key_values = num_routed_key_values < num_key_values

        query_indices, query_scores, queries, _ = self.query_router(x, mask = mask, num_tokens = num_routed_queries)
        query_scores = rearrange(query_scores, 'b g n -> b g n 1')

        _, key_value_scores, key_values, key_value_mask = self.key_value_router(context, mask = context_mask, num_tokens = num_routed_key_values)
        key_value_scores = rearrange(key_value_scores, 'b g n -> b g 1 n 1')
        
        attn_out = self.attn(
            queries,
            context = key_values,
            mask = key_value_mask,
            values_scale = key_value_scores,
        )

        attn_out = attn_out * query_scores

        need_route_queries = exists(query_indices)

        if not need_route_queries:
            return reduce(attn_out, 'b e n d -> b n d', 'mean')

        out = torch.zeros_like(x)
        counts = torch.zeros(x.shape[:-1], device = x.device)

        query_indices = rearrange(query_indices, 'b g n -> b (g n)')
        attn_out = rearrange(attn_out, 'b g n d -> b (g n) d')

        expanded_query_indices = repeat(query_indices, 'b n -> b n d', d = x.shape[-1])

        attn_out_summed = out.scatter_add(1, expanded_query_indices, attn_out)

        counts = counts.scatter_add(1, query_indices, torch.ones(attn_out.shape[:-1], device = self.device))
        counts = rearrange(counts, '... -> ... 1')

        scatter_meaned = attn_out_summed / counts.clamp(min = 1e-5)

        # for the positions that were not routed, use a learned routing token instead of just 0s

        out = torch.where(
            counts > 0,
            scatter_meaned,
            self.null_routed_token
        )

        return out
