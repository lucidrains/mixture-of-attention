## Mixture-of-Attention

Some personal experiments around routing tokens to different autoregressive attention, akin to mixture-of-experts

Learned from researcher friend that this has been tried in Switch Transformers unsuccessfully, but I'll give it a go, bringing in some learning points from recent papers like <a href="https://github.com/lucidrains/CoLT5-attention">CoLT5</a>.

In my opinion, the CoLT5 paper basically demonstrates mixture of attention already for 2 experts. This just has to be generalized to greater than 2 experts, and for autoregressive case. Local attention branch would just be a special case of one expert with fixed routing. If I route only half the tokens, that would lead to a savings of 4x. If I can show even ~4 experts being better than 1 attention, that should be a win.

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://github.com/arogozhnikov/einops">einops</a> for making tensor manipulation fun and easy

## Install

```bash
$ pip install mixture-of-attention
```

## Usage

```python
import torch
from mixture_of_attention import MixtureOfAttention

mixture_of_attn = MixtureOfAttention(
    dim = 512,
    dim_context = 256,
    num_routed_queries = 16,
    num_routed_key_values = 16,
    num_experts = 2,
    dim_head = 64,
    heads = 8
)

x = torch.randn(1, 1024, 512)
mask = torch.ones((1, 1024)).bool()

context = torch.randn(1, 512, 256)
context_mask = torch.ones((1, 512)).bool()

mixture_of_attn(x, context = context, mask = mask) # (1, 1024, 512)
```

Autoregressive flavor

```python
import torch
from mixture_of_attention import MixtureOfAutoregressiveAttention

mixture_of_attn = MixtureOfAutoregressiveAttention(
    dim = 512,
    local_attn_window_size = 64,       # local attention window size
    routed_window_size = None,         # will be set to the same as local_attn_window_size if None. ideally less than or equal to local attention window size for full receptive field
    num_routed_queries = 12,
    num_routed_key_values = 12,
    num_experts = 2,
    dim_head = 64,
    heads = 8
)

x = torch.randn(1, 1023, 512)

out = mixture_of_attn(x) # (1, 1023, 512)
```

## Todo

- [x] allow for local attention to be automatically included, either for grouped attention, or use `LocalMHA` from `local-attention` repository in parallel, weighted properly
- [x] make it work for autoregressive

- [ ] try dynamic routing tokens, using projection of masked mean-pooled queries

## Citations

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Wright2015CoordinateDA,
    title   = {Coordinate descent algorithms},
    author  = {Stephen J. Wright},
    journal = {Mathematical Programming},
    year    = {2015},
    volume  = {151},
    pages   = {3-34}
}
```

```bibtex
@article{Schmitzer2016StabilizedSS,
    title   = {Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems},
    author  = {Bernhard Schmitzer},
    journal = {ArXiv},
    year    = {2016},
    volume  = {abs/1610.06519}
}
```

```bibtex
@inproceedings{rogozhnikov2022einops,
    title   = {Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation},
    author  = {Alex Rogozhnikov},
    booktitle = {International Conference on Learning Representations},
    year    = {2022},
    url     = {https://openreview.net/forum?id=oapKSVM2bcj}
}
```
