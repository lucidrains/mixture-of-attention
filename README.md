## Mixture-of-Attention (wip)

Some personal experiments around routing tokens to different autoregressive attention, akin to mixture-of-experts

Learned from researcher friend that this has been tried in Switch Transformers unsuccessfully, but I'll give it a go, bringing in some learning points from recent papers like <a href="https://github.com/lucidrains/CoLT5-attention">CoLT5</a>.

The CoLT5 paper basically demonstrates mixture of attention already for 2 experts. This just has to be generalized to greater than 2 experts, and for autoregressive case. Local attention branch would just be a special case of one expert with fixed routing.
