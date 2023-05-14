## Mixture-of-Attention (wip)

Some personal experiments around routing tokens to different autoregressive attention, akin to mixture-of-experts

Learned from researcher friend that this has been tried in Switch Transformers unsuccessfully, but I'll give it a go, bringing in some learning points from recent papers like <a href="https://github.com/lucidrains/CoLT5-attention">CoLT5</a>.

In my opinion, the CoLT5 paper basically demonstrates mixture of attention already for 2 experts. This just has to be generalized to greater than 2 experts, and for autoregressive case. Local attention branch would just be a special case of one expert with fixed routing. If I route only half the tokens, that would lead to a savings of 4x. If I can show even ~4 experts being better than 1 attention, that should be a win.

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
