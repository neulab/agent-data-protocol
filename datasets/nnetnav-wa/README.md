# NNetNav-wa Dataset

## Description

NNetNav-wa is a dataset containing web browsing trajectories generated through unsupervised interaction with WebArena environments. The dataset was created using NNetNav, a method that generates synthetic demonstrations for training browser agents by retroactively labeling action sequences from an exploration policy.

The dataset focuses on:
- Retroactively labeled WebArena web exploration
- Unsupervised interaction with real websites
- Synthetic demonstrations generated through exploration policies
- Hierarchical decomposition of complex web tasks into simpler sub-tasks

NNetNav-wa exploits the hierarchical structure of language instructions to make exploration more tractable, automatically pruning interaction episodes when intermediate trajectories cannot be annotated with meaningful sub-tasks. The method addresses the challenge of expensive human supervision in browser agent training by providing effective search through the exponentially large space of web exploration.

## Paper Citation

```bibtex
@article{murty2024nnetnav,
  title={Nnetnav: Unsupervised learning of browser agents through environment interaction in the wild},
  author={Murty, Shikhar and Zhu, Hao and Bahdanau, Dzmitry and Manning, Christopher D},
  journal={arXiv preprint arXiv:2410.02907},
  year={2024}
}
```

## Dataset Information

**Source URL**: https://huggingface.co/datasets/stanfordnlp/nnetnav-wa

**License**: Apache 2.0
