# Nemotron Terminal Corpus Dataset

## Description

Terminal-Corpus is a large-scale Supervised Fine-Tuning (SFT) dataset designed to scale the terminal interaction capabilities of Large Language Models (LLMs). Developed by NVIDIA, this dataset was built using the **Terminal-Task-Gen** pipeline, which combines dataset adaptation with synthetic task generation across diverse domains.

The dataset contains approximately 366k high-quality execution trajectories for terminal agents, enabling models to achieve performance that rivals or exceeds much larger frontier models on terminal-related benchmarks.

The dataset includes multiple configurations:
- `skill_based_easy`: Easy skill-based tasks
- `skill_based_medium`: Medium difficulty skill-based tasks
- `skill_based_mixed`: Mixed difficulty skill-based tasks

## Paper Citation

```bibtex
@article{pi2026terminal,
  title={On Data Engineering for Scaling LLM Terminal Capabilities},
  author={Pi, Renjie and Lam, Grace and Shoeybi, Mohammad and Jannaty, Pooya and Catanzaro, Bryan and Ping, Wei},
  journal={arXiv preprint},
  year={2026}
}
```

## Dataset Information

**Source URL (Hugging Face)**: https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus

**License**: Please refer to the original dataset for license information.
