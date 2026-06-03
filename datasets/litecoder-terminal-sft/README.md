# LiteCoder Terminal SFT Dataset

## Description

LiteCoder-SFT-Terminal is a supervised fine-tuning dataset of terminal-agent trajectories introduced with LiteCoder-Terminal. The dataset contains command-line problem-solving conversations collected from multiple scaffolds, including Terminus-2, OpenHands, and Claude Code, across coding, scientific/numerical computing, games, and other terminal task categories.

## Dataset Information

**Source URL (Hugging Face)**: https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT

**Source file used**: `litecoder-sft.json`

**License**: MIT

**Size / split used**: 11,255 trajectories in the single published JSON file.

## Schema Mapping

- Raw `human` turns are mapped to `TextObservation`. The first `human` turn is the user task prompt, and later `human` turns are terminal/environment observations.
- Raw `gpt` turns are JSON action batches. Each command object's `keystrokes` is converted to a bash `CodeAction`.
- `task_complete: true` responses add a terminal `MessageAction` containing a finish marker.
- Terminal output is split on prompt-like lines when possible so individual command actions can be paired with adjacent observations.

## Paper Citation

```bibtex
@article{peng2026litecoderterminal,
  title={LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents},
  author={Peng, Xiaoxuan and Zhang, Kaiqi and Lu, Xinyu and Cao, Boxi and Lu, Yaojie and Lin, Hongyu and Han, Xianpei and Sun, Le},
  journal={arXiv preprint arXiv:2605.29559},
  year={2026}
}
```
