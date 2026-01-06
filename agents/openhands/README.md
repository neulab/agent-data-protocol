# OpenHands Agent

## Description

OpenHands is a general-purpose AI agent capable of executing software engineering tasks and web browsing activities. This converter transforms ADP standardized trajectories into OpenHands' supervised fine-tuning (SFT) format using XML-style function calls.

The converter supports both coding tasks (bash, Python) and web browsing scenarios, with configurable API environments and automatic HTML-to-accessibility-tree conversion for web observations.

## Usage

See [DATASETS.md](DATASETS.md) for dataset specific arguments for the converter.

## Agent Information

**Repository**: https://github.com/OpenHands/OpenHands

**Version**: v0.30.0

**Key Features**: Multi-domain support (coding + web + tool call), extensive tool set, MCP format support, etc.

**Citation**:

```bibtex
@inproceedings{wangopenhands,
  title={OpenHands: An Open Platform for AI Software Developers as Generalist Agents},
  author={Wang, Xingyao and Li, Boxuan and Song, Yufan and Xu, Frank F and Tang, Xiangru and Zhuge, Mingchen and Pan, Jiayi and Song, Yueqi and Li, Bowen and Singh, Jaskirat and others},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
