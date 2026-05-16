# CoderForge-Preview Dataset

## Description

CoderForge-Preview is the largest open test-verified coding agent dataset designed for training efficient software engineering agents. The dataset contains agent trajectories solving real-world coding tasks, with all trajectories being test-verified for quality.

Fine-tuning Qwen-3 32B on this dataset boosts SWE-Bench Verified performance from 23.0% to 59.4% pass@1, ranking #1 among open-data and #2 among open-weight models ≤32B parameters.

The dataset focuses on:
- Large-scale agentic data generation from 51K distinct open-source tasks
- Long-horizon, multi-step SFT trajectories
- Test-verified coding agent trajectories
- Data collected using OpenHands agent framework

## Paper Citation

```bibtex
@misc{CoderForge2026,
  title = {CoderForge-Preview: SOTA Open Dataset for Training Efficient Agents},
  author = {Ariyak, Alpay and Zhang, Junda and Wang, Junxiong and Zhu, Shang and Bianchi, Federico and Srivastava, Sanjana and Panda, Ashwinee and Bharti, Siddhant and Xu, Chenfeng and Heo, John and Wu, Xiaoxia Shirley and Zhou, James and Liang, Percy and Song, Leon and Zhang, Ce and Athiwaratkun, Ben and Zhou, Zhongzhu and Wu, Qingyang},
  year = {2026},
  month = feb,
  publisher = {TogetherAI Blog},
  url = {https://www.together.ai/blog/coderforge-preview},
  note = {Project core leads: Alpay Ariyak; Zhongzhu Zhou; Qingyang Wu}
}
```

## Dataset Information

**Source URL**: https://huggingface.co/datasets/togethercomputer/CoderForge-Preview

**License**: Apache-2.0
