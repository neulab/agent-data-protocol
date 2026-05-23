# CharlieDreemur OpenManus-RL Dataset

## Description

OpenManusRL is a heterogeneous ReAct-style agent trajectory dataset assembled for OpenManus-RL. It combines trajectories from AgentInstruct, Agent-FLAN, and AgentTraj-L/AgentGym, covering tool-augmented dialogues and interactive environments across OS, database, web, knowledge graph, household, and e-commerce domains.

The converter preserves the raw conversation structure and maps the dataset's major action styles into ADP primitives:

- Text-world and ScienceWorld `Action:` lines become `perform_action` API actions with native string kwargs.
- `Action: <tool> with Action Input: ...`, JSON `Tool`/`Param` responses, and Python-dict tool call snippets become direct `ApiAction` calls using normalized Python identifiers and native JSON kwargs. Raw tool names containing punctuation are normalized by replacing non-word characters with `_` (for example, `weather.get_120_hour_forecast_for_weather` becomes `weather_get_120_hour_forecast_for_weather`).
- ReAct `Thought:` / JSON `goal` text is stored in `reasoning_content` instead of being mixed into API kwargs.
- Explicit tool catalogs in the raw prompt populate top-level `available_apis`; tool-catalog-only setup prompts are not duplicated as user observations.
- `finish` and final-action outputs become terminal `MessageAction` events without OpenHands-specific `<finish>` tags.
- User turns following actions are treated as environment observations, with structured observations normalized to canonical JSON strings when possible.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL

**GitHub**: https://github.com/OpenManus/OpenManus-RL

**License**: Apache 2.0 for OpenManusRL. The dataset card notes that source datasets include AgentInstruct (CC-BY-NC-4.0) and Agent-FLAN (Apache 2.0).

**Split used**: `train` from the default configuration.

**Size**: 48,927 trajectories.

## Citation

```bibtex
@misc{zeng2023agenttuning,
  title={AgentTuning: Enabling Generalized Agent Abilities for LLMs},
  author={Aohan Zeng and Mingdao Liu and Rui Lu and Bowen Wang and Xiao Liu and Yuxiao Dong and Jie Tang},
  year={2023},
  eprint={2310.12823},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{chen2024agent,
  title={Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models},
  author={Chen, Zehui and Liu, Kuikun and Wang, Qiuchen and Zhang, Wenwei and Liu, Jiangning and Lin, Dahua and Chen, Kai and Zhao, Feng},
  journal={arXiv preprint arXiv:2403.12881},
  year={2024}
}

@misc{xi2024agentgym,
  title={AgentGym: Evolving Large Language Model-based Agents across Diverse Environments},
  author={Zhiheng Xi and Yiwen Ding and Wenxiang Chen and Boyang Hong and Honglin Guo and Junzhe Wang and Dingwen Yang and Chenyang Liao and Xin Guo and Wei He and Songyang Gao and Lu Chen and Rui Zheng and Yicheng Zou and Tao Gui and Qi Zhang and Xipeng Qiu and Xuanjing Huang and Zuxuan Wu and Yu-Gang Jiang},
  year={2024},
  eprint={2406.04151},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
