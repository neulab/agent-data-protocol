# SWE-agent

## Description

SWE-agent is a specialized AI agent designed for software engineering tasks, particularly focused on solving GitHub issues and performing repository-level code changes. This converter transforms ADP standardized trajectories into SWE-agent's supervised fine-tuning (SFT) format using XML-style function calls with explicit reasoning blocks.

The agent uses a minimal, focused tool set optimized for software development: bash commands, a file editor, and a submit action. Dataset-specific APIs are automatically wrapped as bash command executions.

## Agent Information

**Repository**: https://github.com/SWE-agent/SWE-agent

**Key Features**: Software engineering focus, minimal tool set (bash, str_replace_editor, submit), explicit `<think>` blocks for reasoning

**Citation**:

```bibtex
@article{yang2024swe,
  title={Swe-agent: Agent-computer interfaces enable automated software engineering},
  author={Yang, John and Jimenez, Carlos E and Wettig, Alexander and Lieret, Kilian and Yao, Shunyu and Narasimhan, Karthik and Press, Ofir},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={50528--50652},
  year={2024}
}
```
