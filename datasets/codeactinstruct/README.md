# CodeActInstruct Dataset

## Description

CodeActInstruct is an instruction-tuning dataset consisting of multi-turn interactions designed to train LLM agents that use executable Python code as their unified action space. The dataset was created to support the CodeAct framework, which consolidates LLM agents' actions into executable Python code that can be dynamically revised and executed through multi-turn interactions.

The dataset focuses on:
- Code generation and tool use with execution capabilities
- Multi-turn interactions where agents execute Python code to perform tasks
- Dynamic action revision based on execution results and new observations
- Integration of code execution with natural language collaboration

CodeActInstruct enables training of agents that can perform sophisticated tasks using existing Python libraries, autonomously self-debug, and interact with environments through interpretable code execution.

## Paper Citation

```bibtex
@inproceedings{wang2024executable,
  title={Executable code actions elicit better llm agents},
  author={Wang, Xingyao and Chen, Yangyi and Yuan, Lifan and Zhang, Yizhe and Li, Yunzhu and Peng, Hao and Ji, Heng},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

## Dataset Information

**Source URL**: https://huggingface.co/datasets/xingyaoww/code-act

**License**: Apache 2.0
