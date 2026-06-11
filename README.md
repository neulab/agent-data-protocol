# Agent Data Protocol (ADP)

A standardized protocol for collecting, processing, and converting agent training data from diverse sources into unified formats suitable for supervised fine-tuning (SFT).

## Recent Release
- Discuss the paper and find linked artifacts on the Hugging Face Paper page: [https://huggingface.co/papers/2510.24702](https://huggingface.co/papers/2510.24702)

- Check out our arXiv preprint: [https://arxiv.org/abs/2510.24702](https://arxiv.org/abs/2510.24702)

- Try a demo of data conversion on our Project Website: [https://www.agentdataprotocol.com/](https://www.agentdataprotocol.com/)

- Download ADP data from the Hugging Face collection: [https://huggingface.co/collections/neulab/agent-data-protocol](https://huggingface.co/collections/neulab/agent-data-protocol)

- Explore the combined ADP dataset repository on the Hub: [https://huggingface.co/datasets/neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection)

## Overview

The Agent Data Protocol provides a systematic approach to handle agent training data across different domains, environments, and agent architectures. It standardizes the representation of agent trajectories, actions, and observations, enabling seamless conversion between raw datasets and agent-specific training formats.

### Key Features

- **Standardized Schema**: Unified representation for agent actions and observations across different domains
- **Multi-Agent Support**: Convert data for different agent architectures (OpenHands v0, SWE-agent, AgentLab, etc.)
- **Type Safety**: Pydantic-based validation ensures data integrity throughout the pipeline
- **Extensible**: Easy to add new datasets and agent implementations
- **Quality Control**: Built-in validation and testing framework

## Quick Start

### Installation

```bash
git clone https://github.com/neulab/agent-data-protocol.git
cd agent-data-protocol
pip install -r requirements.txt
```

### Load ADP Data from Hugging Face

The ADP dataset collection is linked from the [Hugging Face Paper page](https://huggingface.co/papers/2510.24702) and is available as a Hub dataset at [`neulab/agent-data-collection`](https://huggingface.co/datasets/neulab/agent-data-collection). You can load individual dataset configurations and splits directly with [`datasets`](https://huggingface.co/docs/datasets):

```python
from datasets import load_dataset

# Load one ADP dataset configuration and split from the Hub
dataset = load_dataset("neulab/agent-data-collection", "swe-smith", split="std")

# Load agent-specific SFT data
sft_dataset = load_dataset("neulab/agent-data-collection", "swe-smith", split="sft_openhands")
```

The repository-local fixture and converter name for this agent format is `openhands_v0`, but the published Hugging Face split remains `sft_openhands` unless a coordinated dataset migration says otherwise.

### Basic Usage

To regenerate data locally for a specific dataset and agent, follow this pattern:

```bash
# Set your dataset name
export MY_DATASET=swe-smith
mkdir -p datasets/$MY_DATASET/full_sft

# Step 1: Extract raw data
echo "Extracting raw data..."
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl

# Step 2: Convert to ATIF and ATIF std formats
export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Converting to ATIF format..."
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_atif.py > datasets/$MY_DATASET/full_atif.jsonl
echo "Normalizing ATIF tool calls..."
cat datasets/$MY_DATASET/full_atif.jsonl | python datasets/$MY_DATASET/atif_to_std.py > datasets/$MY_DATASET/full_std.jsonl

# Step 3: Convert to agent-specific SFT format
echo "Converting to SFT format..."

# OpenHands v0 consumes normalized ATIF; there are dataset specific arguments to pass in
export MY_AGENT=openhands_v0
cat datasets/$MY_DATASET/full_std.jsonl | python agents/$MY_AGENT/std_to_sft.py --is_web=no --api_env=execute_bash > datasets/$MY_DATASET/full_sft/full_sft_$MY_AGENT.jsonl

# SWE-agent consumes normalized ATIF std records
export MY_AGENT=sweagent
cat datasets/$MY_DATASET/full_std.jsonl | python agents/$MY_AGENT/std_to_sft.py > datasets/$MY_DATASET/full_sft/full_sft_$MY_AGENT.jsonl
```

### Available Datasets

The repository currently supports datasets from various domains (we welcome more contributions!):

- **Coding**: `code_feedback`, `codeactinstruct`, `nemotron_terminal_corpus`
- **Software Engineering**: `swe-smith`, `swe-gym_openhands_sampled_trajectories`, `nebius_SWE-agent-trajectories`, `logicstar_swe-star`, `mini-coder`
- **Web Browsing**: `mind2web`, `nnetnav-live`, `nnetnav-wa`, `go-browse-wa`, `synatra`
- **Multi-domain**: `agenttuning_*`, `CharlieDreemur_OpenManus-RL`, `orca_agentinstruct`, `openhands`, `toucan_1_5m`

### Supported Agents

- **[OpenHands v0](https://github.com/OpenHands/OpenHands)**: General-purpose coding and web browsing agent
- **[SWE-agent](https://github.com/SWE-agent/SWE-agent)**: Software engineering focused agent
- **[AgentLab](https://github.com/ServiceNow/AgentLab)**: Web automation and GUI interaction agent

## Data Flow

The ADP follows a staged pipeline with ATIF as an interchange layer:

```
sample_raw.json → raw_to_atif.py → sample_atif.json → atif_to_std.py → sample_std.json → agents/*/std_to_sft.py → sample_sft/<agent_name>.json
```

### 1. Raw Data
Original format from various sources (research papers, datasets, etc.)

### 2. ATIF Format
Dataset-specific raw-to-ATIF conversion using Harbor's Agent Trajectory Interchange Format. This layer preserves the raw tool/action shape with minimal normalization and is validated by `ATIFTrajectory`.

### 3. ATIF Normalization and Standardized Format
`atif_to_std.py` normalizes ATIF tool names/arguments and emits ATIF JSONL. `raw_to_standardized.py` is retained as a compatibility command name that also emits normalized ATIF. Committed `sample_std.json` fixtures are ATIF std data:
- **Steps**: `system`, `user`, or `agent` turns with natural-language messages
- **Tool calls**: `function_name`, `arguments`, and `tool_call_id`
- **Observations**: Tool/environment results linked with `source_call_id`

### 4. SFT Format
Agent-specific format ready for supervised fine-tuning. Shared `std_to_sft.py` converters accept ATIF input after dataset-specific `atif_to_std.py` normalization.

## Documentation

### Schema Documentation
The canonical standardized schema is the ATIF model in [`schema/atif.py`](schema/atif.py).

### Contributing Guidelines
To contribute new datasets or agent implementations:
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Step-by-step contribution guide

## Repository Structure

```
agent-data-protocol/
├── datasets/           # Dataset implementations
│   ├── swe-smith/     # Example dataset
│   │   ├── extract_raw.py
│   │   ├── raw_to_atif.py
│   │   ├── atif_to_std.py
│   │   ├── raw_to_standardized.py
│   │   ├── metadata.json
│   │   ├── sample_raw.json
│   │   ├── sample_atif.json
│   │   ├── sample_std.json
│   │   ├── sample_sft/   # Sample SFT format
│   │   │   ├── openhands_v0.json
│   │   │   ├── sweagent.json
│   │   │   └── ...
│   │   ├── README.md
│   │   └── LICENSE
│   └── ...
├── agents/            # Agent implementations
│   ├── openhands_v0/  # OpenHands v0 agent
│   ├── sweagent/      # SWE-agent
│   ├── agentlab/      # AgentLab
│   └── ...
├── schema/            # ATIF schema definitions
│   └── atif.py        # ATIF trajectory, step, tool-call, and observation models
├── scripts/           # Utility scripts
└── tests/            # Validation tests
```

## Examples

### Converting a Single Dataset

```bash
# Example: Convert swe-smith dataset -> ATIF -> OpenHands v0 SFT
export MY_DATASET=swe-smith
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract and convert through ATIF
python datasets/$MY_DATASET/extract_raw.py | \
python datasets/$MY_DATASET/raw_to_atif.py | \
python datasets/$MY_DATASET/atif_to_std.py | \
python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash \
> swe_smith_openhands_v0.jsonl
```

### Web-based Dataset Example

```bash
# Example: Convert web browsing dataset
export MY_DATASET=mind2web
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py | \
python datasets/$MY_DATASET/raw_to_atif.py | \
python datasets/$MY_DATASET/atif_to_std.py | \
python agents/openhands_v0/std_to_sft.py --is_web=yes --api_env=browser \
> mind2web_openhands_v0.jsonl
```

## Testing and Validation

Run the test suite to validate data integrity:

```bash
# Test all datasets
python -m pytest tests/ -v

# Test specific dataset
python -m pytest tests/test_standardized_schemas.py -v -k swe-smith

# Test SFT conversion
python -m pytest tests/test_std_to_sft_conversion.py -v
```

## Quality Control

The repository includes built-in quality control measures:

- **Schema Validation**: Pydantic models ensure type safety
- **Pre-commit Hooks**: Code formatting and linting
- **Automated Testing**: Comprehensive test suite for data validation
- **Sample Verification**: Each dataset includes validated samples

## License

This project is licensed under the MIT License. Individual datasets may have their own licenses - please check the respective dataset README files.

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{song2025agentdataprotocolunifying,
    title={Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents},
    author={Yueqi Song and Ketan Ramaneti and Zaid Sheikh and Ziru Chen and Boyu Gou and Tianbao Xie and Yiheng Xu and Danyang Zhang and Apurva Gandhi and Fan Yang and Joseph Liu and Tianyue Ou and Zhihao Yuan and Frank Xu and Shuyan Zhou and Xingyao Wang and Xiang Yue and Tao Yu and Huan Sun and Yu Su and Graham Neubig},
    year={2025},
    url={https://arxiv.org/abs/2510.24702},
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Adding new datasets
- Implementing new agent formats
- Improving existing conversions
- Reporting issues and bugs

## Support

For questions, issues, or discussions:

- **Issues**: [GitHub Issues](https://github.com/neulab/agent-data-protocol/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neulab/agent-data-protocol/discussions)

---

**Note**: This repository is actively maintained and regularly updated with new datasets and agent implementations. Check the [Recent Release](#recent-release) for the latest updates.
