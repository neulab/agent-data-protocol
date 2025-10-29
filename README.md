# Agent Data Protocol (ADP)

A standardized protocol for collecting, processing, and converting agent training data from diverse sources into unified formats suitable for supervised fine-tuning (SFT).

## Recent Release
- Check out our arXiv preprint: [https://arxiv.org/abs/2510.24702](https://arxiv.org/abs/2510.24702)

- Try a demo of data conversion on our Project Website: [https://www.agentdataprotocol.com/](https://www.agentdataprotocol.com/)

- Download ADP data: [https://huggingface.co/collections/neulab/agent-data-protocol](https://huggingface.co/collections/neulab/agent-data-protocol)

## Overview

The Agent Data Protocol provides a systematic approach to handle agent training data across different domains, environments, and agent architectures. It standardizes the representation of agent trajectories, actions, and observations, enabling seamless conversion between raw datasets and agent-specific training formats.

### Key Features

- **Standardized Schema**: Unified representation for agent actions and observations across different domains
- **Multi-Agent Support**: Convert data for different agent architectures (OpenHands, SWE-agent, AgentLab, etc.)
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

### Basic Usage

To obtain data for a specific dataset and agent, follow this pattern:

```bash
# Set your dataset name
export MY_DATASET=swe-smith
mkdir -p datasets/$MY_DATASET/full_sft

# Step 1: Extract raw data
echo "Extracting raw data..."
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl

# Step 2: Convert to standardized format
echo "Converting to standardized format..."
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_standardized.py > datasets/$MY_DATASET/full_std.jsonl

# Step 3: Convert to agent-specific SFT format
echo "Converting to SFT format..."
export PYTHONPATH=`pwd`:$PYTHONPATH

# For OpenHands, there are dataset specific arguments to pass in
export MY_AGENT=openhands
cat datasets/$MY_DATASET/full_std.jsonl | python agents/$MY_AGENT/std_to_sft.py --is_web=no --api_env=execute_bash > datasets/$MY_DATASET/full_sft/full_sft_$MY_AGENT.jsonl

# For SWE-agent
export MY_AGENT=sweagent
cat datasets/$MY_DATASET/full_std.jsonl | python agents/$MY_AGENT/std_to_sft.py > datasets/$MY_DATASET/full_sft/full_sft_$MY_AGENT.jsonl
```

### Available Datasets

The repository includes datasets from various domains (we welcome more contributions!):

- **Coding**: `code_feedback`, `codeactinstruct`
- **Software Engineering**: `swe-smith`, `swe-gym_openhands_sampled_trajectories`,
- **Web Browsing**: `mind2web`, `nnetnav-live`, `nnetnav-wa`, `go-browse-wa`, `synatra`
- **Multi-domain**: `agenttuning_*`, `orca_agentinstruct`, `openhands`

### Supported Agents

- **[OpenHands](https://github.com/OpenHands/OpenHands)**: General-purpose coding and web browsing agent
- **[SWE-agent](https://github.com/SWE-agent/SWE-agent)**: Software engineering focused agent
- **[AgentLab](https://github.com/ServiceNow/AgentLab)**: Web automation and GUI interaction agent

## Data Flow

The ADP follows a three-stage pipeline:

```
Raw Dataset      →  Standardized Format  →  Agent Specific SFT Format
      ↓                   ↓                       ↓
sample_raw.json  →  sample_std.json      →  sample_sft.json
```

### 1. Raw Data
Original format from various sources (research papers, datasets, etc.)

### 2. Standardized Format
Unified representation using ADP schemas:
- **Actions**: `MessageAction`, `CodeAction`, `ApiAction`
- **Observations**: `TextObservation`, `WebObservation`
- **Trajectory**: Container for complete interaction sequences

### 3. SFT Format
Agent-specific format ready for supervised fine-tuning

## Documentation

### Schema Documentation
For detailed information about ADP schemas, data structures, and validation:
- **[SCHEMA.md](schema/SCHEMA.md)** - Complete schema documentation with examples

### Contributing Guidelines
To contribute new datasets or agent implementations:
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Step-by-step contribution guide

## Repository Structure

```
agent-data-protocol/
├── datasets/           # Dataset implementations
│   ├── swe-smith/     # Example dataset
│   │   ├── extract_raw.py
│   │   ├── raw_to_standardized.py
│   │   ├── api.py
│   │   └── sample_*.json
│   └── ...
├── agents/            # Agent implementations
│   ├── openhands/     # OpenHands agent
│   ├── sweagent/      # SWE-agent
│   ├── agentlab/      # AgentLab
│   └── ...
├── schema/            # ADP schema definitions
│   ├── SCHEMA.md      # Schema documentation
│   ├── action/        # Action schemas
│   ├── observation/   # Observation schemas
│   └── trajectory.py  # Trajectory container
├── scripts/           # Utility scripts
└── tests/            # Validation tests
```

## Examples

### Converting a Single Dataset

```bash
# Example: Convert swe-smith dataset for OpenHands
export MY_DATASET=swe-smith
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract and convert
python datasets/$MY_DATASET/extract_raw.py | \
python datasets/$MY_DATASET/raw_to_standardized.py | \
python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash \
> swe_smith_openhands.jsonl
```

### Web-based Dataset Example

```bash
# Example: Convert web browsing dataset
export MY_DATASET=mind2web
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py | \
python datasets/$MY_DATASET/raw_to_standardized.py | \
python agents/openhands/std_to_sft.py --is_web=yes --api_env=browser \
> mind2web_openhands.jsonl
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
