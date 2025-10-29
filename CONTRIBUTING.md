# Contributing to Agent Data Protocol

This repository serves as a standardized collection of agent training data from the research community. We welcome contributions of new datasets and agent implementations that follow the standardized format.

## Table of Contents

- [Overview](#overview)
- [Development Setup](#development-setup)
- [Contributing New Datasets](#contributing-new-datasets)
- [Contributing New Agents](#contributing-new-agents)
- [Testing and Validation](#testing-and-validation)
- [Code Quality Standards](#code-quality-standards)
- [Submission Guidelines](#submission-guidelines)

## Overview

This repository contains:
- **Datasets**: Agent training data in standardized format ([`datasets/`](datasets))
- **Agents**: Agent-specific implementations and conversion scripts ([`agents/`](agents)])
- **Schema**: ADP standardized format definitions ([`schema/`](schema))
- **Scripts**: General tility scripts ([`scripts/`](scripts))

### Data Flow
```
Raw Dataset      →  Standardized Format  →  Agent Specific SFT Format
      ↓                   ↓                       ↓
sample_raw.json  →  sample_std.json      →  sample_sft.json
```

### Standardized Schema Components

Our standardized format uses these main components:

#### Actions
- **MessageAction**: Text-based communication
- **CodeAction**: Code execution requests
- **ApiAction**: API calls

#### Observations
- **TextObservation**: Text-based responses
- **WebObservation**: Web page content

## Development Setup

### Prerequisites

- Python 3.12+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/neulab/agent-data-protocol.git
cd agent-data-protocol
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install pre-commit hooks:
```bash
./install_hooks.sh
```

This will install pre-commit and set up the hooks to ensure code quality.

## Contributing New Datasets

Note: see [`schema/SCHEMA.md`](schema/SCHEMA.md) first for better understanding of the ADP standardized data format.

### Dataset Structure

Each dataset must follow this directory structure:

```
datasets/$YOUR_DATASET_NAME/
├── README.md                   # Dataset description and usage
├── requirements.txt            # Dataset-specific dependencies (optional)
├── schema_raw.py               # Raw data schema (optional)
├── extract_raw.py              # Script to extract raw data
├── raw_to_standardized.py      # Conversion script
├── api.py                      # API definitions (optional)
├── sample_raw.json             # 5 raw samples
├── sample_std.json             # 5 standardized samples
└── sample_sft                  # SFT format samples
  ├── sample_sft_$AGENT1.json   # 5 SFT format samples (if applicable)
  ├── sample_sft_$AGENT2.json   # 5 SFT format samples (if applicable)
  └── sample_sft_$AGENT3.json   # 5 SFT format samples (if applicable)
```

### Step-by-Step Guide

#### Step 1: Create Sample Raw Data

1. Create your dataset directory:
```bash
export MY_DATASET=$YOUR_DATASET_NAME
mkdir -p datasets/$MY_DATASET
```

2. Create `extract_raw.py` that outputs raw data in JSONL format to stdout. Example:
```python
#!/usr/bin/env python3
"""Extract raw data from the original dataset source."""

import json
import sys
from pathlib import Path

def extract_raw_data():
    """Extract and yield raw data samples."""
    # Your extraction logic here
    # This should read from the original data source
    # and yield individual samples

    for sample in your_data_source:
        yield {
            "id": sample.id,
            "content": sample.content,
            # ... other fields
        }

if __name__ == "__main__":
    for sample in extract_raw_data():
        print(json.dumps(sample)) # The output file will be in jsonl format.
```

3. **Create raw data schema (optional)**: If your raw data has a complex or well-defined structure, create a `schema_raw.py` file to define Pydantic models for validation and type safety.

   **When to include schema_raw.py:**
   - Your raw data has a complex nested structure with many fields
   - You want type validation and better error messages during processing
   - The original dataset has a well-documented schema you want to preserve
   - You're working with structured data formats (JSON with specific schemas, API responses, etc.)

   **How to create schema_raw.py:**
   ```python
   from typing import List, Optional
   from pydantic import BaseModel

   class YourDataItem(BaseModel):
       """Define structure for individual data items."""
       field1: str
       field2: Optional[int] = None
       nested_data: List[dict]

   class SchemaRaw(BaseModel):
       """Main schema for your raw data format."""
       id: str
       items: List[YourDataItem]
       metadata: Optional[dict] = None
   ```

4. Generate sample raw data:
```bash
python datasets/$MY_DATASET/extract_raw.py | python scripts/jsonl_to_json.py | jq '.[0:5]' > datasets/$MY_DATASET/sample_raw.json
```

#### Step 2: Create Standardized Format Converter

1. Create `raw_to_standardized.py` that converts raw data to ADP standardized schemas.
Essentially, you should map each action / observation in the raw data to an action / observation in [ADP schemas](schema/SCHEMA.md#core-schema-components).

**Brief conversion examples:**
```python
# Raw data examples → Standardized actions

# Text message: {"type": "message", "text": "Hello"} →
MessageAction(class_="MessageAction", message="Hello")

# Code execution: {"type": "code", "language": "python", "code": "print('hi')"} →
CodeAction(class_="CodeAction", language="python", content="print('hi')")

# Function call: {"type": "action", "function": "click", "args": {"xpath": "//button"}} →
ApiAction(class_="api_action", function="click", kwargs={"xpath": "//button"})

# Environment response: {"type": "output", "text": "Command executed"} →
TextObservation(class_="TextObservation", text="Command executed", source="environment")
```

**Complete example:**
```python
#!/usr/bin/env python3
"""Convert raw data to standardized format."""

import json
import sys
from schema.trajectory import Trajectory
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.action.api import ApiAction
from schema.observation.text import TextObservation

def convert_raw_to_standardized(raw_data):
    """Convert a raw data sample to standardized format."""

    content = []

    # Example conversion logic
    for item in raw_data["content"]:
        if item["type"] == "message":
            content.append(MessageAction(
                class_="MessageAction",
                message=item["message"],
                description=item.get("thoughts", "")
            ))
        elif item["type"] == "python":
            content.append(CodeAction(
                class_="CodeAction",
                language="python",
                content=item["code"],
                description=item.get("thoughts", "")
            ))
        elif item["type"] == "action":
            content.append(ApiAction(
                class_="api_action",
                function=item["function"],
                kwargs=item["args"],
                description=item.get("thoughts", "")
            ))
        elif item["type"] == "observation":
            content.append(TextObservation(
                class_="TextObservation",
                text=item["text"],
                source="environment"
            ))

    return Trajectory(
        id=raw_data["id"],
        content=content,
        details=raw_data.get("metadata", {})
    )

if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line.strip())
        standardized = convert_raw_to_standardized(raw_data)
        print(standardized.model_dump_json())
```

**Using schema_raw.py in raw_to_standardized.py (if applicable):**
   ```python
   from schema_raw import SchemaRaw

   def convert_raw_to_standardized(raw_data):
       # Validate and parse raw data
       data = SchemaRaw(**raw_data)

       # Now use typed data with IDE support and validation
       content = []
       for item in data.items:
           # Process with type safety
           pass
   ```

2. **Create API definitions (when applicable)**: If your standardized dataset contains [ApiAction](schema/SCHEMA.md#apiaction), create an `api.py` file to define the available functions. This is crucial for later conversion to agent specific formats.

   **When to include api.py:**
   - Your dataset contains structured actions (e.g., `go("bedroom")`, `click("button_id")`, `search("query")`)
   - Actions have specific parameters and can be represented as function calls
   - The data contains interactions with a specific environment, tool, or API
   - You want to convert actions to `ApiAction` schema objects in your standardized format

   **Why api.py is important for agents:**
   - **Function Call Training**: Agents learn to make structured function calls rather than generating free-form text
   - **Parameter Validation**: Ensures agents learn correct parameter types and formats
   - **Standardization**: Provides a consistent representation for tool calls across different datasets
   - **SFT Conversion**: Many agent training frameworks expect function call formats for fine-tuning

   **What to include in api.py:**
   ```python
   def function_name(param1: type, param2: type) -> return_type:
       """Clear description of what the function does.

       Args:
           param1 (type): Description of parameter 1
           param2 (type): Description of parameter 2

       Example:
           function_name("example_value", 123)
       """
       pass
   ```

   **Example api.py for a web browsing dataset:**
   ```python
   def click(xpath: str) -> None:
       """Click on a web element.

       Args:
           xpath (str): The xpath of the element to click.

       Example:
           click("//button[@id='submit']")
       """
       pass

   def type(xpath: str, text: str) -> None:
       """Type text into an input field.

       Args:
           xpath (str): The xpath of the input element.
           text (str): The text to type.

       Example:
           type("//input[@name='username']", "john_doe")
       """
       pass
   ```

   Then in your `raw_to_standardized.py`, use `ApiAction` for structured actions:
   ```python
   from schema.action.api import ApiAction

   # Convert structured actions to ApiAction
   if action_type == "click":
       content.append(ApiAction(
           class_="api_action",
           function="click",
           kwargs={"xpath": action_data["xpath"]},
           description=action_data.get("thought", "")
       ))
   ```


3. Generate standardized samples:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
```

#### Step 3: Validate Standardized Format

Run validation tests to ensure your standardized format is correct:
```bash
python -m pytest tests/test_standardized_schemas.py::test_sample_standardized_against_schema -v
```

Or to test a specific dataset:
```bash
python -m pytest tests/test_standardized_schemas.py -v -k $MY_DATASET
```

#### Step 4: Create Dataset README

Create a comprehensive `README.md` for your dataset:

- **Original Paper**: [Paper Title](link)
- **Original Repository**: [Repository](link)
- **License**: License information
- **Size**: Number of samples, average steps per trajectory
- **Description**: Description of the dataset and its purpose.

## Contributing New Agents

### Agent Structure

Each agent implementation should follow this structure:

```
agents/YOUR_AGENT_NAME/
├── __init__.py                # Agent module initialization
├── api.py                     # Agent-specific API definitions
├── std_to_sft.py              # Standardized to SFT conversion
├── system_prompt/             # System prompts and templates
└── README.md                  # Agent documentation
```

### Step-by-Step Guide

#### Step 1: Create Agent Directory

```bash
export AGENT_NAME=YOUR_AGENT_NAME
mkdir -p agents/$AGENT_NAME
```

#### Step 2: Define Agent APIs

Create `api.py` to load APIs from datasets, use any format that suits your agent.
The loaded specs could be used for training function calling models.

Example:
```python
import importlib.util
import inspect
api_file_path = os.path.expanduser(f"datasets/{dataset}/api.py")
if os.path.exists(api_file_path):
    spec = importlib.util.spec_from_file_location("api", api_file_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    functions = inspect.getmembers(api_module, inspect.isfunction)
    sigs = {}
    for name, func in functions:
        docstring = "\n" + inspect.getdoc(func)
        sig = inspect.signature(func)
```

#### Step 3: Create System Prompts

Create system prompt templates in `system_prompt/default.txt`:

```
You are an AI assistant specialized in [your agent's domain].

Your capabilities include:
- Capability 1
- Capability 2
- Capability 3

Guidelines:
- Guideline 1
- Guideline 2
- Guideline 3

When responding, always:
- Be clear and concise
- Follow the specified format
- Provide reasoning for your actions
```

#### Step 4: Create SFT Conversion Script

Create `std_to_sft.py` for converting standardized format to your agent's SFT format.
The output data should ideally be directly usable for training using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

```python
#!/usr/bin/env python3
"""Convert standardized format to SFT format for Your Agent."""

import json
import sys
from typing import Dict, Any
from schema.trajectory import Trajectory

def convert_trajectory_to_sft(trajectory: Trajectory) -> Dict[str, Any]:
    """Convert a standardized trajectory to SFT format."""

    messages = []

    # Add system message
    messages.append({
        "role": "system",
        "content": "Your agent's system prompt here"
    })

    # Convert ADP action and observation content to agent specific messages
    for item in trajectory.content:
        if item.class_ == "MessageAction":
            messages.append({
                "role": "user" if item.source == "user" else "assistant",
                "content": item.message
            })
        elif item.class_ == "TextObservation":
            messages.append({
                "role": "system",
                "content": f"Observation: {item.text}"
            })
        # Handle other observation types...

    return {
        "id": trajectory.id,
        "conversations": messages,
        "metadata": trajectory.details
    }

if __name__ == "__main__":
    for line in sys.stdin:
        trajectory_data = json.loads(line.strip())
        trajectory = Trajectory(**trajectory_data)
        sft_data = convert_trajectory_to_sft(trajectory)
        print(json.dumps(sft_data))
```

#### Step 5: Generate SFT format samples using the appropriate conversion script:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/YOUR_AGENT_NAME/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_YOUR_AGENT_NAME.json
```

For example, for OpenHands:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=yes --api_env=browser | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_openhands.json
```

#### Step 6: Create Agent README

Create a comprehensive `README.md`:

```markdown
# Your Agent Name

Description of your agent and its capabilities.

## Overview

- **Domain**: Agent's specialized domain
- **Capabilities**: List of key capabilities
- **API Environment**: Required API environment
- **Supported Actions**: List of supported actions
- **Supported Datasets**: List of supported datasets given this implementation
- **Instructions**: Command to generate the full SFT data for supported datasets

## Testing and Validation

### Running Tests

Run the full test suite:
```bash
pytest tests/
```

Run specific validation tests:
```bash
# Test raw schemas
pytest tests/test_raw_schemas.py

# Test standardized schemas
pytest tests/test_standardized_schemas.py

# Test dataset structure
pytest tests/test_dataset_structure.py
```

### Validation Checklist

Before submitting your contribution:

- [ ] Raw data extraction script works correctly
- [ ] Standardized format conversion passes validation
- [ ] SFT format conversion produces valid output
- [ ] All required files are present
- [ ] README is comprehensive and accurate
- [ ] Tests pass
- [ ] Code follows style guidelines

## Code Quality Standards

### Style Guidelines

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Include docstrings for functions and classes
- Keep line length under 100 characters

### Pre-commit Hooks

The repository uses pre-commit hooks to ensure code quality:
- **ruff**: Code formatting and linting
- **mypy**: Type checking (where applicable)

### File Organization

- Keep imports at the top of files
- Group imports: standard library, third-party, local
- Use absolute imports where possible
- Avoid circular imports

## Submission Guidelines

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Implement your changes** following the guidelines above
3. **Test thoroughly** using the validation scripts
4. **Update documentation** including README files
5. **Submit a pull request** with a clear description

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] New dataset
- [ ] New agent
- [ ] Bug fix
- [ ] Documentation update
- [ ] Other (please describe)

## Dataset/Agent Information
- **Name**:
- **Source**:
- **Size**:
- **Domain**:

## Testing
- [ ] All tests pass
- [ ] Validation scripts run successfully
- [ ] Sample files generated correctly

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is complete
- [ ] All required files are present
- [ ] No sensitive data included
```

### Review Process

1. **Automated checks** will run on your PR
2. **Manual review** by maintainers
3. **Feedback incorporation** if needed
4. **Final approval** and merge

### Common Issues

- **Schema validation failures**: Ensure your standardized format follows the schema exactly
- **Missing files**: Check that all required files are present
- **Import errors**: Verify that all dependencies are properly specified
- **Test failures**: Run tests locally before submitting

## Getting Help

### Resources

- **Documentation**: Check existing dataset READMEs for examples
- **Schema definitions**: Review files in [`schema/`](../schema/) directory
- **Example implementations**: Look at [existing datasets](../datasets/) and [agents](../agents/)

### Contact

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub discussions for general questions
- **Email**: Contact the maintainers directly for sensitive issues

### FAQ

**Q: How do I handle datasets with special requirements?**
A: Create a `requirements.txt` file in your dataset directory and document any special setup in the README.

**Q: What if my dataset doesn't fit the standard schema?**
A: Open an issue to discuss potential solutions.

**Q: How do I handle large datasets?**
A: Focus on the conversion scripts and sample files. Full datasets should be processed separately.

**Q: Can I contribute datasets that require special licenses?**
A: Yes, but clearly document the license requirements in your README.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.
