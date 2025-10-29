# Schema Documentation

This document describes the standardized schema used in the agent-data-protocol for representing agent trajectories and interactions.

## What are Schemas?

Schemas define the structure and format for representing agent training data in a consistent, standardized way. They ensure that data from different sources can be processed uniformly while maintaining semantic meaning to enable effective conversion to agent specific SFT formats.

We uses Pydantic models to enforce type safety and validation, ensuring data integrity across the entire pipeline from raw data extraction to supervised fine-tuning (SFT) format conversion.

## Core Schema Components

All schema implementation could be found at [`schema`](../schema/)

### Trajectory

**File**: [`schema/trajectory.py`](../schema/trajectory.py)

The root container for all agent interaction data.

**Fields**:
- `id` (str): Unique identifier for the trajectory
- `content` (list): Sequence of actions and observations that make up the trajectory
- `details` (dict): Additional dataset-specific metadata (typically not used for training)

**Purpose**: Represents a complete sequence of agent interactions, containing both actions taken by the agent and observations received from the environment / user.

### Action Schemas

Actions represent steps taken by an agent during task execution.

Base Class Implementation: [`schema/action/action.py`](../schema/action/action.py)

#### ApiAction

**File**: [`schema/action/api.py`](../schema/action/api.py)

Represents function/API calls made by the agent.

**Fields**:
- `class_` (str): Always "api_action"
- `function` (str): Name of the function being called
- `kwargs` (dict): Arguments passed to the function
- `description` (str, optional): Agent's reasoning or thought process

**Use Case**: Tool usage, API calls, function invocations (e.g., file operations, web requests, calculations)

#### CodeAction

**File**: [`schema/action/code.py`](../schema/action/code.py)

Represents code execution by the agent.

**Fields**:
- `class_` (str): Always "code_action"
- `language` (Literal): Programming language (supports 300+ languages including Python, JavaScript, bash, etc.)
- `content` (str): The actual code to execute
- `description` (str): Agent's reasoning or explanation

**Use Case**: Code generation, script execution, programming tasks, terminal commands

#### MessageAction

**File**: [`schema/action/message.py`](schema/action/message.py)

Represents communication/messages from the agent.

**Fields**:
- `class_` (str): Always "message_action"
- `content` (str): The message content
- `description` (str, optional): Additional context or reasoning

**Use Case**: Agent responses, explanations, status updates, user communication

### Observation Schemas

Observations represent information received by the agent from its environment.

Base Observation Implementation: [`schema/observation/observation.py`](../schema/observation/observation.py)

#### TextObservation

**File**: [`schema/observation/text.py`](../schema/observation/text.py)

Represents textual information received by the agent.

**Fields**:
- `class_` (str): Always "text_observation"
- `content` (str): The textual content
- `name` (str, optional): Name of the participant/source
- `source` (Literal): Origin of the text - "user", "agent", or "environment"

**Use Case**: User messages, system outputs, file contents, terminal responses, error messages

#### WebObservation

**File**: [`schema/observation/web.py`](../schema/observation/web.py)

Represents web page state and structure.

**Fields**:
- `class_` (str): Always "web_observation"
- `html` (str, optional): Raw HTML content
- `axtree` (str, optional): Accessibility tree representation
- `url` (str, optional): Web page URL
- `image_observation` (ImageObservation, optional): Screenshot of the page
- `viewport_size` (tuple, optional): Browser viewport dimensions

**Use Case**: Web automation, browser interactions, web scraping, UI testing

## Example Standardized Format

```json
{
  "id": "example_trajectory_001",
  "content": [
    {
      "class_": "MessageAction",
      "message": "Please help me solve this problem",
      "thoughts": "I need to understand what the user is asking for"
    },
    {
      "class_": "TextObservation",
      "text": "I'll help you solve this problem. What specific issue are you facing?",
      "source": "agent"
    }
  ],
  "details": {
    "dataset": "example_dataset",
    "task_type": "problem_solving"
  }
}
```

## Schema Validation

The repository uses **Pydantic validation** to ensure data integrity and type safety. All schemas are built on Pydantic BaseModel, providing:

- **Automatic type checking**: Fields are validated against their declared types
- **Custom validators**: Using `@field_validator` decorators to enforce specific constraints
- **Required field validation**: Ensures all mandatory fields are present
- **Class field validation**: Each schema validates its `class_` field matches the expected value
- **Runtime validation**: Data is validated when objects are created or modified

Key validation features:
- Required `class_` fields match expected values (e.g., "api_action", "text_observation")
- Type constraints are enforced (e.g., Literal types for `source` and `language` fields)
- Data integrity is maintained across conversions
- Validation errors provide clear feedback for debugging

## Data Flow

1. **Raw Data**: Original format from various sources
2. **Standardized Format**: Converted using these schemas
3. **SFT Format**: Further processed for supervised fine-tuning

The schemas serve as the bridge between diverse raw data formats and the standardized representation needed for effective agent training.
