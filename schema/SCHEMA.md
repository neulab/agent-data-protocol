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
- `schema_version` (str): ADP standardized schema version used by this trajectory. Defaults to the current schema version for backward-compatible parsing, but committed `sample_std.json` files must include it explicitly.
- `id` (str): Unique identifier for the trajectory
- `content` (list): Sequence of actions and observations that make up the trajectory
- `available_apis` (list, optional): API function names available for this trajectory. Only populate this for datasets that have `api.py` and whose source data explicitly specifies per-instance tool/API availability; do not populate it by copying all functions from `api.py` or inferring it from APIs used in the trajectory. When present, it must be a subset of the dataset's `api.py` functions and cover every `ApiAction.function` in the trajectory.
- `details` (dict): Additional dataset-specific metadata.

**Purpose**: Represents a complete sequence of agent interactions, containing both actions taken by the agent and observations received from the environment / user.

### Action Schemas

Actions represent steps taken by an agent during task execution.

Base Class Implementation: [`schema/action/action.py`](../schema/action/action.py)

All action types inherit from the base `Action` class which provides common fields:

- `tool_call_id` (str, optional): Stable identifier for this tool/action call.
  When populated, exactly one later observation must use the same `tool_call_id`
  so converters can emit matched tool-call/result pairs.
- `reasoning_content` (str, optional): Extended chain-of-thought reasoning or internal thinking from the agent. This captures deliberate reasoning processes (e.g., `<think>` blocks) that are separate from the action's brief description. This field aligns with Harbor ATIF's `reasoning_content` field and Agent Client Protocol's `agent_thought_chunk` concept.
- `reward` (float, optional): Per-step reward signal associated with this action. Used for capturing rewards earned during reinforcement learning training and evaluation.

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

All observation types inherit from the base `Observation` class which provides a common field:

- `tool_call_id` (str, optional): Stable identifier for the action/tool call
  that produced this observation. When populated, it must match a preceding
  `Action.tool_call_id`.
- `reward` (float, optional): Per-step reward signal associated with this observation. Used for reinforcement learning training data.

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
  "schema_version": "1.3.2",
  "id": "example_trajectory_001",
  "content": [
    {
      "class_": "text_observation",
      "content": "Please list the files in this project.",
      "source": "user"
    },
    {
      "class_": "code_action",
      "tool_call_id": "call_000001",
      "language": "bash",
      "content": "ls",
      "description": "I'll inspect the current directory."
    },
    {
      "class_": "text_observation",
      "tool_call_id": "call_000001",
      "content": "README.md\nschema\ndatasets",
      "source": "environment"
    },
    {
      "class_": "message_action",
      "content": "The project contains README.md, schema, and datasets."
    }
  ],
  "details": {
    "dataset": "example_dataset",
    "task_type": "problem_solving"
  }
}
```

### Matched Tool Calls And Tool Results

ADP represents a tool call as an `Action` and the corresponding tool result as
an `Observation`. When the source data can identify this relationship, populate
the same `tool_call_id` on both records:

- `Action.tool_call_id` identifies the call made by the agent.
- `Observation.tool_call_id` identifies the action that produced the result.
- A populated observation `tool_call_id` must match a preceding action
  `tool_call_id`.
- A populated action `tool_call_id` must have exactly one matched observation
  result.
- In schema version 1.3.0 and later, a tool action (`ApiAction` or `CodeAction`)
  that is immediately followed by an observation result must include a
  `tool_call_id`, and the observation must use the same ID. Existing 1.2.0 data
  with adjacent tool-call/result pairs but no IDs should be migrated by running
  the dataset converter through `create_trajectory_with_tool_call_links`.

This distinction matters because some datasets encode environment or tool
feedback as text that otherwise looks like a user message. `source="user"`
should be reserved for actual user instructions, corrections, or interruptions.
Tool/environment feedback after an action should use `source="environment"` and
the matching `tool_call_id` when available.

## Schema Versioning

The current ADP standardized schema version is defined in [`schema/version.py`](../schema/version.py) as `SCHEMA_VERSION`. Versions use `MAJOR.MINOR.PATCH` semantics:

- **Major**: Backward-incompatible changes such as removing or renaming fields, adding required fields, or narrowing accepted values.
- **Minor**: Backward-compatible additions such as optional fields, new action/observation types, or additional accepted values.
- **Patch**: Validation or documentation fixes that preserve compatibility.

Any PR that changes schema-impacting Python files under `schema/` must increase `SCHEMA_VERSION`. CI enforces this with `scripts/check_schema_version_bump.py`. Documentation-only changes to `schema/SCHEMA.md` do not require a version bump.

Committed standardized samples must include a root-level `schema_version` equal to the current `SCHEMA_VERSION`. The `Trajectory` model still defaults missing versions to the current version so older external data can be parsed during migration.

`SUPPORTED_SCHEMA_VERSIONS` should include prior versions when a bump is intended to preserve compatibility with existing standardized data. Remove older versions only when the schema intentionally drops parsing support for that version.

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
- Extra fields are rejected on standardized trajectory, action, and observation models
- Data integrity is maintained across conversions
- Validation errors provide clear feedback for debugging

## Data Flow

1. **Raw Data**: Original format from various sources
2. **Standardized Format**: Converted using these schemas
3. **SFT Format**: Further processed for supervised fine-tuning

The schemas serve as the bridge between diverse raw data formats and the standardized representation needed for effective agent training.
