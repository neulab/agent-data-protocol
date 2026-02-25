import json
import re
import sys

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def extract_thinking_and_json(content: str) -> tuple[str | None, dict | None]:
    """Extract thinking block and JSON from assistant response."""
    thinking = None
    json_data = None

    # Extract <think>...</think> block
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    # Extract JSON object from content
    # Find the first { and last } to extract JSON
    json_start = content.find("{")
    json_end = content.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = content[json_start : json_end + 1]
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return thinking, json_data


def convert_step(step: dict, is_first_user: bool = False) -> list:
    """Convert a conversation step to standardized format."""
    role = step["role"]
    content = step["content"]

    if role == "user":
        # User messages are task descriptions or terminal outputs
        if is_first_user:
            # First user message contains system prompt and task description
            # Extract just the task part
            source = "user"
        else:
            # Subsequent user messages are terminal outputs
            source = "environment"
        return [TextObservation(content=content, source=source)]

    elif role == "assistant":
        result = []
        thinking, json_data = extract_thinking_and_json(content)

        if json_data:
            # Extract analysis as description
            description = json_data.get("analysis") or json_data.get("plan")

            # Process commands
            commands = json_data.get("commands", [])
            if commands:
                for cmd in commands:
                    keystrokes = cmd.get("keystrokes", "")
                    if keystrokes:
                        # Clean keystrokes - remove trailing newline for display
                        clean_cmd = keystrokes.rstrip("\n")
                        if clean_cmd:
                            result.append(
                                CodeAction(
                                    language="bash",
                                    content=clean_cmd,
                                    description=thinking,
                                )
                            )
                            thinking = None  # Only use thinking for first command

            # Check if task is complete
            task_complete = json_data.get("task_complete", False)
            if task_complete and not result:
                result.append(
                    MessageAction(
                        content="<finish> Task completed successfully. </finish>",
                        description=description,
                    )
                )
            elif task_complete:
                # Add finish message at end
                result.append(
                    MessageAction(
                        content="<finish> Task completed successfully. </finish>",
                        description=None,
                    )
                )
        else:
            # No JSON found, treat as plain message
            result.append(MessageAction(content=content, description=None))

        if not result:
            # Return a message action if no commands were extracted
            result.append(MessageAction(content=content, description=None))

        return result

    return []


def process_trajectory(raw_data: dict) -> Trajectory | None:
    """Process a raw trajectory into standardized format."""
    conversations = raw_data["conversations"]
    content = []
    is_first_user = True

    for step in conversations:
        converted = convert_step(step, is_first_user=is_first_user)
        if step["role"] == "user" and is_first_user:
            is_first_user = False
        content.extend(converted)

    if not content:
        return None

    # Generate ID from task and episode if available
    task = raw_data.get("task", "")
    episode = raw_data.get("episode", "")
    run_id = raw_data.get("run_id", "")

    if task and episode:
        traj_id = f"{task}_{episode}"
    elif run_id:
        traj_id = run_id
    else:
        traj_id = f"traj_{hash(str(raw_data)) % 100000}"

    return Trajectory(id=traj_id, content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        trajectory = process_trajectory(raw_data)
        if trajectory:
            print(trajectory.model_dump_json())
