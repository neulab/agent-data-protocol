import json
import re
import sys
from typing import List

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


def _split_observation_chunks(obs_text: str) -> List[str]:
    """Split an environment observation into chunks per shell prompt line.

    If no prompt-like lines are found, returns an empty list to signal 'no split'.
    """
    if not obs_text:
        return []
    prompt_regex = re.compile(r"(?m)^[^\n]*# .*$")
    matches = list(prompt_regex.finditer(obs_text))
    if not matches:
        return []

    chunks: List[str] = []
    header = obs_text[: matches[0].start()]
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(obs_text)
        segment = obs_text[start:end]
        if i == 0 and header:
            segment = header + segment
        chunks.append(segment)
    return chunks


def convert_step(step: dict, is_first_user: bool = False) -> list:
    """Convert a conversation step to standardized format."""
    role = step["role"]
    content = step["content"]

    if role == "user":
        content = content.replace("New Terminal Output:\n", "")
        if is_first_user:
            return [TextObservation(content=content, source="user")]
        # Environment observation: detect and split into multiple chunks if present
        chunks = _split_observation_chunks(content)
        if len(chunks) > 1:
            return [TextObservation(content=c, source="environment") for c in chunks]
        return [TextObservation(content=content, source="environment")]

    elif role == "assistant":
        result = []
        thinking, json_data = extract_thinking_and_json(content)

        if json_data:
            # Extract analysis as brief description
            description = json_data.get("analysis") or json_data.get("plan")

            # Process commands
            commands = json_data.get("commands", [])
            if commands:
                for idx, cmd in enumerate(commands):
                    if isinstance(cmd, str):
                        return []
                    keystrokes = cmd.get("keystrokes", "")
                    if keystrokes:
                        # Clean keystrokes - remove trailing newline for display
                        clean_cmd = keystrokes.rstrip("\n")
                        if clean_cmd:
                            result.append(
                                CodeAction(
                                    language="bash",
                                    content=clean_cmd,
                                    # Use reasoning_content for think blocks,
                                    # description for brief action description
                                    reasoning_content=thinking if idx == 0 else None,
                                    description=description if idx == 0 else None,
                                )
                            )

            # Check if task is complete
            task_complete = json_data.get("task_complete", False)
            if task_complete and not result:
                result.append(
                    MessageAction(
                        content="<finish> Task completed successfully. </finish>",
                        reasoning_content=thinking,
                        description=description,
                    )
                )
            elif task_complete:
                # Add finish message at end
                result.append(
                    MessageAction(
                        content="<finish> Task completed successfully. </finish>",
                        reasoning_content=None,
                        description=None,
                    )
                )
        else:
            # No JSON found, treat as plain message
            # Check if there's a think block in the content
            result.append(
                MessageAction(content=content, reasoning_content=thinking, description=None)
            )

        if not result:
            # Return a message action if no commands were extracted
            result.append(
                MessageAction(content=content, reasoning_content=thinking, description=None)
            )

        return result

    return []


def process_trajectory(raw_data: dict) -> Trajectory | None:
    """Process a raw trajectory and pair code actions with subsequent observations.

    - Buffer code actions when assistant outputs them
    - Split environment observations (already split in convert_step when needed)
    - Pair code actions to environment observations in order
    - Append message actions once buffered code actions are drained
    """
    conversations = raw_data["conversations"]
    content: List = []
    is_first_user = True

    pending_code: List = []
    pending_other: List = []  # message actions to emit after a batch of code actions

    for step in conversations:
        converted = convert_step(step, is_first_user=is_first_user)
        if step["role"] == "user" and is_first_user:
            # First user task description
            content.extend(converted)
            is_first_user = False
            continue

        # Classify converted items
        env_observations = [
            x for x in converted if isinstance(x, TextObservation) and x.source == "environment"
        ]
        user_observations = [
            x for x in converted if isinstance(x, TextObservation) and x.source == "user"
        ]
        code_actions = [x for x in converted if getattr(x, "class_", None) == "code_action"]
        other_actions = [
            x
            for x in converted
            if getattr(x, "class_", None) not in ("code_action", "text_observation")
        ]

        if user_observations:
            content.extend(user_observations)

        if code_actions:
            pending_code.extend(code_actions)

        if env_observations:
            i = 0
            while i < len(env_observations) and pending_code:
                content.append(pending_code.pop(0))
                content.append(env_observations[i])
                i += 1
            if i < len(env_observations):
                content.extend(env_observations[i:])
            if not pending_code and pending_other:
                content.extend(pending_other)
                pending_other = []

        if other_actions:
            if pending_code:
                pending_other.extend(other_actions)
            else:
                content.extend(other_actions)

    # Flush any remaining
    if pending_code:
        # Unpaired code actions present -> drop trajectory
        return None
    if pending_other:
        content.extend(pending_other)

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
