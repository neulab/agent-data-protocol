"""Generate missing reasoning text for normalized ATIF tool-call steps."""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

from schema.atif import ATIFTrajectory, content_to_text, normalize_atif_trajectory

DATASET = os.getenv("MY_DATASET")
assert DATASET, "Please set the environment variable MY_DATASET"

GENERATED_THOUGHTS_FILE = os.path.join(f"datasets/{DATASET}/generated_thoughts.json")
if os.path.exists(GENERATED_THOUGHTS_FILE):
    with open(GENERATED_THOUGHTS_FILE) as f:
        GENERATED_THOUGHTS = json.load(f)
else:
    GENERATED_THOUGHTS = {}

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("openai api_key not found", file=sys.stderr)
client = openai.OpenAI(api_key=api_key)

EXAMPLES = """
EXAMPLE:
History:
User: Fix the failing tests.

Current Tool Call: execute_bash
Arguments: {"command": "pytest -q"}

Model Output:
{"description": "I need to run the focused test suite first so I can see the current failure before editing the code."}
"""


def generate_thought(context: str, function_name: str, arguments: dict) -> str:
    prompt = f"""
Based on the history and current tool call, generate a concise reasoning sentence from the agent's perspective.
{EXAMPLES}

History:
{context}

Current Tool Call: {function_name}
Arguments: {json.dumps(arguments, ensure_ascii=False)}

Respond only in valid JSON format with a single field "description".
"""
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    match = re.search(r'\{.*?"description"\s*:\s*".*?"\s*\}', content, re.DOTALL)
    if not match:
        print("No valid JSON found in GPT response:", content, file=sys.stderr)
        return ""
    try:
        return json.loads(match.group(0))["description"]
    except Exception as exc:  # noqa: BLE001
        print("JSON parsing failed:", exc, file=sys.stderr)
        return ""


def history_before_step(trajectory: ATIFTrajectory, step_index: int) -> str:
    lines = []
    for step in trajectory.steps[:step_index]:
        message = content_to_text(step.message)
        if message:
            lines.append(f"{step.source}: {message[:500]}")
        if step.observation:
            for result in step.observation.results:
                lines.append(f"observation: {content_to_text(result.content)[:500]}")
    return "\n".join(lines)


def generate_thoughts_for_line(line: str) -> None:
    trajectory = normalize_atif_trajectory(ATIFTrajectory(**json.loads(line)))
    trajectory_id = trajectory.trajectory_id or trajectory.session_id or "atif-trajectory"
    GENERATED_THOUGHTS.setdefault(trajectory_id, {})
    print(f"generating function thoughts for {trajectory_id}", file=sys.stderr)
    for step_index, step in enumerate(trajectory.steps):
        if not step.tool_calls or content_to_text(step.message).strip():
            continue
        key = str(step_index)
        if key not in GENERATED_THOUGHTS[trajectory_id]:
            call = step.tool_calls[0]
            GENERATED_THOUGHTS[trajectory_id][key] = generate_thought(
                history_before_step(trajectory, step_index), call.function_name, call.arguments
            )
    with open(GENERATED_THOUGHTS_FILE, "w") as f:
        json.dump(GENERATED_THOUGHTS, f, indent=2, ensure_ascii=False)


def process_line(line: str) -> ATIFTrajectory:
    trajectory = normalize_atif_trajectory(ATIFTrajectory(**json.loads(line)))
    trajectory_id = trajectory.trajectory_id or trajectory.session_id or "atif-trajectory"
    for step_index, step in enumerate(trajectory.steps):
        thought = GENERATED_THOUGHTS.get(trajectory_id, {}).get(str(step_index))
        if thought and not content_to_text(step.message).strip():
            step.message = thought
    return trajectory


def test(line: str) -> ATIFTrajectory:
    generate_thoughts_for_line(line)
    return process_line(line)


if __name__ == "__main__":
    with open(f"datasets/{DATASET}/full_std.jsonl") as f:
        lines = f.readlines()

    output = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(test, line) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures)):
            data_with_thoughts = future.result()
            output.append(data_with_thoughts.model_dump_json(exclude_none=True))
    with open(f"datasets/{DATASET}/full_std.jsonl", "w") as f:
        f.write("\n".join(output))
