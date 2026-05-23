import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

_FORMATTING_RULES_RE = re.compile(r"^.*?(?=##\s*Submission)", re.DOTALL | re.IGNORECASE)
_STOP_BASH_FENCE_RE = re.compile(
    r"```bash\s*\n(.*?MINI_SWE_AGENT_FINAL_OUTPUT.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
_RETURN_CODE_RE = re.compile(r"<returncode>(.*?)</returncode>", re.DOTALL)
_OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def strip_user_formatting(content: str) -> str:
    if not content:
        return content

    match = re.search(r"<instructions>(.*?)</instructions>", content, re.DOTALL | re.IGNORECASE)
    if match:
        instructions = re.sub(_FORMATTING_RULES_RE, "", match.group(1)).strip()
        content = content[: match.start()] + instructions + content[match.end() :]

    content = re.sub(_STOP_BASH_FENCE_RE, r"\1", content)
    return content.strip()


def strip_thought_prefix(content: str) -> str:
    content = content.strip()
    return re.sub(r"^THOUGHT:\s*", "", content, flags=re.IGNORECASE).strip()


def convert_user_message(content: str) -> TextObservation:
    returncode_match = _RETURN_CODE_RE.search(content)
    output_match = _OUTPUT_RE.search(content)
    if returncode_match:
        if output_match:
            output = output_match.group(1).strip()
        else:
            output = _RETURN_CODE_RE.sub("", content).strip()
        return TextObservation(content=output, source="environment")
    return TextObservation(content=strip_user_formatting(content), source="user")


def convert_assistant_message(content: str) -> CodeAction | MessageAction:
    bash_matches = list(_BASH_BLOCK_RE.finditer(content))
    if not bash_matches:
        return MessageAction(content=content)

    match = bash_matches[-1]
    description = strip_thought_prefix(content[: match.start()])
    command = match.group(1).strip()
    return CodeAction(language="bash", content=command, description=description or None)


def convert_step(step) -> list:
    if step.role == "system":
        return []
    if step.role == "user":
        return [convert_user_message(step.content)]
    if step.role == "assistant":
        return [convert_assistant_message(step.content)]
    raise ValueError(f"Invalid role: {step.role}")


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    for step in data.messages:
        content.extend(convert_step(step))

    if not content or not isinstance(content[-1], CodeAction):
        return None
    if "MINI_SWE_AGENT_FINAL_OUTPUT" not in content[-1].content:
        return None

    content.append(
        TextObservation(
            content="Congratulations! You have successfully solved the task.",
            source="user",
        )
    )
    content.append(
        MessageAction(
            content="<finish> I have successfully completed the task. </finish>",
            description="",
        )
    )
    return Trajectory(id=data.instance_id, content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
