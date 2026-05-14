import json
import sys

from schema_raw import SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

PLACEHOLDER_MARKER = "no solution written"


def is_placeholder_solution(content: str) -> bool:
    return not content.strip() or PLACEHOLDER_MARKER in content.lower()


def make_details(data: SchemaRaw) -> dict[str, str]:
    return {
        "source": "https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev",
        "task_toml": data.task_toml,
        "solution_path": data.solution.path,
        "verification_files": ", ".join(file.path for file in data.verification_files),
    }


def make_verification_observation(data: SchemaRaw) -> str:
    files = ", ".join(file.path for file in data.verification_files) or "not included"
    return (
        f"Reference solution script `{data.solution.path}` is provided by the source dataset. "
        f"Verifier files available for this task: {files}."
    )


def process_data(data: SchemaRaw) -> Trajectory | None:
    if is_placeholder_solution(data.solution.content):
        return None

    content = [
        TextObservation(content=data.instruction.strip(), source="user"),
        CodeAction(
            language="bash",
            content=data.solution.content.rstrip() + "\n",
            description="Run the dataset-provided reference solution script for this terminal task.",
        ),
        TextObservation(content=make_verification_observation(data), source="environment"),
        MessageAction(
            content="<finish> Completed the terminal task with the provided reference solution. </finish>"
        ),
    ]
    return Trajectory(id=data.id, content=content, details=make_details(data))


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data is not None:
            print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))
