import json
import sys
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

global errors, languages
errors = []
languages = set()
def convert_step(step: dict[str, str], id: str) -> list[Action | Observation]:

    if step["role"] == "system":
        system_msg = step["content"]
        return [
            TextObservation(content=system_msg, source=step["role"]),
        ]

    assert step["role"] in ["assistant", "user"], f"Invalid role: {step['role']}"

    try:

        if step["role"] == "assistant":
            code_extract_regex = re.match(r'(.*?)?```(\w+)?\n(.*?)\n```', step["content"], re.DOTALL)
            if code_extract_regex and code_extract_regex.group(2):
                code_language = code_extract_regex.group(2).lower()
                code_content = code_extract_regex.group(3)
                code_description = code_extract_regex.group(1)
                return [
                    CodeAction(language=code_language, 
                            content=code_content, 
                            description=code_description),
                ]
            
            else:
                return [
                    MessageAction(content=step["content"], description=None),
                ]
        else:
            return [
                TextObservation(content=step["content"], source="user"),
            ]
    
    except Exception as e:
        # print(f"Error: {e}", file=sys.stderr)
        errors.extend([{'id': id, 'role': step['role'], 'content': step['content'], 'error': str(e)}])
        with open("datasets/orca_agentinstruct/errors.json", "w") as f:
            json.dump(errors, f, indent=4)
        languages.add(code_extract_regex.group(2).lower())
        with open("datasets/orca_agentinstruct/languages.json", "w") as f2:
            json.dump(list(languages), f2, indent=4)
        return "error"
        

for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["conversations"]:
        traj_step = convert_step(step, id=raw_data["id"])
        content.extend(traj_step if traj_step != "error" else [])

    # Standardize the data
    standardize_data = Trajectory(
        id=str(raw_data["id"]),
        content=content
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())

   