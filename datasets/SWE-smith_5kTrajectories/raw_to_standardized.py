import json
import sys
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "user":
        # Check if it's an observation (starts with OBSERVATION:)
        if step["content"].startswith("OBSERVATION:"):
            # Remove "OBSERVATION:" prefix and clean up
            content = step["content"][len("OBSERVATION:"):].strip()
            return [TextObservation(content=content, source="execution")]
        else:
            return [TextObservation(content=step["content"], source="user")]
    
    elif step["role"] == "system":
        return [TextObservation(content=step["content"], source="system")]
    
    elif step["role"] == "assistant":
        # First check for traditional code blocks (```language\ncode\n```)
        code_block_regex = re.search(r'```(\w+)\n(.*?)\n```', step["content"], re.DOTALL)
        
        if code_block_regex:
            description_text = step["content"][:code_block_regex.start()].strip()
            return [
                CodeAction(
                    language=code_block_regex.group(1).lower(),
                    content=code_block_regex.group(2),
                    description=description_text if description_text else None,
                )
            ]
        
        # Check for function calls (<function=name><parameter=param>value</parameter></function>)
        function_regex = re.search(r'<function=(\w+)>\s*<parameter=\w+>(.*?)</parameter>\s*</function>', step["content"], re.DOTALL)
        
        if function_regex:
            function_name = function_regex.group(1).lower()
            content = function_regex.group(2).strip()
            description_text = step["content"][:function_regex.start()].strip()
            
            # Map function names to valid languages
            language_mapping = {
                'bash': 'bash',
                'python': 'python',
                'repl': 'python',
                'str_replace_editor': 'python',
                'submit': 'python',
                # Add other mappings as needed
            }
            
            if function_name in language_mapping:
                return [
                    CodeAction(
                        language=language_mapping[function_name],
                        content=content,
                        description=description_text if description_text else None,
                    )
                ]
            else:
                # For unknown functions, treat them as MessageActions
                return [
                    MessageAction(
                        content=step["content"],
                        description=None,
                    )
                ]
        
        # Otherwise, it's a regular message
        return [
            MessageAction(
                content=step["content"],
                description=None,
            )
        ]
    else:
        raise Exception("Invalid role.")

for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    # Standardize the data - use instance_id as the id
    standardize_data = Trajectory(
        id=raw_data["instance_id"],
        content=content
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())