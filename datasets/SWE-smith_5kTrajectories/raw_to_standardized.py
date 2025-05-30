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


def convert_step(step: dict[str, str]) -> list[dict]:
    if step["role"] == "user":
        # Check if it's an observation (starts with OBSERVATION:)
        if step["content"].startswith("OBSERVATION:"):
            # Remove "OBSERVATION:" prefix and clean up
            content = step["content"][len("OBSERVATION:"):].strip()
            return [{
                "class_": "text_observation",
                "content": content,
                "source": "system"
            }]
        else:
            return [{
                "class_": "text_observation", 
                "content": step["content"],
                "source": "user"
            }]
    
    elif step["role"] == "system":
        return [{
            "class_": "text_observation",
            "content": step["content"],
            "source": "system"
        }]
    
    elif step["role"] == "assistant":
        result = []
        content = step["content"]
        
        # Check for function calls in the format <function=name>\n<parameter=param>value</parameter>\n</function>
        function_pattern = r'<function=([^>]+)>\s*(.*?)\s*</function>'
        function_matches = list(re.finditer(function_pattern, content, re.DOTALL))
        
        if function_matches:
            current_pos = 0
            
            for match in function_matches:
                # Add any text before this function call as a text observation
                before_text = content[current_pos:match.start()].strip()
                if before_text:
                    result.append({
                        "class_": "text_observation",
                        "content": before_text,
                        "source": "system"
                    })
                
                # Parse the function call
                function_name = match.group(1)
                params_content = match.group(2)
                
                # Map function names
                if function_name == "bash":
                    function_name = "execute_bash"
                
                # Parse parameters
                kwargs = {}
                param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
                param_matches = re.findall(param_pattern, params_content, re.DOTALL)
                
                for param_name, param_value in param_matches:
                    param_value = param_value.strip()
                    
                    # Try to parse as JSON for arrays/objects, otherwise keep as string
                    if param_value.startswith('[') and param_value.endswith(']'):
                        try:
                            kwargs[param_name] = json.loads(param_value)
                        except:
                            kwargs[param_name] = param_value
                    elif param_value.isdigit():
                        kwargs[param_name] = int(param_value)
                    elif param_value in ['true', 'false']:
                        kwargs[param_name] = param_value == 'true'
                    else:
                        kwargs[param_name] = param_value
                
                # Add the function call
                result.append({
                    "function": function_name,
                    "kwargs": kwargs,
                    "description": None
                })
                
                current_pos = match.end()
            
            # Add any remaining text after the last function call
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                result.append({
                    "class_": "text_observation",
                    "content": remaining_text,
                    "source": "system"
                })
        
        # Check for traditional code blocks if no function calls found
        elif '```' in content:
            code_block_regex = re.search(r'```(\w+)\n(.*?)\n```', content, re.DOTALL)
            if code_block_regex:
                description_text = content[:code_block_regex.start()].strip()
                if description_text:
                    result.append({
                        "class_": "text_observation",
                        "content": description_text,
                        "source": "system"
                    })
                
                # For code blocks, we'll treat them as function calls to a code execution function
                result.append({
                    "function": "code_execution",
                    "kwargs": {
                        "language": code_block_regex.group(1).lower(),
                        "code": code_block_regex.group(2)
                    },
                    "description": description_text if description_text else None
                })
            else:
                # Regular message content
                result.append({
                    "class_": "text_observation",
                    "content": content,
                    "source": "system"
                })
        else:
            # Regular message content
            result.append({
                "class_": "text_observation",
                "content": content,
                "source": "system"
            })
        
        return result
    else:
        raise Exception("Invalid role.")

for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    # Standardize the data - use instance_id as the id
    standardize_data = {
        "id": raw_data["instance_id"],
        "content": content
    }

    # Print the standardized data
    print(json.dumps(standardize_data))