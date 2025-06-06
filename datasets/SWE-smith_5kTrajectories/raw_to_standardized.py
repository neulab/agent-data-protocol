import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step) -> list:
    if step.role in ["system", "user"]:
        content = step.content
        # Handle observations that start with "OBSERVATION:"
        if content.startswith("OBSERVATION:"):
            content = content[len("OBSERVATION:") :].strip()
            return [TextObservation(content=content, source="environment")]
        else:
            source = "user" if step.role == "user" else "environment"
            return [TextObservation(content=content, source=source)]

    elif step.role == "assistant":
        result = []
        content = step.content

        # Check for function calls in the format <function=name>\n<parameter=param>value</parameter>\n</function>
        function_pattern = r"<function=([^>]+)>\s*(.*?)\s*</function>"
        function_matches = list(re.finditer(function_pattern, content, re.DOTALL))

        if function_matches:
            current_pos = 0

            for match in function_matches:
                # Get text before this function call to use as description
                before_text = content[current_pos : match.start()].strip()

                # Parse the function call
                function_name = match.group(1)
                params_content = match.group(2)

                # Map function names
                if function_name == "bash":
                    function_name = "execute_bash"
                elif function_name == "submit":
                    function_name = "finish"

                # Parse parameters
                kwargs = {}
                param_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"
                param_matches = re.findall(param_pattern, params_content, re.DOTALL)

                for param_name, param_value in param_matches:
                    param_value = param_value.strip()

                    # Try to parse as JSON for arrays/objects, otherwise keep as string
                    if param_value.startswith("[") and param_value.endswith("]"):
                        try:
                            kwargs[param_name] = json.loads(param_value)
                        except:
                            kwargs[param_name] = param_value
                    elif param_value.isdigit():
                        kwargs[param_name] = int(param_value)
                    elif param_value in ["true", "false"]:
                        kwargs[param_name] = param_value == "true"
                    else:
                        kwargs[param_name] = param_value

                # Add required message parameter for finish function if not present
                if function_name == "finish" and "message" not in kwargs:
                    kwargs["message"] = "Task completed."

                # Add the API action with description from before_text
                description = before_text if before_text else None
                result.append(
                    ApiAction(function=function_name, kwargs=kwargs, description=description)
                )

                current_pos = match.end()

            # Add any remaining text after the last function call as a separate message
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                result.append(TextObservation(content=remaining_text, source="system"))

        else:
            # No function calls found, treat as regular message
            result.append(MessageAction(content=content))

        return result
    else:
        raise Exception(f"Invalid role: {step.role}")


def process_data(data):
    content = []
    for step in data.messages:
        content.extend(convert_step(step))

    return Trajectory(id=data.instance_id, content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
