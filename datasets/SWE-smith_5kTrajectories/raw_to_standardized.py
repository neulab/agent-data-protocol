import json
import random
import re
import sys

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step) -> list:
    if step.role == "system":
        return []

    elif step.role == "user":
        content = step.content
        # Handle observations that start with "OBSERVATION:"
        if content.startswith("OBSERVATION:"):
            content = content[len("OBSERVATION:") :].strip()
            return [TextObservation(content=content, source="environment")]
        else:
            return [TextObservation(content=content, source="user")]

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

                # Create description from before_text
                description = before_text if before_text else None

                if function_name == "bash":
                    result.append(
                        CodeAction(
                            language="bash", content=kwargs["command"], description=description
                        )
                    )
                else:
                    result.append(
                        ApiAction(function=function_name, kwargs=kwargs, description=description)
                    )

                current_pos = match.end()

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
    # Only keep successful trajectories
    if not isinstance(content[-1], ApiAction) or content[-1].function != "submit":
        return None
    user_end_message = random.choice(
        [
            [
                TextObservation(
                    content="Congratulations! You have successfully solved the task.",
                    source="user",
                ),
            ],
            [
                TextObservation(
                    content="Your solution has been verified as correct. ", source="user"
                ),
            ],
            [
                TextObservation(
                    content="Well done on successfully completing the task!", source="user"
                ),
            ],
            [
                TextObservation(
                    content="Your implementation satisfies the task requirements.",
                    source="user",
                ),
            ],
            [
                TextObservation(content="Task completed successfully.", source="user"),
            ],
        ]
    )
    content.extend(user_end_message)
    assistant_end_message = random.choice(
        [
            [
                MessageAction(
                    content="<finish> I have successfully completed the task. </finish>",
                    description="",
                ),
            ],
            [
                MessageAction(
                    content="<finish> I did it! The task is now complete. </finish>",
                    description="",
                ),
            ],
            [
                MessageAction(
                    content="<finish> The objective has been achieved with no outstanding issues. </finish>",
                    description="",
                ),
            ],
            [
                MessageAction(
                    content="<finish> I have fulfilled all the requirements of the task. </finish>",
                    description="",
                ),
            ],
            [
                MessageAction(
                    content="<finish> I've wrapped up the task successfully. </finish>",
                    description="",
                ),
            ],
        ]
    )
    content.extend(assistant_end_message)
    return Trajectory(id=data.instance_id, content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
