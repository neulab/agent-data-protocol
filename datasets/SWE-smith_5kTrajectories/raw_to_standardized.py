import json
import re
import sys

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str]) -> list:
    if step["role"] == "user":
        # Check if it's an observation (starts with OBSERVATION:)
        if step["content"].startswith("OBSERVATION:"):
            # Remove "OBSERVATION:" prefix and clean up
            content = step["content"][len("OBSERVATION:") :].strip()
            return [TextObservation(content=content, source="system")]
        else:
            return [TextObservation(content=step["content"], source="user")]

    elif step["role"] == "system":
        return [TextObservation(content=step["content"], source="system")]

    elif step["role"] == "assistant":
        result = []
        content = step["content"]

        # Check for function calls in the format <function=name>\n<parameter=param>value</parameter>\n</function>
        # We need to handle nested function calls properly
        # First, find all top-level function calls (not nested within other function calls)

        # This is a more complex problem than it seems. We need to parse the XML-like structure
        # while respecting nesting. Let's use a stack-based approach.

        # Find all opening and closing function tags
        open_tags = [(m.start(), m.group(1)) for m in re.finditer(r"<function=([^>]+)>", content)]
        close_tags = [(m.start(), m.end()) for m in re.finditer(r"</function>", content)]

        # Sort all tags by position
        all_tags = [(pos, "open", name) for pos, name in open_tags] + [
            (pos, "close", None) for pos, _ in close_tags
        ]
        all_tags.sort()

        # Use a stack to track nesting and find top-level functions
        stack = []
        top_level_functions = []

        for pos, tag_type, name in all_tags:
            if tag_type == "open":
                stack.append((pos, name))
            elif tag_type == "close" and stack:
                start_pos, func_name = stack.pop()
                # If stack is empty after popping, this is a top-level function
                if not stack:
                    top_level_functions.append((start_pos, pos + len("</function>"), func_name))

        if top_level_functions:
            current_pos = 0

            for start_pos, end_pos, func_name in top_level_functions:
                # Add any text before this function call as a text observation
                before_text = content[current_pos:start_pos].strip()
                if before_text:
                    result.append(TextObservation(content=before_text, source="system"))

                # Extract the full function call text
                full_function_call = content[start_pos:end_pos]

                # Parse the function call
                function_match = re.match(
                    r"<function=([^>]+)>(.*)</function>", full_function_call, re.DOTALL
                )
                if function_match:
                    function_name = function_match.group(1)
                    params_content = function_match.group(2)

                    # Map function names
                    if function_name == "bash":
                        function_name = "execute_bash"
                    if function_name == "submit":
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

                    # Add the API action with the full function call text
                    result.append(
                        ApiAction(
                            function=function_name, kwargs=kwargs, function_call=full_function_call
                        )
                    )

                current_pos = end_pos

            # Add any remaining text after the last function call
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                result.append(TextObservation(content=remaining_text, source="system"))

        # Check for traditional code blocks if no function calls found
        elif "```" in content:
            code_block_regex = re.search(r"```(\w+)\n(.*?)\n```", content, re.DOTALL)
            if code_block_regex:
                description_text = content[: code_block_regex.start()].strip()
                if description_text:
                    result.append(TextObservation(content=description_text, source="system"))

                # For code blocks, treat as API action
                result.append(
                    ApiAction(
                        function="code_execution",
                        kwargs={
                            "language": code_block_regex.group(1).lower(),
                            "code": code_block_regex.group(2),
                        },
                    )
                )
            else:
                # Regular message content
                result.append(TextObservation(content=content, source="system"))
        else:
            # Regular message content
            result.append(TextObservation(content=content, source="system"))

        return result
    else:
        raise Exception("Invalid role.")


def process_data(raw_data):
    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    return Trajectory(id=raw_data["instance_id"], content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        standardized_data = process_data(raw_data)
        print(standardized_data.model_dump_json())
