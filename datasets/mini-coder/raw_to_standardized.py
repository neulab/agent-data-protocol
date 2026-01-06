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

# Remove ONLY formatting / style constraints, not termination semantics
_FORMATTING_RULES_RE = re.compile(
    r"^.*?(?=##\s*Submission)",
    re.DOTALL | re.IGNORECASE,
)

_TAG_WRAPPER_RE = re.compile(
    r"^\s*<(?P<tag>[a-zA-Z0-9_:-]+)>\s*(?P<body>.*)\s*</\1>\s*$", re.DOTALL
)
_STOP_BASH_FENCE_RE = re.compile(
    r"```bash\s*\n(.*?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def strip_user_formatting(content: str) -> str:
    """
    Removes formatting-specific instructions but preserves
    semantic control instructions (e.g. how to signal task completion).
    """
    if not content:
        return content

    # Extract <instructions> block if present
    m = re.search(r"<instructions>(.*?)</instructions>", content, re.DOTALL | re.IGNORECASE)
    if m:
        instructions = m.group(1)

        # Remove formatting rules but KEEP the Submission / stop protocol
        instructions = re.sub(_FORMATTING_RULES_RE, "", instructions).strip()

        # Replace original instructions block with the cleaned version
        content = content[: m.start()] + instructions + content[m.end() :]

    # Unwrap <pr_description> if it is the outer wrapper
    wrapper = _TAG_WRAPPER_RE.match(content)
    if wrapper and wrapper.group("tag").lower() == "pr_description":
        content = wrapper.group("body").strip()
    content = re.sub(_STOP_BASH_FENCE_RE, r"\1", content)
    return content.strip()


def convert_step(step) -> list:
    if step.role == "system":
        # Skip system prompt - it only contains formatting instructions, not task content
        return []

    elif step.role == "user":
        content = step.content

        # Check if this is the initial task description or a command execution result
        # Command execution results have <returncode> and <output> tags
        returncode_pattern = r"<returncode>(\d+)</returncode>"
        output_pattern = r"<output>(.*?)</output>"

        returncode_match = re.search(returncode_pattern, content, re.DOTALL)
        output_match = re.search(output_pattern, content, re.DOTALL)

        if returncode_match and output_match:
            # This is a command execution result
            returncode = returncode_match.group(1)
            output = output_match.group(1).strip()

            # Format as plain text observation (similar to other coding datasets)
            observation_content = output if output else ""
            return [TextObservation(content=observation_content, source="environment")]
        else:
            # This is the initial task description from user
            content = strip_user_formatting(content)
            return [TextObservation(content=content, source="user")]

    elif step.role == "assistant":
        result = []
        content = step.content

        # Check for bash code blocks in the format ```bash\ncommand\n```
        bash_block_pattern = r"```bash\n(.*?)\n```"
        bash_matches = list(re.finditer(bash_block_pattern, content, re.DOTALL))

        if bash_matches:
            # Extract everything before the code block as the description/thought
            first_match = bash_matches[0]
            description = content[: first_match.start()].strip()

            # Remove "THOUGHT: " prefix to match other datasets
            if description.startswith("THOUGHT: "):
                description = description[len("THOUGHT: ") :].strip()

            # Extract the bash command
            bash_command = first_match.group(1).strip()

            # Create a CodeAction for the bash command
            result.append(
                CodeAction(
                    language="bash",
                    content=bash_command,
                    description=description if description else None,
                )
            )
        else:
            # Check for function calls in the format <function=name>
            function_pattern = r"<function=([^>]+)>"
            function_match = re.search(function_pattern, content)

            if function_match:
                # Get text before the function call as description
                description = content[: function_match.start()].strip()
                function_name = function_match.group(1)

                # For submit and other simple functions without parameters
                result.append(
                    ApiAction(
                        function=function_name,
                        kwargs={},
                        description=description if description else None,
                    )
                )
            else:
                # No code blocks or function calls, treat as regular message
                result.append(MessageAction(content=content))

        return result

    else:
        raise Exception(f"Invalid role: {step.role}")


def process_data(data):
    content = []
    for step in data.messages:
        content.extend(convert_step(step))
    if not isinstance(content[-1], CodeAction):
        print(f"not codeaction: {content[-1]}", file=sys.stderr)
        return None
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in content[-1].content:
        print(f"not COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT: {content[-1]}", file=sys.stderr)
        return None
    # Add success message from user
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

    # Add assistant end message
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
    return Trajectory(id=data.id, content=content)


if __name__ == "__main__":
    # Read input as newline-delimited JSON (JSONL)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
