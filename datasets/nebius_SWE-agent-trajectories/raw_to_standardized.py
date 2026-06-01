import json
import random
import re
import shlex
import sys
from dataclasses import dataclass

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

WINDOW_SIZE = 100


@dataclass
class SweAgentEditorState:
    open_file: str | None = None
    window_start: int = 1


def parse_line_range(value: str) -> tuple[int, int]:
    if ":" in value:
        start, end = value.split(":", 1)
        return int(start), int(end)
    line = int(value)
    return line, line


def parse_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def is_path_like(value: str) -> bool:
    return "/" in value or "." in value


def parse_edit_action(action_str: str) -> tuple[str | None, int, int, str]:
    parts = action_str.split(None, 3)
    if len(parts) < 3:
        raise ValueError(f"Malformed edit action: {action_str!r}")
    if re.match(r"^\d+(?::\d+)?$", parts[1]):
        path = None
        start_line, end_line = parse_line_range(parts[1])
        replacement_text = parts[2] if len(parts) == 3 else parts[2]
        if len(parts) == 4:
            replacement_text = action_str.split(None, 2)[2]
    else:
        path = parts[1]
        start_line, end_line = parse_line_range(parts[2])
        replacement_text = parts[3] if len(parts) > 3 else ""
    if replacement_text.endswith("end_of_edit"):
        replacement_text = replacement_text[: -len("end_of_edit")]
    return path, start_line, end_line, replacement_text.rstrip("\n")


def line_edit_command(path: str, start_line: int, end_line: int, replacement_text: str) -> str:
    return "\n".join(
        [
            "python - <<'PY'",
            "from pathlib import Path",
            f"path = Path({path!r})",
            f"start_line = {start_line}",
            f"end_line = {end_line}",
            f"replacement = {replacement_text!r}",
            "if replacement and not replacement.endswith('\\n'):",
            "    replacement += '\\n'",
            "lines = path.read_text().splitlines(keepends=True)",
            "lines[start_line - 1:end_line] = replacement.splitlines(keepends=True)",
            "path.write_text(''.join(lines))",
            "PY",
        ]
    )


def grep_command(search_term: str, path: str, recursive: bool) -> str:
    flag = "-RIn" if recursive else "-n"
    return f"grep {flag} -- {shlex.quote(search_term)} {shlex.quote(path)} | head -100"


def parse_action_text(item) -> tuple[str, str, str]:
    thought, action_str, remainder = item.text.rsplit("```", 2)
    thought, remainder = thought.strip(), remainder.strip()
    thought = thought[len("DISCUSSION") :].strip() if thought.startswith("DISCUSSION") else thought
    thought = thought[: -len("COMMAND")].strip() if thought.endswith("COMMAND") else thought
    thought = thought + " " + remainder if remainder else thought
    codeblock_lang = re.fullmatch(r"\w+\s*", action_str.splitlines()[0])
    if codeblock_lang:
        action_str = "\n".join(action_str.splitlines()[1:])
    action_str = action_str.strip()
    action_name = action_str.split()[0]
    return thought, action_name, action_str


def parse_api_action(item, state: SweAgentEditorState):
    thought, action_name, action_str = parse_action_text(item)
    action_args = shlex.split(action_str)[1:]

    if action_name == "open":
        path = action_args[0]
        state.open_file = path
        line_number = parse_int(action_args[1]) if len(action_args) > 1 else None
        state.window_start = line_number or 1
        kwargs = {"command": "view", "path": path}
        if line_number is not None:
            kwargs["view_range"] = [line_number, line_number + WINDOW_SIZE]
        return ApiAction(function="str_replace_editor", kwargs=kwargs, description=thought)

    if action_name == "goto":
        target = action_args[0]
        line_number = parse_int(target)
        if line_number is not None and state.open_file is not None:
            state.window_start = line_number
            return ApiAction(
                function="str_replace_editor",
                kwargs={
                    "command": "view",
                    "path": state.open_file,
                    "view_range": [line_number, line_number + WINDOW_SIZE],
                },
                description=thought,
            )
        if line_number is None and is_path_like(target):
            state.open_file = target
            state.window_start = 1
            return ApiAction(
                function="str_replace_editor",
                kwargs={"command": "view", "path": target},
                description=thought,
            )
        if state.open_file is not None:
            return CodeAction(
                language="bash",
                content=grep_command(target, state.open_file, recursive=False),
                description=thought,
            )

    if action_name == "scroll_down" and state.open_file is not None:
        state.window_start += WINDOW_SIZE
        return ApiAction(
            function="str_replace_editor",
            kwargs={
                "command": "view",
                "path": state.open_file,
                "view_range": [state.window_start, state.window_start + WINDOW_SIZE],
            },
            description=thought,
        )

    if action_name == "scroll_up" and state.open_file is not None:
        state.window_start = max(1, state.window_start - WINDOW_SIZE)
        return ApiAction(
            function="str_replace_editor",
            kwargs={
                "command": "view",
                "path": state.open_file,
                "view_range": [state.window_start, state.window_start + WINDOW_SIZE],
            },
            description=thought,
        )

    if action_name == "create":
        path = action_args[0]
        state.open_file = path
        state.window_start = 1
        return ApiAction(
            function="str_replace_editor",
            kwargs={"command": "create", "path": path, "file_text": ""},
            description=thought,
        )

    if action_name == "edit":
        path, start_line, end_line, replacement_text = parse_edit_action(action_str)
        path = path or state.open_file
        if path is None:
            raise ValueError("edit action encountered before any file was opened")
        state.open_file = path
        state.window_start = start_line
        return CodeAction(
            language="bash",
            content=line_edit_command(path, start_line, end_line, replacement_text),
            description=thought,
        )

    if action_name == "search_dir":
        search_term = action_args[0]
        directory = action_args[1] if len(action_args) > 1 else "."
        return CodeAction(
            language="bash",
            content=grep_command(search_term, directory, recursive=True),
            description=thought,
        )

    if action_name == "search_file":
        search_term = action_args[0]
        path = action_args[1] if len(action_args) > 1 else state.open_file or "."
        return CodeAction(
            language="bash",
            content=grep_command(search_term, path, recursive=False),
            description=thought,
        )

    if action_name == "find_file":
        file_name = action_args[0]
        directory = action_args[1] if len(action_args) > 1 else "."
        return CodeAction(
            language="bash",
            content=f"find {shlex.quote(directory)} -name {shlex.quote(file_name)} -print",
            description=thought,
        )

    if action_name == "submit":
        return ApiAction(function="submit", kwargs={}, description=thought)

    return CodeAction(
        language="bash",
        content=action_str,
        description=thought,
    )


def process_item(item, state: SweAgentEditorState):
    if item.role == "system":
        return None
    elif item.role == "user":
        return TextObservation(content=item.text, source="user")
    elif item.role == "ai" and "```" in item.text:
        try:
            return parse_api_action(item, state)
        except Exception:
            return MessageAction(content=item.text)
    elif item.role == "ai":
        return MessageAction(content=item.text)
    else:
        print(f"Unknown role: {item.role}", file=sys.stderr)
        return None


def process_data(data):
    content = []
    state = SweAgentEditorState()
    for item in data.trajectory:
        observation = process_item(item, state)
        if observation is not None:
            content.append(observation)

    # Handle finish action
    if isinstance(content[-1], ApiAction) or isinstance(content[-1], CodeAction):
        terminal_message_rng = random.Random(str(data.instance_id))
        user_end_message = terminal_message_rng.choice(
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
        assistant_end_message = terminal_message_rng.choice(
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

    return create_trajectory_with_tool_call_links(
        id=data.instance_id,
        content=content,
        details={
            "model_name": data.model_name or "",
            "exit_status": data.exit_status or "",
            "generated_patch": data.generated_patch or "",
            "eval_logs": data.eval_logs or "",
        },
    )


if __name__ == "__main__":
    # Process each line of input individually
    for line in sys.stdin:
        raw_data = json.loads(line)
        if not raw_data["target"]:
            continue
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)

        # Print the standardized data as JSON
        print(json.dumps(standardized_data.model_dump()))
