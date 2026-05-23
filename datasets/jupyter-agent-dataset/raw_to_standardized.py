import json
import sys

from schema_raw import Message, SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

JUPYTER_TOOL = "add_and_execute_jupyter_code_cell"
FINAL_ANSWER_TOOL = "final_answer"


def _optional_text(text: str | None) -> str | None:
    if text and text.strip():
        return text
    return None


def _finish_message(answer: str, description: str | None = None) -> MessageAction:
    return MessageAction(content=f"<finish> {answer} </finish>", description=description)


def _message_to_actions(message: Message, fallback_answer: str):
    content = message.content or ""
    description = _optional_text(content)

    if message.role == "user":
        return [TextObservation(content=content, source="user")]

    if message.role == "tool":
        return [TextObservation(content=content, source="environment")]

    if message.role != "assistant":
        print(f"Unknown message role: {message.role}", file=sys.stderr)
        return []

    if not message.tool_calls:
        return [MessageAction(content=content)] if description else []

    actions = []
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments or {}

        if function_name == JUPYTER_TOOL:
            code = arguments.get("code")
            if not code:
                print(f"Skipping Jupyter tool call without code: {arguments}", file=sys.stderr)
                continue
            actions.append(CodeAction(language="python", content=code, description=description))
            continue

        if function_name == FINAL_ANSWER_TOOL:
            answer = arguments.get("answer") or fallback_answer or content
            actions.append(_finish_message(answer, description))
            continue

        if arguments.get("code"):
            actions.append(
                CodeAction(language="python", content=arguments["code"], description=description)
            )
        elif arguments.get("answer"):
            actions.append(_finish_message(arguments["answer"], description))
        else:
            print(f"Unknown tool call: {function_name}", file=sys.stderr)

    return actions


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    for message in data.messages:
        content.extend(_message_to_actions(message, data.answer))

    if not content:
        return None

    if not isinstance(content[-1], MessageAction) or "<finish>" not in content[-1].content:
        content.append(_finish_message(data.answer))

    details = {
        "dataset": "jupyter-agent/jupyter-agent-dataset",
        "source_id": data.id,
        "split": data.split or "non_thinking",
        "question": data.question,
        "answer": data.answer,
        "executor_type": data.executor_type or "",
        "kaggle_dataset_name": data.kaggle_dataset_name or "",
        "files_used": json.dumps(data.files_used or []),
        "packages_used": json.dumps(data.packages_used or []),
    }

    return Trajectory(id=data.id, content=content, details=details)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
