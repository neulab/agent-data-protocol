import json
import re
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory


def message_text(message: Message) -> str:
    return "\n".join(part.text for part in message.content if part.text).strip()


def browser_function(message: Message) -> str | None:
    target = message.recipient or message.channel
    if not target:
        return None
    if target.startswith("browser."):
        return target.split(".", 1)[1]
    return None


def parse_tool_arguments(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def extract_browser_url(text: str) -> str | None:
    line_match = re.search(r"^L\d+: URL: (\S+)", text, re.MULTILINE)
    if line_match:
        return line_match.group(1)

    header_match = re.match(r"\[\d+\].*\(([^()\s]+://[^()\s]+)\)", text)
    if header_match:
        return header_match.group(1)

    return None


def browser_observation(text: str, name: str | None) -> WebObservation | TextObservation:
    url = extract_browser_url(text)
    if not url:
        return TextObservation(content=text, source="environment", name=name)
    return WebObservation(
        html=None,
        axtree=text,
        url=url,
        image_observation=None,
        viewport_size=None,
    )


def append_pending_message(content: list, pending_description: str | None) -> None:
    if pending_description:
        content.append(MessageAction(content=pending_description))


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    developer_messages = []
    pending_description = None
    saw_user = False

    for message in data.messages:
        text = message_text(message)

        if message.role == "system":
            continue

        if message.role == "developer":
            if text:
                developer_messages.append(text)
            continue

        if message.role == "user":
            initial_parts = []
            if not saw_user and developer_messages:
                initial_parts.extend(developer_messages)
            if text:
                initial_parts.append(text)
            if initial_parts:
                content.append(TextObservation(content="\n\n".join(initial_parts), source="user"))
                saw_user = True
            continue

        if message.role == "tool":
            append_pending_message(content, pending_description)
            pending_description = None
            if message.name and message.name.startswith("browser."):
                content.append(browser_observation(text, message.name))
            else:
                content.append(
                    TextObservation(content=text, source="environment", name=message.name)
                )
            continue

        if message.role != "assistant":
            continue

        if message.channel == "final":
            if text:
                content.append(
                    MessageAction(
                        content=f"<finish> {text} </finish>",
                        description=pending_description,
                    )
                )
                pending_description = None
            continue

        function = browser_function(message)
        if message.content_type == "code" and function:
            kwargs = parse_tool_arguments(text)
            if kwargs is None:
                return None
            content.append(
                ApiAction(
                    function=function,
                    kwargs=kwargs,
                    description=pending_description,
                )
            )
            pending_description = None
            continue

        if text:
            pending_description = (
                f"{pending_description}\n\n{text}" if pending_description else text
            )

    if pending_description:
        content.append(MessageAction(content=pending_description))

    if not content:
        return None

    return Trajectory(
        id=f"{data.config or 'openresearcher'}_{data.split or 'train'}_{data.qid}",
        content=content,
        details={
            "source_qid": str(data.qid),
            "answer": data.answer or "",
            "config": data.config or "",
            "split": data.split or "",
            "status": data.status,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
