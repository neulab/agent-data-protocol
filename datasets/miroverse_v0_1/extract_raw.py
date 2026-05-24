import ast
import json
import keyword
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

SOURCE_DATASET = os.environ.get("MIROVERSE_SOURCE_DATASET", "miromind-ai/MiroVerse-v0.1")
CONFIG_FILES = [
    ("MiroVerse-Voyager1.0", "jsonl_sft/MiroVerse-Voyager1.0.jsonl"),
    ("MiroVerse-MuSiQue", "jsonl_sft/MiroVerse-MuSiQue.jsonl"),
    ("MiroVerse-HotpotQA", "jsonl_sft/MiroVerse-HotpotQA.jsonl"),
    ("MiroVerse-WebWalkerQA-Silver", "jsonl_sft/MiroVerse-WebWalkerQA-Silver.jsonl"),
    ("MiroVerse-MegaScience", "jsonl_sft/MiroVerse-MegaScience.jsonl"),
    ("MiroVerse-TaskCraft", "jsonl_sft/MiroVerse-TaskCraft.jsonl"),
    ("MiroVerse-QA-Expert-Multi-Hop-V1.0", "jsonl_sft/MiroVerse-QA-Expert-Multi-Hop-V1.0.jsonl"),
    (
        "MiroVerse-OneGen-TrainDataset-MultiHopQA",
        "jsonl_sft/MiroVerse-OneGen-TrainDataset-MultiHopQA.jsonl",
    ),
    ("MiroVerse-2WikiMultihopQA", "jsonl_sft/MiroVerse-2WikiMultihopQA.jsonl"),
    ("MiroVerse-WikiTables", "jsonl_sft/MiroVerse-WikiTables.jsonl"),
    ("MiroVerse-WebShaper", "jsonl_sft/MiroVerse-WebShaper.jsonl"),
    ("MiroVerse-WebDancer", "jsonl_sft/MiroVerse-WebDancer.jsonl"),
]

SERVER_RE = re.compile(
    r"## Server name:\s*(?P<server>.*?)\s*\n(?P<body>.*?)(?=\n## Server name:|\Z)",
    re.DOTALL,
)
TOOL_RE = re.compile(
    r"### Tool name:\s*(?P<tool>.*?)\s*\n(?P<body>.*?)(?=\n### Tool name:|\Z)",
    re.DOTALL,
)


def _selected_configs():
    requested = os.environ.get("MIROVERSE_CONFIGS")
    if not requested:
        return CONFIG_FILES
    wanted = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [(name, path) for name, path in CONFIG_FILES if name in wanted]
    missing = wanted - {name for name, _ in selected}
    if missing:
        raise ValueError(f"Unknown MiroVerse configs: {sorted(missing)}")
    return selected


def _resolve_path(path):
    # The source repository stores SFT JSONL files under jsonl_sft/. Some public mirrors
    # flatten those files at the repository root; this keeps sample regeneration possible
    # without changing the default source dataset.
    if os.environ.get("MIROVERSE_FLAT_LAYOUT") == "1":
        return path.rsplit("/", 1)[-1]
    return path


def _open_hf_file(path):
    url = f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/main/{_resolve_path(path)}"
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"MiroVerse-v0.1 is gated on Hugging Face (HTTP {exc.code}). "
                "Accept the dataset terms at "
                "https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1 "
                "and provide an authorized HF_TOKEN, or set MIROVERSE_SOURCE_DATASET and "
                "MIROVERSE_FLAT_LAYOUT for a mirror with the same JSONL files."
            ) from exc
        raise


def sanitize_identifier(name: str) -> str:
    identifier = re.sub(r"\W", "_", name.strip())
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        identifier = "value"
    if identifier[0].isdigit():
        identifier = f"tool_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_"
    return identifier


def tool_function_name(server_name: str, tool_name: str) -> str:
    return f"{sanitize_identifier(server_name)}__{sanitize_identifier(tool_name)}"


def _first_mapping_literal(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _parse_input_schema(schema_text: str) -> dict[str, Any]:
    schema_text = _first_mapping_literal(schema_text.strip())
    if not schema_text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(schema_text)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, SyntaxError, TypeError, ValueError):
            continue
    return {}


def _argument_name_map(input_schema: dict[str, Any]) -> dict[str, str]:
    properties = input_schema.get("properties", {})
    used: set[str] = set()
    argument_map = {}
    for raw_name in properties:
        sanitized = sanitize_identifier(raw_name)
        candidate = sanitized
        suffix = 2
        while candidate in used:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1
        used.add(candidate)
        argument_map[raw_name] = candidate
    return argument_map


def extract_available_tools_from_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
    tools = []
    seen = set()
    for server_match in SERVER_RE.finditer(system_prompt):
        server_name = server_match.group("server").strip()
        for tool_match in TOOL_RE.finditer(server_match.group("body")):
            tool_name = tool_match.group("tool").strip()
            dedupe_key = (server_name, tool_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            block = tool_match.group("body").strip()
            if "Input JSON schema:" in block:
                description_text, schema_text = block.split("Input JSON schema:", 1)
            else:
                description_text, schema_text = block, ""
            description = re.sub(r"^Description:\s*", "", description_text.strip())
            input_schema = _parse_input_schema(schema_text)
            tools.append(
                {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "function_name": tool_function_name(server_name, tool_name),
                    "description": description,
                    "input_schema": input_schema,
                    "argument_name_map": _argument_name_map(input_schema),
                }
            )
    return tools


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def extract_available_tools_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    system_prompt = "\n\n".join(
        _message_value(message, "content") or ""
        for message in messages
        if _message_value(message, "role") == "system"
    )
    return extract_available_tools_from_system_prompt(system_prompt)


def iter_config_rows(config_name, path):
    with _open_hf_file(path) as response:
        for row_index, raw_line in enumerate(response):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line.decode("utf-8"))
            row.setdefault("split", config_name)
            row.setdefault("id", f"{config_name}-{row_index}")
            row["available_tools"] = extract_available_tools_from_messages(row.get("messages", []))
            yield row


def main():
    max_per_config = os.environ.get("MIROVERSE_MAX_PER_CONFIG")
    max_per_config = int(max_per_config) if max_per_config else None
    for config_name, path in _selected_configs():
        for row_index, row in enumerate(iter_config_rows(config_name, path)):
            if max_per_config is not None and row_index >= max_per_config:
                break
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
