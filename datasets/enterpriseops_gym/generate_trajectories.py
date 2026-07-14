#!/usr/bin/env python3
"""Generate EnterpriseOps-Gym agent trajectories for ADP.

EnterpriseOps-Gym (ServiceNow Research) publishes *tasks* -- a system prompt, a
user request, the oracle tool set, and SQL ``verifiers`` -- that are evaluated
against live, resettable, containerized MCP servers. It does **not** publish
ready-made agent rollouts: a trajectory only exists once an agent is run against
the gym. This script produces the OpenAI-style chat trajectory file
(``enterpriseops_gym_gold.json``) that the ADP pipeline consumes.

Three modes:

1. ``--from-hf`` (default, no gym/containers needed): download the published
   tasks from Hugging Face (``ServiceNow-AI/EnterpriseOps-Gym``) and build
   *illustrative reference trajectories* grounded in the real task fields
   (task id, domain, system/user prompts, oracle tool list, SQL verifiers). The
   tool-call arguments and observations are compact, representative values that
   demonstrate the trajectory shape; each record is flagged with a ``note``.
   Example access tokens embedded in ``gym_servers_config`` are redacted.

       python datasets/enterpriseops_gym/generate_trajectories.py --from-hf \
           --per-domain 1 --config oracle \
           --output datasets/enterpriseops_gym/enterpriseops_gym_gold.json

2. ``--from-results``: convert result logs written by EnterpriseOps-Gym's own
   ``evaluate.py`` into authentic gold trajectories (keeps rollouts whose score
   meets ``--min-reward`` by default).

       python datasets/enterpriseops_gym/generate_trajectories.py \
           --from-results /path/to/EnterpriseOps-Gym/results \
           --output datasets/enterpriseops_gym/enterpriseops_gym_gold.json

3. ``--run``: placeholder for a live driver against the gym MCP servers. Running
   the real benchmark requires the containerized environment, MCP endpoints, and
   model credentials documented in the upstream repository; wire your rollout
   loop into ``run_live`` and emit records via ``record_from_result``.

After producing ``enterpriseops_gym_gold.json``, regenerate the committed
samples with::

    python datasets/enterpriseops_gym/make_samples.py --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATASET_DIR / "enterpriseops_gym_gold.json"
HF_REPO = "ServiceNow-AI/EnterpriseOps-Gym"
DOMAINS = ["calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid"]

LOOKUP_VERBS = {"get", "list", "query", "search", "find", "fetch", "read", "check", "lookup"}
CREATE_VERBS = {"create", "add", "insert", "new", "schedule", "send", "post", "upload"}
UPDATE_VERBS = {"update", "edit", "set", "modify", "assign", "move", "change", "share", "grant"}
DELETE_VERBS = {"delete", "remove", "clear", "cancel", "revoke", "close"}

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _redact_tokens(config_json: str) -> str:
    """Blank out example access tokens in a gym_servers_config JSON string."""
    try:
        config = json.loads(config_json)
    except (TypeError, json.JSONDecodeError):
        return config_json
    for server in config if isinstance(config, list) else []:
        context = server.get("context") if isinstance(server, dict) else None
        if isinstance(context, dict):
            for key in list(context):
                if "token" in key.lower() or "secret" in key.lower() or "key" in key.lower():
                    context[key] = "<redacted>"
    return json.dumps(config, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Illustrative trajectory construction from published tasks
# ---------------------------------------------------------------------------


def _prompt_context(user_prompt: str) -> dict[str, list[str]]:
    """Pull representative entities out of a user prompt for tool arguments."""
    quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,60})[\"“”'‘’]", user_prompt)
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", user_prompt)
    dates = re.findall(
        r"(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}",
        user_prompt,
    )
    return {"quoted": quoted, "emails": emails, "dates": dates}


def _representative_arguments(tool: str, ctx: dict[str, list[str]]) -> dict[str, Any]:
    name = tool.lower()
    verb = name.split("_", 1)[0]
    quoted, emails, dates = ctx["quoted"], ctx["emails"], ctx["dates"]
    args: dict[str, Any] = {}

    if verb in LOOKUP_VERBS:
        if "freebusy" in name and dates:
            args["time_min"] = dates[0]
        elif emails and ("user" in name or "member" in name or "people" in name):
            args["email"] = emails[0]
        elif quoted:
            args["query"] = quoted[0]
    elif verb in CREATE_VERBS:
        if quoted:
            args["name"] = quoted[0]
        if ("event" in name or "meeting" in name) and dates:
            args["start"] = dates[0]
        if "acl" in name or "permission" in name or "share" in name:
            if emails:
                args["email"] = emails[0]
            args["role"] = "writer"
        if emails and "email" not in args and ("invite" in name or "send" in name):
            args["to"] = emails[0]
    elif verb in UPDATE_VERBS:
        if quoted:
            args["name"] = quoted[0]
        if emails:
            args["assignee"] = emails[0]
    elif verb in DELETE_VERBS:
        if quoted:
            args["name"] = quoted[0]

    if not args and emails:
        args["email"] = emails[0]
    return args


def _representative_observation(tool: str, domain: str, ctx: dict[str, list[str]]) -> str:
    verb = tool.lower().split("_", 1)[0]
    example_id = f"{domain}_0001"
    if verb in LOOKUP_VERBS:
        record = {"id": example_id}
        if ctx["quoted"]:
            record["name"] = ctx["quoted"][0]
        if ctx["emails"]:
            record["email"] = ctx["emails"][0]
        return json.dumps([record], ensure_ascii=False)
    return json.dumps({"status": "success", "id": example_id}, ensure_ascii=False)


def _final_message(user_prompt: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", user_prompt.strip())[0]
    if len(first) > 160:
        first = first[:157] + "..."
    return f"I have completed the requested workflow ({first}). All required actions executed."


def build_illustrative_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    """Turn one published EnterpriseOps-Gym task into an illustrative ADP record."""
    domain = task.get("domain", "unknown")
    system_prompt = task.get("system_prompt", "")
    user_prompt = task.get("user_prompt", "")
    selected_tools = [str(t) for t in (task.get("selected_tools") or [])]
    ctx = _prompt_context(user_prompt)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for i, tool in enumerate(selected_tools, start=1):
        call_id = f"call_{i}"
        arguments = _representative_arguments(tool, ctx)
        messages.append(
            {
                "role": "assistant",
                "content": f"Calling `{tool}` to progress the task.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool,
                "content": _representative_observation(tool, domain, ctx),
            }
        )
    messages.append({"role": "assistant", "content": _final_message(user_prompt)})

    verifiers = task.get("verifiers")
    if isinstance(verifiers, str):
        try:
            verifiers = json.loads(verifiers)
        except json.JSONDecodeError:
            pass

    record: dict[str, Any] = {
        "id": task.get("task_id"),
        "task": domain,
        "domain": domain,
        "answer": None,
        "reward": None,
        "selected_tools": selected_tools,
        "verifiers": verifiers,
        "source": HF_REPO,
        "note": (
            "Illustrative reference trajectory: task id, domain, system/user "
            "prompts, oracle tool list, and SQL verifiers are the real published "
            "EnterpriseOps-Gym task; tool-call arguments and observations are "
            "representative. Regenerate authentic rollouts with --from-results."
        ),
        "messages": messages,
    }
    gym_config = task.get("gym_servers_config")
    if isinstance(gym_config, str) and gym_config:
        record["gym_servers_config"] = _redact_tokens(gym_config)
    return record


def convert_from_hf(config: str, domains: list[str], per_domain: int) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional deps
        raise SystemExit(
            "--from-hf needs `pandas` and `huggingface_hub`. Install the commented "
            "extras in datasets/enterpriseops_gym/requirements.txt."
        ) from exc

    records: list[dict[str, Any]] = []
    for domain in domains:
        path = hf_hub_download(
            HF_REPO, f"{config}/{domain}-00000-of-00001.parquet", repo_type="dataset"
        )
        frame = pd.read_parquet(path)
        for _, row in frame.head(per_domain).iterrows():
            task = row.to_dict()
            task["selected_tools"] = list(task.get("selected_tools", []))
            records.append(build_illustrative_trajectory(task))
    return records


# ---------------------------------------------------------------------------
# Authentic trajectories from EnterpriseOps-Gym evaluate.py result logs
# ---------------------------------------------------------------------------


def _valid_trajectory(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    return "assistant" in roles


def record_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Reshape one EnterpriseOps-Gym result entry into an ADP raw record."""
    messages = (
        result.get("messages")
        or result.get("trajectory")
        or result.get("traj")
        or result.get("conversation")
    )
    if not _valid_trajectory(messages):
        return None
    reward = result.get("score", result.get("reward", result.get("success")))
    if isinstance(reward, bool):
        reward = 1.0 if reward else 0.0
    return {
        "id": result.get("task_id", result.get("id")),
        "task": result.get("domain", result.get("task")),
        "domain": result.get("domain"),
        "answer": result.get("answer"),
        "reward": reward,
        "verifiers": result.get("verifiers"),
        "source": HF_REPO,
        "messages": messages,
    }


def _iter_result_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
            yield from sorted(path.rglob("*.jsonl"))
        elif path.exists():
            yield path


def _iter_results(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    data = json.loads(text)
    results = data if isinstance(data, list) else data.get("results", [data])
    yield from (r for r in results if isinstance(r, dict))


def convert_from_results(
    paths: list[Path], *, keep_all: bool, min_reward: float
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file in _iter_result_files(paths):
        try:
            for result in _iter_results(file):
                record = record_from_result(result)
                if record is None:
                    continue
                reward = record.get("reward")
                if not keep_all and not (isinstance(reward, (int, float)) and reward >= min_reward):
                    continue
                records.append(record)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {file}: {exc}", file=sys.stderr)
    return records


def run_live(args: argparse.Namespace) -> list[dict[str, Any]]:  # pragma: no cover
    raise SystemExit(
        "--run is a placeholder. Running EnterpriseOps-Gym live requires the "
        "containerized MCP servers and model credentials from the upstream repo. "
        "Run the benchmark's evaluate.py and use --from-results, or implement your "
        "rollout loop here and emit records via record_from_result()."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-hf",
        action="store_true",
        help="Build illustrative trajectories from the published HF tasks.",
    )
    mode.add_argument(
        "--from-results",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="EnterpriseOps-Gym evaluate.py result file(s)/dir(s) to convert.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Drive the gym live (placeholder; needs containers + credentials).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--min-reward", type=float, default=1.0)
    # --from-hf options.
    parser.add_argument("--config", default="oracle", help="HF config: oracle|plus_5_tools|...")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domain splits to pull.")
    parser.add_argument("--per-domain", type=int, default=1, help="Tasks per domain.")
    args = parser.parse_args()

    if args.run:
        records = run_live(args)
    elif args.from_results:
        records = convert_from_results(
            args.from_results, keep_all=args.keep_all, min_reward=args.min_reward
        )
    else:
        records = convert_from_hf(args.config, args.domains, args.per_domain)

    if not records:
        raise SystemExit("No trajectories produced. Check the inputs / reward filter.")

    args.output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
