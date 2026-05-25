import ast
import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

SQL_STATEMENT_RE = re.compile(
    r"^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|REPLACE|WITH)\b",
    re.IGNORECASE,
)


def parse_final_answer(answer: str) -> object:
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return ast.literal_eval(answer)


def extract_sql_answer(answer: str) -> str | None:
    try:
        parsed_answer = parse_final_answer(answer)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed_answer, str):
        candidates = [parsed_answer]
    elif isinstance(parsed_answer, list):
        candidates = parsed_answer
    else:
        return None

    if len(candidates) != 1 or not isinstance(candidates[0], str):
        return None

    sql = candidates[0].strip()
    if SQL_STATEMENT_RE.match(sql):
        return sql
    return None


def convert_system(system_regex: re.Match[str]) -> list[Observation]:
    assert re.search(r"Final Answer: \[\"ANSWER1\", \"ANSWER2\", ...\]", system_regex.group(1))
    answer_subs = re.sub(
        r"Final Answer: \[\"ANSWER1\", \"ANSWER2\", ...\]",
        r"<solution> [\"ANSWER1\", \"ANSWER2\", ...] </solution>",
        system_regex.group(1),
    )
    sys_sql_regex = re.match(r"(.*)\n```sql\n(.*)\n```\n", answer_subs, re.DOTALL)
    sys_sql_subs = re.sub(
        r"```sql\n(.*)\n```",
        f"<execute_mysql>\n{sys_sql_regex.group(2)}\n</excute_mysql>",
        answer_subs,
    )
    sys_sql_subs = sys_sql_subs.replace("Action: Operation", "").replace("Action: Answer", "")
    return [
        TextObservation(content=sys_sql_subs + "\n\n" + "Ok? Understood?", source="user"),
    ]


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    # parse system prompt
    system_regex = re.match(
        r"(I will ask you a question,.*)\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    if system_regex:
        return convert_system(system_regex)

    sql_act_regex = re.match(
        r"(.*)Action: Operation\n```sql\n(.*)\n```", step["content"], re.DOTALL
    )
    sql_solution_regex = re.match(
        r"(.*)Action: Answer\nFinal Answer: (.*)", step["content"], re.DOTALL
    )

    if sql_act_regex:
        return [
            CodeAction(
                language="mysql",
                content=sql_act_regex.group(2),
                description=sql_act_regex.group(1),
            ),
        ]
    elif sql_solution_regex:
        sql_answer = extract_sql_answer(sql_solution_regex.group(2))
        if sql_answer is not None:
            return [
                CodeAction(
                    language="mysql",
                    content=sql_answer,
                    description=sql_solution_regex.group(1),
                ),
            ]
        return [
            MessageAction(
                content=f"<solution> {sql_solution_regex.group(2)} </solution>",
                description=sql_solution_regex.group(1),
            ),
        ]
    elif (
        "ok." == step["content"].strip().lower()
        or "ok. i'll follow" in step["content"].strip().lower()
    ):
        return [
            MessageAction(content=step["content"]),
        ]

    else:
        if step["role"] == "assistant" and "Final Answer:" in step["content"]:
            answer_extract_regex = re.search(r"Final Answer:\s*(.*)", step["content"], re.DOTALL)
            step["content"] = re.sub(
                r"Final Answer:\s*(.*)",
                r"ACTION: <solution>" + f" {answer_extract_regex.group(1)} </solution>",
                step["content"],
            )

        return [
            TextObservation(
                content=step["content"]
                .replace("Thought:", "THOUGHT:")
                .replace("Action:", "ACTION:")
                .replace("Observation:", "OBSERVATION:"),
                source=step["role"] if step["role"] != "system" else "environment",
            ),
        ]


def convert_trajectory(raw_data: dict) -> Trajectory:
    content = []

    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions
    if isinstance(content[-1], MessageAction) and "<finish>" not in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    return Trajectory(
        id=raw_data["id"],
        content=content,
    )


def main() -> None:
    for line in sys.stdin:
        raw_data = json.loads(line)
        standardize_data = convert_trajectory(raw_data)
        print(standardize_data.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
