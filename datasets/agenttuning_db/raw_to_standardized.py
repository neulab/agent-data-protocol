import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


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
        TextObservation(content=sys_sql_subs, source="system"),
        TextObservation(content="Ok? Understood?", source="user"),
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
                source=step["role"] if step["role"] != "system" else "user",
            ),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)
    content = []

    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions
    if isinstance(content[-1], MessageAction) and "<finish>" not in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    # Handle finish actions for code actions: Should not have a code action without an observation, skip
    if isinstance(content[-1], CodeAction):
        continue

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
