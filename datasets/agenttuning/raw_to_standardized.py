import json
import sys
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    system_regex = re.match(
        r"(You are an assistant.*\n\nNow, my problem is:|I will ask you a question,.*|You are web shopping.*|Interact with a household to solve a task.*|You are an agent that answers questions.*|Now, I will start a new problem in a new OS. My problem is:)\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    code_act_regex = re.match(r"Think: (.*)\n\nAct: (.*)", step["content"], re.DOTALL)
    code_obs_regex = re.match(
        r"The output of the OS:\n(.*)", step["content"], re.DOTALL
    )

    sql_act_regex = re.match(r"(.*)Action: Operation\n```sql\n(.*)\n```", step["content"], re.DOTALL)
    sql_solution_regex = re.match(r"(.*)Action: Answer\nFinal Answer: (.*)", step["content"], re.DOTALL)

    if system_regex:
        """
        The system regex has to be handled in a special way because it contains multiple actions.
        Kind of manually parsing and setting the execution format and structure with expicit tags.
        std_to_sft.py will handle tags for the code and user actions.
        """
        if "You are an assistant" in system_regex.group(1):
            assert re.search(r'\"bash\"', system_regex.group(1), re.DOTALL)
            assert re.search(r'Act: finish', system_regex.group(1), re.DOTALL)
            assert re.search(r'```bash\n(.*?)\n```', system_regex.group(1), re.DOTALL)
            assert re.search(r'answer(.*)', system_regex.group(1), re.DOTALL)

            bash_subs = re.sub(r'\"bash\"', '<execute_bash>', system_regex.group(1))
            bash_subs = re.sub(r'Act: bash\n\n```bash\n(.*?)\n```', r'ACTION: <execute_bash>\n# put your bash code here\n</execute_bash>', bash_subs, flags=re.DOTALL)

            finish_subs = re.sub(r'\"finish\"', r'exit', bash_subs)
            finish_subs = re.sub(r'Act: finish', 'ACTION: <execute_bash>\nexit\n</execute_bash>', finish_subs)

            answer_subs = re.sub(r'\"answer\"', r'<solution>', finish_subs)
            answer_subs = re.sub(r'Act: answer\(.*\)', r'ACTION: <solution> Your solution here </solution>', answer_subs)

            return [
                TextObservation(content = answer_subs.replace("Think:", "THOUGHT:").replace("Act:", "ACTION:"), source="system"),
                TextObservation(content=system_regex.group(2).strip(), source="user"),
            ]
        
        elif "I will ask you a question" in system_regex.group(1):
            assert re.search(r'Final Answer: \[\"ANSWER1\", \"ANSWER2\", ...\]', system_regex.group(1))

            answer_subs = re.sub(r"Final Answer: \[\"ANSWER1\", \"ANSWER2\", ...\]", r"<solution> [\"ANSWER1\", \"ANSWER2\", ...] </solution>", system_regex.group(1))
            sys_sql_regex = re.match(r'(.*)\n```sql\n(.*)\n```\n', answer_subs, re.DOTALL)
            # print(sys_sql_regex, file = sys.stderr)
            sys_sql_subs = re.sub(r"```sql\n(.*)\n```", f"<execute_mysql>\n{sys_sql_regex.group(2)}\n</excute_mysql>", answer_subs)

            return [
                TextObservation(content=sys_sql_subs, source="system"),
            ]
        elif "You are web shopping" in system_regex.group(1):
            webshop_sys_msg = system_regex.group(1) + "\nclick[something]"
            # print(webshop_sys_msg, file=sys.stderr)
            assert re.search(r'\nThought:\nI think ... \n\nAction: \n', webshop_sys_msg)

            thought_subs = re.sub(r'\nThought:\nI think ... \n\nAction: \n', r'\nTHOUGHT:\nI think ... \n\nACTION:\n', webshop_sys_msg)

            return [
                TextObservation(content=thought_subs, source="system"),
            ]
        
        elif "You are an agent that answers questions" in system_regex.group(1):
            assert re.search(r'Final Answer: #3', system_regex.group(1), re.DOTALL)

            answer_sub = re.sub(r'Final Answer: #3', r'ACTION: <solution> #3 </solution>', system_regex.group(1))
            return [
                TextObservation(content=answer_sub, source="system"),
            ]
        
        elif "Interact with a household to solve a task" in system_regex.group(1):
            
            return [
                TextObservation(content=system_regex.group(1), source="system"),
            ] 
        
        return [
            TextObservation(content=system_regex.group(1), source="system"),
            TextObservation(content=system_regex.group(2), source="user"),
        ]
    
    # Special case for SQL
    elif "I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question." in step["content"]:
        return [
            TextObservation(content=step["content"], source="user"),
        ]
    # Special case for alfworld
    elif "Interact with a household to solve a task." in step["content"]:
        return [
            TextObservation(content=step["content"], source="user").replace('Thought:', 'THOUGHT:').replace('Action:', 'ACTION:'),
        ]

    elif code_act_regex:
        bash_extract_regex = re.match(
            r"bash\n\n```bash\n(.*)\n```|bash \n\n```bash\n(.*)\n```|bash\n  \n```bash\n(.*)\n```", code_act_regex.group(2), re.DOTALL
        )
        answer_extract_regex = re.match(
            r"answer\((.*)\)", code_act_regex.group(2), re.DOTALL
        )
        finish_extract_regex = re.match(
            r"finish", code_act_regex.group(2), re.DOTALL
        )
        if bash_extract_regex:
            return [
                CodeAction(
                    language="bash",
                    content=bash_extract_regex.group(1) or bash_extract_regex.group(2) or bash_extract_regex.group(3),
                    description=code_act_regex.group(1),
                ),
            ]
        elif answer_extract_regex:
            return [
                MessageAction(
                    content=f"<solution> {answer_extract_regex.group(1)} </solution>",
                    description=code_act_regex.group(1),
                ),
            ]
        elif finish_extract_regex:
            return [
                MessageAction(
                    # content=finish_extract_regex.group(0),
                    content="<execute_bash>\nexit\n</execute_bash>",
                    description=code_act_regex.group(1)
                ),
            ]
        else:
            raise ValueError(
                "Could not extract code from code action in"
                f" {json.dumps(step, indent=2)}"
            )
    elif sql_act_regex:
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
                description=sql_solution_regex.group(1)
            ),
        ]
    elif code_obs_regex:
        return [
            TextObservation(content=code_obs_regex.group(1), source="os"),
        ]
    
    elif "ok" == step["content"].strip().lower() or "ok. i'll follow" in step["content"].strip().lower():
         return [
             MessageAction(content=step["content"]),
         ]
    
    else:
        if step["role"]=="assistant" and "Final Answer:" in step["content"]:
            answer_extract_regex = re.search(r"Final Answer:\s*(.*)", step["content"], re.DOTALL)
            step["content"] = re.sub(r"Final Answer:\s*(.*)", r"ACTION: <solution>"+f" {answer_extract_regex.group(1)} </solution>", step["content"])
        
        return [
            TextObservation(content=step["content"].replace('Thought:', 'THOUGHT:').replace('Action:', 'ACTION:').replace("Observation:", "OBSERVATION:"),
                            source= step["role"] or "user"),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)
    web_system_msg = '''You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.'''

    content = []
    if "mind2web" in raw_data["id"]:
        content.extend([
            TextObservation(content=web_system_msg, source="system"),
        ])
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

   # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
