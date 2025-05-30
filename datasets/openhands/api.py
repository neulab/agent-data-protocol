def run(command: str):
    """Execute a bash command.

    Args:
    ----
        command (str): The bash command to execute.

    """
    pass


def run_ipython(code: str, kernel_init_code: str):
    """Execute an IPython cell.

    Args:
    ----
        code (str): The python code to execute.
        kernel_init_code (str): The kernel initialization code.

    """
    pass


#
def initialize(env_vars: dict):
    """Set environment variables.

    Args:
    ----
        env_vars (dict): The environment variables.

    """
    pass


def change_agent_state(agent_state: str):
    """Change the agent state.

    Args:
    ----
        agent_state (str): The new agent state.

    """
    pass


def delegate_to_agent(agent: str, task: str):
    """Delegate the task to the BrowsingAgent.

    Args:
    ----
        agent (str): The agent to delegate the task to.
        task (str): task description.

    """
    pass


def delegate_to_CrawlAgent(task: str, link: str):
    """Delegate the task to the CrawlAgent.

    Args:
    ----
        task (str): task description.
        link (str): the link to crawl.

    """
    pass


def delegate_to_RagAgent(task: str, query: str):
    """Delegate the task to the RagAgent.

    Args:
    ----
        task (str): task description.
        query (str): the query to search.

    """
    pass


def finish(output: str):
    """Finish the task.

    Args:
    ----
        output (str): The output of the task.

    """
    pass


def add_task(goal: str):
    """Add a task to the task planner.

    Args:
    ----
        goal (str): The goal of the task.

    """
    pass


def modify_task(task_id: str, state: str):
    """Modify the state of a task.

    Args:
    ----
        task_id (str): The id of the task.
        state (str): The new state of the task.

    """
    pass


def save_plan(plan: list[str]):
    """Save the plan.

    Args:
    ----
        plan (list[str]): The plan to save.

    """
    pass


def task_plan(task: str, plan: str):
    """Plan a task.

    Args:
    ----
        task (str): The task to plan.
        plan (list[str]): The plan.

    """
    pass


def edit(path: str, content: str, start: int, end: int):
    """Edit a file.

    Args:
    ----
        path (str): The path of the file.
        content (str): The new content of the file.
        start (int): The start position of the edit.
        end (int): The end position of the edit.

    """
    pass


def read(path: str, start: int, end: int):
    """Read a file.

    Args:
    ----
        path (str): The path of the file.
        start (int): The start position of the read.
        end (int): The end position of the read.

    """
    pass


def crawl(link: str):
    """Crawl a webpage.

    Args:
    ----
        link (str): The link to crawl.

    """
    pass


def rag_search(query: str):
    """Search using the RAG model.

    Args:
    ----
        query (str): The query to search.

    """
    pass