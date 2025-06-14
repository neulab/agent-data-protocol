from typing import List, Literal, Union


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


def goto(url: str):
    """Navigate to a URL.

    Args:
    ----
        url (str): The target URL to visit.

    Example: goto('http://www.example.com')

    """
    pass


def go_back():
    """Navigate to the previous page in history.

    Example: go_back()
    """
    pass


def go_forward():
    """Navigate to the next page in history.

    Example: go_forward()
    """
    pass


def noop(wait_ms: float = 1000):
    """Do nothing and optionally wait.

    Args:
    ----
        wait_ms (float, optional): Time to wait in milliseconds (default is 1000).

    Example: noop()
             noop(500)

    """
    pass


def scroll(delta_x: float, delta_y: float):
    """Scroll horizontally and/or vertically in pixels.

    Args:
    ----
        delta_x (float): Horizontal scroll in pixels.
        delta_y (float): Vertical scroll in pixels.

    """
    pass


def fill(bid: str, value: str):
    """Fill a form field with text.

    Args:
    ----
        bid (str): Element ID to fill.
        value (str): Text to input.

    """
    pass


def select_option(bid: str, options: Union[str, List[str]]):
    """Select one or more options in a dropdown/select element.

    Args:
    ----
        bid (str): The element ID.
        options (str or list[str]): One or more option values or labels to select.

    """
    pass


def click(
    bid: str,
    button: Literal["left", "middle", "right"] = "left",
    modifiers: List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = [],
):
    """Click an element.

    Args:
    ----
        bid (str): Element ID to click.
        button (str, optional): Mouse button to use.
        modifiers (list, optional): List of modifier keys.

    """
    pass


def dblclick(
    bid: str,
    button: Literal["left", "middle", "right"] = "left",
    modifiers: List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = [],
):
    """Double-click an element.

    Args:
    ----
        bid (str): Element ID to double-click.
        button (str, optional): Mouse button to use.
        modifiers (list, optional): List of modifier keys.

    """
    pass


def hover(bid: str):
    """Hover over an element.

    Args:
    ----
        bid (str): Element ID to hover on.

    Example: hover('b8')

    """
    pass


def press(bid: str, key_comb: str):
    """Press a key combination on a focused element.

    Args:
    ----
        bid (str): Element ID to focus.
        key_comb (str): Key combination to simulate.

    """
    pass


def focus(bid: str):
    """Focus the specified element.

    Args:
    ----
        bid (str): Element ID to focus.

    Example: focus('b455')

    """
    pass


def clear(bid: str):
    """Clear the value of an input field.

    Args:
    ----
        bid (str): Element ID to clear.

    Example: clear('996')

    """
    pass


def drag_and_drop(from_bid: str, to_bid: str):
    """Perform drag-and-drop from one element to another.

    Args:
    ----
        from_bid (str): Source element ID.
        to_bid (str): Target element ID.

    """
    pass


def upload_file(bid: str, file: Union[str, List[str]]):
    """Upload file(s) via a file input element.

    Args:
    ----
        bid (str): Element ID to click for upload.
        file (str or list[str]): Path(s) to file(s) to upload.

    """
    pass


def send_msg_to_user(msg: str):
    """Send a message to the user.

    Args:
    ----
        msg (str): The message content to send.

    Example: send_msg_to_user('The number of stars for the React repository on GitHub is 225k.')

    """
    pass
