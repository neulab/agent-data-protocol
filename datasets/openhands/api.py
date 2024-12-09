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


def browse_interactive(browser_actions: str):
    """Interact with the browser using Python code (BrowserGym action space).

    Use this only if the command string contains multiple browser actions.
    Otherwise, use the specific browser action functions below.

    Args:
    ----
        browser_actions (str): The python code to execute.

    """
    pass


# all browsergym actions supported by OpenHands
# https://github.com/All-Hands-AI/OpenHands/blob/2f11634ccaeb3aa38893a994d53ff280bbf485d0/openhands/agenthub/codeact_agent/function_calling.py#L316


def goto(url: str):
    """Navigate to a url.

    Args:
    ----
        url (str): The url to navigate to.

    Examples:
    --------
        goto('http://www.example.com')

    """
    pass


def go_back():
    """Navigate to the previous page in history.

    Examples
    --------
        go_back()

    """
    pass


def go_forward():
    """Navigate to the next page in history.

    Examples
    --------
        go_forward()

    """
    pass


def noop(wait_ms: float = 1000):
    """Do nothing, and optionally wait for the given time (in milliseconds).

    You can use this to get the current page content and/or wait for the page to load.

    Args:
    ----
        wait_ms (float): The time to wait in milliseconds.

    Examples:
    --------
        noop()

        noop(500)

    """
    pass


def scroll(delta_x: float, delta_y: float):
    """Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.

    Args:
    ----
        delta_x (float): The amount to scroll horizontally.
        delta_y (float): The amount to scroll vertically.

    Examples:
    --------
        scroll(0, 200)

        scroll(-50.2, -100.5)

    """
    pass


def fill(bid: str, value: str):
    r"""Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.

    Args:
    ----
        bid (str): The id of the form field.
        value (str): The value to fill in the form field.

    Examples:
    --------
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

    """
    pass


def select_option(bid: str, options: str | list[str]):
    """Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.

    Args:
    ----
        bid (str): The id of the select element.
        options (str | list[str]): The option(s) to select.

    Examples:
    --------
        select_option('a48', 'blue')

        select_option('c48', ['red', 'green', 'blue'])

    """
    pass


def click(bid: str, button: str = "left", modifiers: list[str] = []):
    """Click an element.

    Args:
    ----
        bid (str): The id of the element to click.
        button (str): The mouse button to use ('left', 'middle', 'right').
        modifiers (list[str]): The modifier keys to hold during the click.

    Examples:
    --------
        click('a51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

    """
    pass


def dblclick(bid: str, button: str = "left", modifiers: list[str] = []):
    """Double click an element.

    Args:
    ----
        bid (str): The id of the element to double click.
        button (str): The mouse button to use ('left', 'middle', 'right').
        modifiers (list[str]): The modifier keys to hold during the double click.

    Examples:
    --------
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

    """
    pass


def hover(bid: str):
    """Hover over an element.

    Args:
    ----
        bid (str): The id of the element to hover over.

    Examples:
    --------
        hover('b8')

    """
    pass


def press(bid: str, key_comb: str):
    """Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta, ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux and to Meta on macOS.

    Args:
    ----
        bid (str): The id of the element to focus.
        key_comb (str): The key combination to press.

    Examples:
    --------
        press('88', 'Backspace')

        press('a26', 'ControlOrMeta+a')

        press('a61', 'Meta+Shift+t')

    """
    pass


def focus(bid: str):
    """Focus the matching element.

    Args:
    ----
        bid (str): The id of the element to focus.

    Examples:
    --------
        focus('b455')

    """
    pass


def clear(bid: str):
    """Clear the input field.

    Args:
    ----
        bid (str): The id of the input field to clear.

    Examples:
    --------
        clear('996')

    """
    pass


def drag_and_drop(from_bid: str, to_bid: str):
    """Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.

    Args:
    ----
        from_bid (str): The id of the element to drag.
        to_bid (str): The id of the element to drop.

    Examples:
    --------
        drag_and_drop('56', '498')

    """
    pass


def upload_file(bid: str, file: str | list[str]):
    """Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.

    Args:
    ----
        bid (str): The id of the element to click.
        file (str | list[str]): The file(s) to upload.

    Examples:
    --------
        upload_file('572', '/home/user/my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

    """
    pass


def send_msg_to_user(msg: str):
    """Send a message to the user.

    Args:
    ----
        msg (str): The message to send.

    Examples:
    --------
        send_msg_to_user('Hello, World!')

    """
    pass


def get():
    """Get the current page content.

    Examples
    --------
        get()

    """
    pass
