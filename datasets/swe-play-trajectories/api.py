"""API definitions for SWE-Play-trajectories dataset.

These functions represent the OpenHands tools used as ApiAction in the dataset.
Note: execute_bash and execute_ipython_cell are converted to CodeAction.
"""

from typing import List, Literal, Optional


def str_replace_editor(
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
) -> str:
    """View, create, and edit files.

    Args:
        command (str): The command to execute. One of: view, create, str_replace, insert, undo_edit.
        path (str): The absolute path to the file or directory.
        file_text (str, optional): Required for 'create' command. The content of the file to create.
        old_str (str, optional): Required for 'str_replace' command. The string to replace.
        new_str (str, optional): Required for 'str_replace' command. The new string to insert.
        insert_line (int, optional): Required for 'insert' command. Line number to insert at.
        view_range (list, optional): Optional for 'view' command. [start_line, end_line] to view.

    Returns:
        str: The result of the operation.

    Example:
        str_replace_editor(command="view", path="/workspace/file.py")
        str_replace_editor(command="create", path="/workspace/new.py", file_text="print('hello')")
        str_replace_editor(command="str_replace", path="/workspace/file.py", old_str="old", new_str="new")

    """
    pass


def think(thought: str) -> None:
    """Use this tool to think through a problem step by step.

    Args:
        thought (str): The thought or reasoning to record.

    Example:
        think("I need to first understand the existing code structure before making changes.")

    """
    pass


def finish(message: str, task_completed: Literal["true", "false"] = "true") -> None:
    """Finish the task and provide a final message.

    Args:
        message (str): A summary message about the task completion.
        task_completed (str): Whether the task was successfully completed. "true" or "false".

    Example:
        finish(message="Successfully implemented the feature.", task_completed="true")

    """
    pass
