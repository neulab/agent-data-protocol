def execute_bash(command: str, is_input: bool = False, timeout: int = None) -> None:
    """Execute a bash command in the terminal.

    Args:
    ----
        command (str): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
        is_input (bool): If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.
        timeout (int): Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.

    """
    pass


def str_replace_editor(
    command: str,
    path: str,
    file_text: str = None,
    old_str: str = None,
    new_str: str = None,
    insert_line: int = None,
    view_range: list = None,
) -> None:
    """View, create, and edit files with this custom editing tool.

    Args:
    ----
        command (str): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
        path (str): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
        file_text (str): Required parameter of `create` command, with the content of the file to be created.
        old_str (str): Required parameter of `str_replace` command containing the string in `path` to replace.
        new_str (str): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        insert_line (int): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        view_range (list): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.

    """
    pass


def finish(message: str, task_completed: bool = True) -> None:
    """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.

    Args:
    ----
        message (str): Final message to send to the user.
        task_completed (bool): Whether you have completed the task.

    """
    pass
