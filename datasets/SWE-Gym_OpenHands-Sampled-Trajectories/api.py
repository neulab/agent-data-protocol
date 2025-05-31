def str_replace_editor(command: str, path: str, old_str: str = None, new_str: str = None, file_text: str = None, insert_line: int = None, view_range: list = None) -> None:
    """Edit files using string replacement.

    Args:
    ----
        command (str): The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
        path (str): Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.
        old_str (str, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
        new_str (str, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        file_text (str, optional): Required parameter of `create` command, with the content of the file to be created.
        insert_line (int, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        view_range (list, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.

    """
    pass

def execute_bash(command: str, is_input: bool = False, timeout: int = None) -> None:
    """Execute a bash command in the terminal.

    Args:
    ----
        command (str): The bash command to execute.
        is_input (bool, optional): If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.
        timeout (int, optional): Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.

    """
    pass

def finish(message: str = "", task_completed: bool = True) -> None:
    """Signals the completion of the current task or conversation.

    Args:
    ----
        message (str): Final message to send to the user.
        task_completed (bool): Whether you have completed the task.

    """
    pass