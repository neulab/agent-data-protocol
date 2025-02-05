from typing import List, Optional


def execute_bash(command: str):
    """Execute a bash command in the terminal.

    * Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
    * Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
    * Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.


    Args:
    ----
        command (str): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.

    """
    pass


def str_replace_editor(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    insert_line: Optional[int] = None,
    new_str: Optional[str] = None,
    old_str: Optional[str] = None,
    view_range: Optional[List[int]] = None,
):
    """Use the str_replace_editor custom editing tool for viewing, creating and editing files.

    * State is persistent across command calls and discussions with the user
    * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
    * The `undo_edit` command will revert the last edit made to the file at `path`.

    Notes for using the `str_replace` command:
    * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
    * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
    * The `new_str` parameter should contain the edited lines that should replace the `old_str`

    Args:
    ----
        command (str): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
        path (str): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
        file_text (str): Required parameter of `create` command, with the content of the file to be created.
        insert_line (int): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        new_str (str): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        old_str (str): Required parameter of `str_replace` command containing the string in `path` to replace.
        view_range (List[int], optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.

    """
    pass


def finish():
    """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task."""
    pass