from typing import List, Optional


def str_replace_editor(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
) -> None:
    """View, create, and edit files with the OpenHands string replacement editor.

    Args:
    ----
        command: Editor command such as view, create, str_replace, insert, or undo_edit.
        path: Absolute path to the file or directory to operate on.
        file_text: Content to write when creating a file.
        old_str: Existing text to replace.
        new_str: Replacement text or text to insert.
        insert_line: Line number after which to insert new_str.
        view_range: Optional 1-indexed inclusive line range for view commands.

    """
    pass


def finish(message: str, task_completed: str) -> None:
    """Finish the task.

    Args:
    ----
        message: Final response to the user.
        task_completed: Whether the task was completed successfully.

    """
    pass
