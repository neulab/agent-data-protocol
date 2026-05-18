from typing import Optional


def str_replace_editor(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[list[int]] = None,
):
    """View, create, and edit files with a custom editing tool.

    Args:
    ----
        command: Editor command. Allowed values include view, create, str_replace, insert, and undo_edit.
        path: Absolute path to a file or directory.
        file_text: File content for create commands.
        old_str: Exact string to replace for str_replace commands.
        new_str: Replacement or inserted string.
        insert_line: Line number after which to insert new_str.
        view_range: Optional line range to view.

    """
    pass
