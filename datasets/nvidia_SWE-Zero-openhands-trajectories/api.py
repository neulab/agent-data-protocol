def think(thought: str) -> None:
    """Log reasoning without changing the environment.

    Args:
    ----
        thought: The reasoning text to record.

    """
    pass


def str_replace_editor(
    command: str,
    path: str,
    file_text: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
    view_range: list[int] | None = None,
) -> None:
    """View, create, and edit files.

    Args:
    ----
        command: The editor command to run.
        path: Absolute path to the target file or directory.
        file_text: Content for create operations.
        old_str: Text to replace for str_replace operations.
        new_str: Replacement text or insertion text.
        insert_line: Line after which to insert text.
        view_range: Optional line range for view operations.

    """
    pass
