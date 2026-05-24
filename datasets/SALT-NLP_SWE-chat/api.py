from typing import Any


def str_replace_editor(
    command: str,
    path: str,
    file_text: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
    view_range: list | None = None,
) -> None:
    """View, create, and edit files with a custom editing tool.

    Args:
    ----
        command: One of `view`, `create`, `str_replace`, `insert`, or `undo_edit`.
        path: Absolute path to the target file or directory.
        file_text: Content for `create` commands.
        old_str: Existing text for `str_replace` commands.
        new_str: Replacement text or inserted text.
        insert_line: Line after which to insert text.
        view_range: Optional `[start_line, end_line]` range to view.

    """
    pass


def think(thought: str) -> None:
    """Record a private reasoning step.

    Args:
    ----
        thought: The model's reasoning trace.

    """
    pass


def generic_tool(tool_name: str, tool_input: dict[str, Any], content: str | None = None) -> None:
    """Represent a source-specific coding-agent tool call.

    Args:
    ----
        tool_name: Original SWE-chat tool name.
        tool_input: Parsed tool input parameters.
        content: Raw tool-call content when no structured input is available.

    """
    pass
