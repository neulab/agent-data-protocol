from typing import List, Literal, Optional


def str_replace_editor(
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
) -> None:
    """View, create, and edit files with the OpenHands editor tool."""
    pass
