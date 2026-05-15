def search(query: str, topn: int = 10, source: str | None = None) -> None:
    """Search for information related to a query and return ranked results."""
    pass


def open(
    cursor: int = -1,
    id: int | str = -1,
    loc: int = -1,
    num_lines: int = -1,
    source: str | None = None,
    view_source: bool = False,
) -> None:
    """Open a search result, URL, local resource, or cursor location."""
    pass


def find(pattern: str, cursor: int = -1) -> None:
    """Find exact matches of a pattern in the current page or cursor."""
    pass
