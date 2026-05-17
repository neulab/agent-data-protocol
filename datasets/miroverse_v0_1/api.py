from typing import Any


def use_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any] | str):
    """Fallback MCP wrapper for rows whose XML arguments cannot be parsed.

    Args:
        server_name: Name of the MCP server that provides the tool.
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as a JSON object, or the raw argument string when parsing fails.

    """
    return None


def browsing_agent__search_and_browse(subtask: str) -> dict:
    """Search and browse the web for a clearly defined factual subtask."""
    return {}


def tool_google_search__google_search(
    q: str,
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    num: float | None = None,
    tbs: str | None = None,
    page: float | None = None,
    autocorrect: bool | None = None,
) -> dict:
    """Perform a Serper web search and retrieve rich search results."""
    return {}


def tool_google_search__scrape(url: str, includeMarkdown: bool | None = None) -> dict:
    """Scrape a webpage and retrieve its text content."""
    return {}


def tool_serper_search__google_search(
    q: str,
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    num: float | None = None,
    tbs: str | None = None,
    page: float | None = None,
    autocorrect: bool | None = None,
) -> dict:
    """Perform a Serper web search and retrieve rich search results."""
    return {}


def tool_serper_search__scrape(url: str, includeMarkdown: bool | None = None) -> dict:
    """Scrape a webpage and retrieve its text content."""
    return {}
