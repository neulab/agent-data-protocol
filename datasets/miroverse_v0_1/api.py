from typing import Any


def use_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any] | str):
    """Use a tool provided by a Model Context Protocol server.

    Args:
        server_name: Name of the MCP server that provides the tool.
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as a JSON object, or the raw argument string when parsing fails.

    """
    return None
