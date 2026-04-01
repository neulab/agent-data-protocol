"""API function definitions for the Mind2Web dataset.

Mind2Web uses browser actions targeting elements by BrowserGym ID (BID).
BIDs are resolved from backend_node_id attributes during standardization
via axtree pre-generation.

Function signatures are loaded by agents/openhands/api.py to generate
tool descriptions for the SFT system prompt.
"""


def click(bid: str) -> None:
    """Click on the element.

    Args:
    ----
        bid (str): The browsergym ID of the element to click.

    """
    pass


def type(bid: str, value: str) -> None:
    """Type text into an input element.

    Args:
    ----
        bid (str): The browsergym ID of the input element.
        value (str): The text to type.

    """
    pass


def select(bid: str, value: str) -> None:
    """Select an option from a dropdown menu.

    Args:
    ----
        bid (str): The browsergym ID of the select element.
        value (str): The option to select.

    """
    pass


def goto(url: str) -> None:
    """Navigate to the given URL.

    Args:
    ----
        url (str): The URL to navigate to.

    """
    pass
