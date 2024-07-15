def click(element_id: str) -> None:
    """Click on an element with the given id.

    Args:
    ----
        element_id (str): The id of the element to click.

    """
    pass


def type(element_id: str, text: str) -> None:
    """Type the given text into an element with the given id.

    Args:
    ----
        element_id (str): The id of the element to type into.
        text (str): The text to type.

    """
    pass


def scroll(direction: str) -> None:
    """Scroll the page in the given direction.

    Args:
    ----
        direction (str): The direction to scroll in. Either "up" or "down".

    """
    pass


def press(key_comb: str) -> None:
    """Press a key combination.

    Args:
    ----
        key_comb (str): The key combination to press. For example, "Ctrl+C". This is system specific.

    """
    pass


def stop(answer: str = "") -> None:
    """Stop the execution of the trajectory.

    Args:
    ----
        answer (str): The answer to the question. This is optional since a task may not require an answer.

    """
    pass


def hover(element_id: str) -> None:
    """Hover over an element with the given id.

    Args:
    ----
        element_id (str): The id of the element to hover over.

    """
    pass


def goto(url: str) -> None:
    """Navigate to the given URL.

    Args:
    ----
        url (str): The URL to navigate to.

    """
    pass


def new_tab(url: str) -> None:
    """Open a new tab with the given URL.

    Args:
    ----
        url (str): The URL to open in the new tab.

    """
    pass


def tab_focus(page_number: int) -> None:
    """Focus on the tab with the given page_number.

    Args:
    ----
        page_number (int): The page_number of the tab to focus on. Starts from 0.

    """
    pass


def tab_close() -> None:
    """Close the current tab."""
    pass


def go_back() -> None:
    """Go back to the previous page."""
    pass


def go_forward() -> None:
    """Go forward to the next page if go_back was called."""
    pass
