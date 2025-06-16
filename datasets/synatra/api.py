def scroll(delta_x: float, delta_y: float):
    """Scroll horizontally and/or vertically in pixels.

    Args:
    ----
        delta_x (float): Horizontal scroll in pixels.
        delta_y (float): Vertical scroll in pixels.

    """
    pass


def type(bid: str, text: str) -> None:
    """Type the given text into an element with the given id.

    Args:
    ----
        bid (str): The id of the element to type into.
        text (str): The text to type.

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


def close_tab() -> None:
    """Close the current tab."""
    pass


def click(bid: str) -> None:
    """Click on an element with the given id.

    Args:
    ----
        bid (str): The id of the element to click on.

    """
    pass
