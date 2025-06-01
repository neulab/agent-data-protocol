def type(id: str, text: str, press_enter_after: bool = True) -> None:
    """Type text into an element and optionally press enter afterwards."""
    pass


def press(key_comb: str) -> None:
    """Press a key combination.

    Args:
    ----
        key_comb (str): The key combination to press. For example, "Ctrl+C". This is system specific.

    """
    pass


def scroll(direction: str) -> None:
    """Scroll the page up or down."""
    pass


def new_tab() -> None:
    """Open a new tab."""
    pass


def tab_focus(tab_index: int) -> None:
    """Switch focus to a specific tab."""
    pass


def close_tab() -> None:
    """Close the current tab."""
    pass


def stop(answer: str = None) -> None:
    """Stop the interaction with an optional answer."""
    pass
