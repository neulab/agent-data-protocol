import time
from typing import List, Literal


def noop(wait_ms: float = 1000) -> None:
    """Do nothing, and optionally wait for the given time (in milliseconds).

    Args:
    ----
        wait_ms (float): Time to wait in milliseconds. Defaults to 1000ms.

    """
    time.sleep(wait_ms / 1000)


def send_msg_to_user(text: str) -> None:
    """Send a message to the user.

    Args:
    ----
        text (str): The message to send to the user.

    """
    pass


def scroll(delta_x: float, delta_y: float) -> None:
    """Scroll horizontally and vertically.

    Amounts in pixels, positive for right or down scrolling,
    negative for left or up scrolling. Dispatches a wheel event.

    Args:
    ----
        delta_x (float): The distance to scroll horizontally in pixels.
        delta_y (float): The distance to scroll vertically in pixels.

    """
    pass


def fill(bid: str, value: str) -> None:
    """Fill out a form field.

    It focuses the element and triggers an input event with the entered text.
    Works for <input>, <textarea> and [contenteditable] elements.

    Args:
    ----
        bid (str): The browser ID of the element to fill.
        value (str): The text to enter into the field.

    """
    pass


def click(
    bid: str,
    button: Literal["left", "middle", "right"] = "left",
    modifiers: List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = None,
) -> None:
    """Click an element.

    Args:
    ----
        bid (str): The browser ID of the element to click.
        button (Literal["left", "middle", "right"]): The mouse button to use. Defaults to "left".
        modifiers (List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]]):
            List of modifier keys to hold while clicking. Defaults to None.

    """
    if modifiers is None:
        modifiers = []
    pass


def hover(bid: str) -> None:
    """Hover over an element.

    Args:
    ----
        bid (str): The browser ID of the element to hover over.

    """
    pass


def type(bid: str, text: str, press_enter_after: int = 1) -> None:
    """Type the given text into an element with the given id.

    Args:
    ----
        bid (str): The browser ID of the element to type into.
        text (str): The text to type.
        press_enter_after (int): Whether to press enter after typing the text. Defaults to 1. 0 means, do not press enter.

    """
    pass


def go_back() -> None:
    """Navigate to the previous page in history."""
    pass


def go_forward() -> None:
    """Navigate to the next page in history."""
    pass


def goto(url: str) -> None:
    """Navigate to a url.

    Args:
    ----
        url (str): The URL to navigate to.

    """
    pass


def new_tab() -> None:
    """Open a new tab."""
    pass


def tab_focus(index: int) -> None:
    """Bring tab to front (activate tab).

    Args:
    ----
        index (int): The index of the tab to focus.

    """
    pass


def tab_close() -> None:
    """Close the current tab."""
    pass


def keyboard_press(key: str) -> None:
    """Press a combination of keys.

    Accepts the logical key names that are emitted in the keyboardEvent.key property
    of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab,
    Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp,
    ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can
    alternatively specify a single character you'd like to produce such as "a" or "#".
    Following modification shortcuts are also supported: Shift, Control, Alt, Meta,
    ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux
    and to Meta on macOS.

    Args:
    ----
        key (str): The key or key combination to press.

    """
    pass
