from typing import List, Literal, Union


def noop(wait_ms: float = 1000) -> None:
    """Do nothing, and optionally wait for the given time (in milliseconds).

    Args:
    ----
        wait_ms (float): Time to wait in milliseconds. Defaults to 1000ms.

    """
    pass


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


def select_option(bid: str, options: Union[str, List[str]]) -> None:
    """Select one or multiple options in a <select> element.

    Args:
    ----
        bid (str): The browser ID of the select element.
        options (Union[str, List[str]]): The option value(s) or label(s) to select.

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
    pass


def dblclick(
    bid: str,
    button: Literal["left", "middle", "right"] = "left",
    modifiers: List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = None,
) -> None:
    """Double click an element.

    Args:
    ----
        bid (str): The browser ID of the element to double click.
        button (Literal["left", "middle", "right"]): The mouse button to use. Defaults to "left".
        modifiers (List[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]]):
            List of modifier keys to hold while double clicking. Defaults to None.

    """
    pass


def hover(bid: str) -> None:
    """Hover over an element.

    Args:
    ----
        bid (str): The browser ID of the element to hover over.

    """
    pass


def press(bid: str, key_comb: str) -> None:
    """Focus the matching element and press a combination of keys.

    Args:
    ----
        bid (str): The browser ID of the element to focus.
        key_comb (str): The key combination to press (e.g., "Backspace", "ControlOrMeta+a").

    """
    pass


def focus(bid: str) -> None:
    """Focus the matching element.

    Args:
    ----
        bid (str): The browser ID of the element to focus.

    """
    pass


def clear(bid: str) -> None:
    """Clear the input field.

    Args:
    ----
        bid (str): The browser ID of the input field to clear.

    """
    pass


def drag_and_drop(from_bid: str, to_bid: str) -> None:
    """Perform a drag & drop operation.

    Args:
    ----
        from_bid (str): The browser ID of the element to drag.
        to_bid (str): The browser ID of the element to drop onto.

    """
    pass


def upload_file(bid: str, file: Union[str, List[str]]) -> None:
    """Click an element and wait for a "filechooser" event, then select file(s) for upload.

    Args:
    ----
        bid (str): The browser ID of the file input element.
        file (Union[str, List[str]]): Path(s) to the file(s) to upload.

    """
    pass


def report_infeasible(reason: str) -> None:
    """Notifies the user that their instructions are infeasible.

    Args:
    ----
        reason (str): The reason why the instructions cannot be followed.

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
