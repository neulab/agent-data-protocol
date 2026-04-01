"""API function definitions for the AgentNet dataset.

AgentNet uses PyAutoGUI actions with normalized coordinates (0-1 range)
for desktop GUI automation across Windows, macOS, and Ubuntu.

Function names match PyAutoGUI's actual API (camelCase) as they appear
in the training data. These function stubs are loaded by
agents/openhands/api.py to generate tool descriptions for the SFT
system prompt.
"""


def click(x: float, y: float) -> None:
    """Click at the specified coordinates.

    Args:
    ----
        x (float): Normalized x coordinate (0-1 range).
        y (float): Normalized y coordinate (0-1 range).

    """
    pass


def doubleClick(x: float, y: float) -> None:
    """Double-click at the specified coordinates.

    Args:
    ----
        x (float): Normalized x coordinate (0-1 range).
        y (float): Normalized y coordinate (0-1 range).

    """
    pass


def rightClick(x: float, y: float) -> None:
    """Right-click at the specified coordinates.

    Args:
    ----
        x (float): Normalized x coordinate (0-1 range).
        y (float): Normalized y coordinate (0-1 range).

    """
    pass


def write(message: str) -> None:
    """Type text using the keyboard.

    Args:
    ----
        message (str): The text to type.

    """
    pass


def press(key: str) -> None:
    """Press a single key.

    Args:
    ----
        key (str): The key to press (e.g. "enter", "tab", "escape", "backspace").

    """
    pass


def hotkey(keys: list) -> None:
    """Press a key combination simultaneously.

    Args:
    ----
        keys (list): List of keys to press together (e.g. ["ctrl", "c"]).

    """
    pass


def scroll(clicks: int) -> None:
    """Scroll vertically. Positive values scroll up, negative scroll down.

    Args:
    ----
        clicks (int): Number of scroll clicks. Positive = up, negative = down.

    """
    pass


def moveTo(x: float, y: float) -> None:
    """Move the mouse cursor to the specified coordinates.

    Args:
    ----
        x (float): Normalized x coordinate (0-1 range).
        y (float): Normalized y coordinate (0-1 range).

    """
    pass


def dragTo(x: float, y: float, button: str = "left") -> None:
    """Drag the mouse from its current position to the specified coordinates.

    Args:
    ----
        x (float): Normalized x coordinate of the drag destination (0-1 range).
        y (float): Normalized y coordinate of the drag destination (0-1 range).
        button (str): Mouse button to use for dragging (default: "left").

    """
    pass


def wait() -> None:
    """Wait for the screen to update before taking the next action."""
    pass


def terminate(status: str) -> None:
    """Signal task completion.

    Args:
    ----
        status (str): Task outcome, e.g. "success" or "failure".

    """
    pass
