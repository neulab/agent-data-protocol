from typing import Any


def mouseup(element: dict[str, Any], misc: dict[str, Any]) -> None:
    """Click on the element.

    Args:
    ----
        element: the dictionary that stores the element information. The most important key is 'xpath'.
        misc: meta information such as whether it is a right click.

    """
    pass


def keystroke(element: dict[str, Any], str: str) -> None:
    """Type the string into the element.

    Args:
    ----
        element: the dictionary that stores the element information. The most important key is 'xpath'.
        str: the string to type, seperate by space. For example 'h' 'e' 'l' 'l' 'o'.

    """
    pass


def keypress(key) -> None:
    """Press the key.

    Args:
    ----
        key: the key to press.

    """
    pass


def scroll(x: float, y: float) -> None:
    """Scroll the page.

    Args:
    ----
        x: the distance to scroll in the x direction.
        y: the distance to scroll in the y direction.

    """
    pass
