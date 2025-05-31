def click(xpath: str) -> None:
    """Click on the element.

    Args:
    ----
        xpath (str): The xpath of the element to click.

    """
    pass


def type(xpath: str, value: str) -> None:
    """Type some text into an input element.

    Args:
    ----
        xpath (str): The xpath of the element to type into.
        value (str): The text to type.

    """
    pass


def keyboard_press(xpath: str, value: str) -> None:
    """Press the key.

    Valid keys are defined in https://playwright.dev/python/docs/api/class-keyboard#keyboard-press

    Args:
    ----
        xpath: the xpath of the selected element.
        value: the key to press.

    """
    pass


def scroll(dx: float, dy: float) -> None:
    """Scroll the page. Scroll horizontally dx pixels, vertically dy pixels. Positive for right or down scrolling, negative for left or up scrolling.

    Args:
    ----
        dx: the distance to scroll in the x direction.
        dy: the distance to scroll in the y direction.

    """
    pass
