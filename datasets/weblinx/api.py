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


def scroll(dx: float, dy: float) -> None:
    """Scroll the page. Scroll horizontally dx pixels, vertically dy pixels. Positive for right or down scrolling, negative for left or up scrolling.

    Args:
    ----
        dx: the distance to scroll in the x direction.
        dy: the distance to scroll in the y direction.

    """
    pass


def goto(url: str) -> None:
    """Navigate to the given URL.

    Args:
    ----
        url (str): The URL to navigate to.

    """
    pass


def select(xpath: str, value: str) -> None:
    """Select an option from a dropdown menu.

    Args:
    ----
        xpath (str): The xpath of the select element.
        value (str): The select option to choose.

    """
    pass


def submit(xpath: str) -> None:
    """Submit the form.

    Args:
    ----
        xpath (str): The xpath of the form to submit.

    """
    pass
