def touch_and_lift(x0: float, y0: float, x1: float, y1: float) -> None:
    """Touch at the given x0, y0 coordinates and lift at x1, y1.

    Args:
    ----
        x0 (float): The x coordinate to touch.
        y0 (float): The y coordinate to touch.
        x1 (float): The x coordinate to lift.
        y1 (float): The y coordinate to lift.

    """
    pass

def type(text: str):
    """Type given text through keyboard.

    Args:
    ----
        text (str): the text to input.

    """
    pass

def press(key_name: str):
    """Press a special key according the key name.

    Args:
    ----
        key_name (str): go_back | go_home | enter, the key to press

    """
    pass

def end(succeeds: bool):
    """Claim the end of the task with whether it is successfully completed.

    Args:
    ----
        succeeds (bool): if the task is successful

    """
    pass
