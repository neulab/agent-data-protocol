def go(location: str):
    """Move to the specified location.

    Args:
        location (str): The target location to move to.

    Example:
        go("bed 1")

    """
    pass


def take(item: str, source: str):
    """Pick up an item from a specified source location.

    Args:
        item (str): The item to pick up.
        source (str): The location from which to take the item.

    Example:
        take("laptop 1", "diningtable 1")

    """
    pass


def put(item: str, target: str):
    """Place an item onto or into a specified target.

    Args:
        item (str): The item to place.
        target (str): The surface or container to place the item in/on.

    Example:
        put("laptop 1", "bed 1")

    """
    pass


def open(obj: str):
    """Open a specified container or object.

    Args:
        obj (str): The object to open (e.g., drawer, door).

    Example:
        open("drawer 1")

    """
    pass


def heat(item: str, appliance: str):
    """Heat an item using a specified appliance.

    Args:
        item (str): The item to be heated (e.g., "plate 1").
        appliance (str): The appliance to use for heating (e.g., "microwave 1").

    Example:
        heat("plate 1", "microwave 1")

    """
    pass


def examine(obj: str):
    """Inspect or look closely at an object in the environment.

    Args:
        obj (str): The object to examine (e.g., "dresser 1").

    Example:
        examine("dresser 1")

    """
    pass


def cool(item: str, appliance: str):
    """Cool an item using a specified appliance.

    Args:
        item (str): The item to be cooled (e.g., "plate 1").
        appliance (str): The appliance to use for cooling (e.g., "fridge 1").

    Example:
        cool("plate 1", "fridge 1")

    """
    pass


def use(obj: str):
    """Use or activate a specified object or appliance.

    Args:
        obj (str): The object to use (e.g., "desklamp 1").

    Example:
        use("desklamp 1")

    """
    pass


def close(obj: str):
    """Close a specified container or object.

    Args:
        obj (str): The object to close (e.g., "fridge 1").

    Example:
        close("fridge 1")

    """
    pass


def clean(item: str, appliance: str):
    """Clean an item using a specified appliance.

    Args:
        item (str): The item to be cleaned (e.g., "ladle 2").
        appliance (str): The appliance used for cleaning (e.g., "sinkbasin 1").

    Example:
        clean("ladle 2", "sinkbasin 1")

    """
    pass


def report_problem(obj: str):
    """Report an issue with a specified object in the environment.

    Args:
        obj (str): The object with a problem (e.g., "toilet 1").

    Example:
        report_problem("toilet 1")

    """
    pass


def inventory():
    """Check currently held items.

    This function retrieves and lists the objects currently in possession.

    Example:
        inventory()

    """
    pass


def look():
    """Survey the surroundings to get a description of the current environment.

    This function allows the agent to observe visible objects and locations nearby.

    Example:
        look()

    """
    pass


def look_at_under(item: str, reference: str):
    """Look closely at an item that is located under a specified object.

    Args:
        item (str): The item to examine (e.g., "cellphone 1").
        reference (str): The object under which the item is located (e.g., "desklamp 1").

    Example:
        look_at_under("cellphone 1", "desklamp 1")

    """
    pass
