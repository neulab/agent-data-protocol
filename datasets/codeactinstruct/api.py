def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a given query.

    This tool provides access to a vast collection of articles covering a wide range of topics.
    It can be used to retrieve accurate and comprehensive information about specific keywords or topics.

    For example: wikipedia_search("Photosynthesis")
    """
    pass


def put(object: str, receptacle: str) -> str:
    """Put an object in/on a receptacle.

    This is used for interacting with a household environment.

    For example: put("mug 1", "desk 2")
    """
    pass


def goto(receptacle: str) -> str:
    """Go to a location of the receptacle.

    This is used for interacting with a household environment.

    For example: goto("drawer 1")
    """
    pass


def take_from(object: str, receptacle: str) -> str:
    """Take an object from a receptacle.

    This is used for interacting with a household environment.

    For example: take_from("mug 1", "shelf 2")
    """
    pass


def open_receptacle(receptacle: str) -> str:
    """Open a receptacle.

    This is used for interacting with a household environment.

    For example: open_receptacle("fridge 1")
    """
    pass


def toggle(object_or_receptacle: str) -> str:
    """Toggle an object or receptacle.

    This is used for interacting with a household environment.

    For example: toggle("light 2")
    """
    pass


def close_receptacle(receptacle: str) -> str:
    """Close a receptacle.

    This is used for interacting with a household environment.

    For example: close_receptacle("microwave 1")
    """
    pass


def clean(object: str, receptacle: str) -> str:
    """Clean an object with a receptacle.

    This is used for interacting with a household environment.

    For example: clean("cloth 1", "sinkbasin 1")
    """
    pass


def heat(object: str, receptacle: str) -> str:
    """Heat an object with a receptacle.

    This is used for interacting with a household environment.

    For example: heat("egg 1", "microwave 1")
    """
    pass


def cool(object: str, receptacle: str) -> str:
    """Cool an object with a receptacle.

    This is used for interacting with a household environment.

    For example: cool("bottle 1", "fridge 1")
    """
    pass


def use(receptacle: str) -> str:
    """Use a receptacle.

    This is used for interacting with a household environment.

    For example: use("lamp 1")
    """
    pass


def look() -> str:
    """Look around. It will return what you see in the room.

    This is used for interacting with a household environment.

    For example: look()
    """
    pass
