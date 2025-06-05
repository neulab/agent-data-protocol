def get_relations(variable: str) -> list[str]:
    """Get all relations connected to an entity or variable in the knowledge base.

    This function helps to explore the knowledge graph by retrieving all relations
    (i.e., edges) that are associated with the given variable, which can be either
    a concrete entity (e.g., "Barack Obama") or a variable placeholder (e.g., "#0").

    Example: get_relations("Barack Obama")
    """
    pass


def get_neighbors(variable: str, relation: str) -> str:
    """Get all entities connected to a variable via a specific relation.

    This function retrieves a new variable containing all entities that are
    connected to the input variable by the given relation. This is typically
    used after get_relations to determine which relation to follow.

    Example: get_neighbors("Barack Obama", "people.person.profession")
    """
    pass


def intersection(variable1: str, variable2: str) -> str:
    """Compute the intersection of two variables.

    This function returns a new variable that includes only the entities
    shared between the two input variables. The input variables must be
    of the same type.

    Example: intersection("#1", "#2")
    """
    pass


def get_attributes(variable: str) -> list[str]:
    """Get all numerical attributes of a variable.

    This function helps to identify which attributes can be used in a
    superlative query (e.g., max/min age). Only use this when a question
    involves ranking or finding extremums.

    Example: get_attributes("#3")
    """
    pass


def argmax(variable: str, attribute: str) -> str:
    """Return the entity with the maximum value of the given attribute.

    Use this function to find the entity with the highest value for the
    specified attribute within a variable. Requires attributes to be known.

    Example: argmax("#2", "age")
    """
    pass


def argmin(variable: str, attribute: str) -> str:
    """Return the entity with the minimum value of the given attribute.

    Use this function to find the entity with the lowest value for the
    specified attribute within a variable. Requires attributes to be known.

    Example: argmin("#2", "age")
    """
    pass


def count(variable: str) -> int:
    """Count the number of entities in a variable.

    Returns the number of distinct entities represented by the variable.

    Example: count("#4")
    """
    pass
