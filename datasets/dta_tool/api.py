def __getattr__(name):
    """Return a permissive placeholder for DTA-Tool's per-trajectory APIs."""

    def dynamic_tool(**kwargs):
        """Placeholder dynamic API used for schema validation."""
        return None

    return dynamic_tool
