def __getattr__(name):
    """Return a permissive placeholder for DTA-Tool's per-trajectory APIs."""

    def dynamic_tool(**kwargs):
        """Provide a placeholder dynamic API for schema validation."""
        return None

    return dynamic_tool
