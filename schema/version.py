"""Canonical ADP standardized schema version metadata."""

from typing import Final

SCHEMA_VERSION: Final = "1.0.0"
# Add prior versions here when bumping to preserve compatibility with existing data.
SUPPORTED_SCHEMA_VERSIONS: Final = (SCHEMA_VERSION,)
