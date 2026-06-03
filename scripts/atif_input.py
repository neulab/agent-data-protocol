"""Input helpers for SFT converters that consume ATIF trajectories."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from schema.atif import (
    ATIF_SCHEMA_VERSION,
    ATIFTrajectory,
    atif_trajectory_to_adp,
    normalize_atif_trajectory,
)
from schema.trajectory import Trajectory


def load_trajectory(line: str) -> Trajectory:
    data: dict[str, Any] = json.loads(line)
    if data.get("schema_version") == ATIF_SCHEMA_VERSION:
        atif_trajectory = normalize_atif_trajectory(ATIFTrajectory(**data))
        return atif_trajectory_to_adp(atif_trajectory)
    if data.get("schema_version") is not None:
        return Trajectory(**data)

    try:
        return Trajectory(**data)
    except ValidationError as adp_error:
        try:
            atif_trajectory = normalize_atif_trajectory(ATIFTrajectory(**data))
        except ValidationError:
            raise adp_error from None
        return atif_trajectory_to_adp(atif_trajectory)
