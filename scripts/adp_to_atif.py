import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.atif import adp_trajectory_to_atif
from schema.trajectory import Trajectory


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        trajectory = Trajectory(**json.loads(line))
        atif_trajectory = adp_trajectory_to_atif(trajectory)
        print(atif_trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
