import argparse
import json
import sys
from typing import List

from datasets.screenagent.schema_raw import SchemaRaw, ScreenAgentItem
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-all", action="store_true", help="Keep all fields")
    return parser.parse_args()


def process_data(data: List[ScreenAgentItem], keep_all: bool = False) -> List[Trajectory]:
    """
    Process the raw data into standardized format.
    """
    trajectories = []

    for i, trajectory_data in enumerate(data):
        content = []

        # Add the task prompt as a user message
        content.append(
            TextObservation(
                content=trajectory_data.task_prompt_en,
                source="user",
            )
        )

        # Add the screenshot as an image observation
        if trajectory_data.screenshot and trajectory_data.screenshot != "<image>":
            content.append(
                ImageObservation(
                    content=trajectory_data.screenshot,
                    source="environment",
                )
            )

        # Add the LLM response as a message action
        content.append(
            MessageAction(
                content=trajectory_data.LLM_response_editer_en,
            )
        )

        # Create a trajectory
        trajectory = Trajectory(
            id=f"{trajectory_data.session_id}_{i}",
            content=content,
        )

        trajectories.append(trajectory)

    return trajectories


if __name__ == "__main__":
    args = get_args()

    # Process each line of JSONL from stdin
    for line in sys.stdin:
        # Parse the JSON object from the current line
        trajectory_group = json.loads(line)

        # Process the data
        data = SchemaRaw.model_validate(trajectory_group).root
        trajectories = process_data(data, keep_all=args.keep_all)

        # Print each trajectory as a separate JSON line
        for trajectory in trajectories:
            print(json.dumps(trajectory.model_dump()))
