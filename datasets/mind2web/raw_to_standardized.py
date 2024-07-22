import json
import sys

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from datasets.mind2web.schema_raw import SchemaRaw, Action as RawAction


def convert_step(step: RawAction) -> tuple[WebObservation, ApiAction]:
    web_observation = WebObservation(
        html=step.raw_html,
        # TODO: this should be added to the schema
        # https://github.com/neulab/agent-data-collection/issues/26
        image_observation=None,
        viewport_size=None,
        url=None,
    )

    api_action = ApiAction(
        function=step.operation.op.lower(),
        kwargs={"value": step.operation.value} if step.operation.value else {},
        description=None,
    )
    return web_observation, api_action


for line in sys.stdin:

    raw_data = json.loads(line)
    data = SchemaRaw(**raw_data)

    content: list = [TextObservation(content=data.confirmed_task, source="user")]
    for action in data.actions:
        content.extend(convert_step(action))

    standardized_data = Trajectory(
        id=data.annotation_id,
        content=content,
        details={
            "website": data.website,
            "domain": data.domain,
            "confirmed_task": data.confirmed_task,
            "subdomain": data.subdomain,
        },
    )

    # Print the standardized data
    print(standardized_data.model_dump_json())
