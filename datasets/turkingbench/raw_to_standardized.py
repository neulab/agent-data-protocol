import sys
import json
from schema.action.api import ApiAction
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw
from bs4 import BeautifulSoup


actions = {
    "radio": "modify_radio",
    "checkbox": "modify_checkbox",
    "range": "modify_range",
    "text": "modify_text",
    "select": "modify_select",
    "textarea": "modify_text",
}


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data).model_dump()

        html_template = data["Template"]
        for key in data:
            # TODO: handle in SchemaRaw using @root_validator?
            if key in [
                "_id",
                "Task",
                "Title",
                "Description",
                "Keywords",
                "Template",
                "Answer",
            ] or key.startswith("Answer."):
                continue
            html_template = html_template.replace(
                f"${{{key}}}", data[key]
            )  # e.g., replaces '${explanation_1}' with raw_data["explanation_1"]
        content: list = [
            WebObservation(
                html=html_template, url=None, viewport_size=None, image_observation=None
            )
        ]

        soup = BeautifulSoup(html_template, "html.parser")
        for k, v in data["Answer"].items():
            input_element = soup.find_all(
                ["input", "textarea", "select"], {"name": k}, limit=1
            )[0]
            if input_element:
                action = actions.get(input_element["type"])
                if not action:
                    print(
                        f"Could not find action for input element with type {input_element['type']}, name {k}",
                        file=sys.stderr,
                    )
                    continue
                api_action = ApiAction(
                    function=action, kwargs={"input_name": k, "input_value": v}
                )
                content.append(api_action)
            else:
                print(f"Could not find input element with name {k}", file=sys.stderr)
                continue

        standardized_data = Trajectory(
            id=data["_id"],
            content=content,
            details={
                "task": data["Task"],
                "title": data["Title"],
                "description": data["Description"],
                "keywords": data["Keywords"],
            },
        )

        # Print the standardized data
        print(standardized_data.model_dump_json())
