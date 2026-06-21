"""Tests for the SWE-ZERO 12M raw_to_atif converter's observation handling."""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "AlienKevin_SWE-ZERO-12M-trajectories"


def load_raw_to_atif():
    # The converter imports ``schema_raw`` from its own directory, so the
    # dataset directory must be on sys.path (as it is when run as a script).
    dataset_dir = str(DATASET_DIR)
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)

    spec = importlib.util.spec_from_file_location(
        "swe_zero_raw_to_atif", DATASET_DIR / "raw_to_atif.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_record(messages):
    return {
        "instance_id": "test__repo-1",
        "repo": "test/repo",
        "messages": messages,
        "trajectory_format": "mini-swe-agent-1",
        "exit_status": "incomplete",
        "duration_sec": 1.0,
    }


def test_two_consecutive_observations_are_both_retained_on_one_agent_step():
    converter = load_raw_to_atif()
    raw = _raw_record(
        [
            {"role": "user", "content": "Please solve this issue in the repository test/repo."},
            {"role": "assistant", "content": "Let me run a command.\n\n```bash\nls -la\n```"},
            {"role": "user", "content": "Observation: first result line"},
            {"role": "user", "content": "Observation: second result line"},
        ]
    )

    trajectory = converter.process_data(converter.SchemaRaw(**raw))

    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 1, "expected a single agent step"
    agent_step = agent_steps[0]

    assert agent_step.observation is not None, "agent step should carry an observation"
    results = agent_step.observation.results
    assert len(results) == 2, f"expected 2 observation results, got {len(results)}"

    contents = [result.content for result in results]
    assert contents == ["first result line", "second result line"], contents
    # Both observations follow a single bash tool call, so both link to it.
    tool_call_id = agent_step.tool_calls[-1].tool_call_id
    assert all(result.source_call_id == tool_call_id for result in results)


def test_single_observation_still_uses_single_result():
    converter = load_raw_to_atif()
    raw = _raw_record(
        [
            {"role": "user", "content": "Please solve this issue in the repository test/repo."},
            {"role": "assistant", "content": "```bash\nls\n```"},
            {"role": "user", "content": "Observation: only result"},
        ]
    )

    trajectory = converter.process_data(converter.SchemaRaw(**raw))
    agent_step = [step for step in trajectory.steps if step.source == "agent"][0]
    assert agent_step.observation is not None
    assert len(agent_step.observation.results) == 1
    assert agent_step.observation.results[0].content == "only result"


def test_committed_samples_validate_and_keep_observation_links():
    sample_atif = json.loads((DATASET_DIR / "sample_atif.json").read_text())
    assert sample_atif, "sample_atif.json should contain at least one trajectory"
    # Every observation result on an agent step must link to a tool call in that
    # step (or be unlinked); this is enforced by the ATIF schema validator.
    from schema.atif import ATIFTrajectory

    for trajectory in sample_atif:
        parsed = ATIFTrajectory(**trajectory)
        for step in parsed.steps:
            if step.observation:
                tool_call_ids = {tc.tool_call_id for tc in (step.tool_calls or [])}
                for result in step.observation.results:
                    assert result.source_call_id is None or result.source_call_id in tool_call_ids
