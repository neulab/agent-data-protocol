# OpenHands SDK Validation Runs

These artifacts validate the OpenHands SDK converter with self-contained per-dataset examples.
Each `example.py` derives its tool set from the dataset's `metadata.json` and the first
standardized record's `available_custom_tools` field, then sends that first task message to a
real OpenHands SDK `Agent`/`Conversation`.

Tool selection follows the OpenHands SDK converter rules:

- metadata `custom_tools` become dataset custom SDK tools unless they map to SDK-native tools;
- `available_custom_tools` narrows the dataset custom tool set when present;
- `code_enabled: ["bash"]` adds the SDK terminal tool;
- `browser_enabled: true` adds the SDK browser tool set, filtered to the browser tools emitted by the converter;
- SDK-native aliases such as `str_replace_editor`, `submit`, `finish`, `think`, and browser action aliases are not duplicated as custom tools.

Run a dataset example with:

```bash
python .pr/validation/<dataset>/example.py
```

By default the script uses `DockerWorkspace`. In environments without Docker socket access, set
`VALIDATION_WORKSPACE=local` to run in a temporary local workspace, or set
`VALIDATION_WORKSPACE=remote` and `VALIDATION_WORKSPACE_HOST` to use an existing OpenHands agent
server.

`run.json` records the resolved tools, workspace mode, SDK events, and final status.
`completion.json` records the latest logged LLM completion when one is available.
