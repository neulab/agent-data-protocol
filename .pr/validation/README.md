# OpenHands SDK Validation Runs

These artifacts were generated with the OpenHands SDK `Agent`/`Conversation`
loop and `LLM(log_completions=True)` using `openhands/minimax-m2.7`.

Each dataset directory contains:

- `example.py`: runnable validation script using an actual converted SFT sample.
- `completion.json`: the last logged SDK completion, or an explicit validation
  error object when no completion was written.
- `run.json`: execution metadata, including selected task text, mocked tools,
  action events, observations, final SDK status, and failure reason if any.

Custom/non-SDK tools are mocked as SDK `ToolDefinition`s with replay executors
that return the observations from the converted dataset trajectory. This is not
a claim that the real external environment was available. It checks whether the
current SDK loop can consume the task and tool schemas and reach completion with
the replayed tool feedback.

Summary from `summary.json`:

- Total datasets: 50
- Completed with at least one parsed non-`finish`/non-`think` tool action: 17
- Completed with such a tool action and a `finish` call: 12
- Completed with such a tool action and a final assistant message: 5
- Reached finished state with only `finish`/`think` or malformed tool calls: 6
- Finished without any parsed tool calls: 13
- Used parsed tool calls but did not finish successfully in the replayed mock
  environment: 13
- Timed out: 1

The incomplete runs are intentionally preserved rather than normalized into
successes.
