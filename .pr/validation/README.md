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
- Completed as expected: 50
- Stale or unvalidated runs: 0
- Incomplete runs: 0
- Completed with at least one parsed non-`finish`/non-`think` tool action: 46
- Completed with at least one parsed tool action of any kind: 50
- Records with no environment tool calls expected in the converted SFT sample: 4

Strict trajectory buckets compare converted environment tool-call names to live
SDK action event tool names, excluding `finish` and `think`:

- Exact action sequence: 15
- Close prefix or small difference: 15
- High overlap: 2
- Moderate overlap: 8
- Low overlap: 6
- Unknown live tools: 0
- No environment tool calls expected: 4

For records with environment tool calls, `completed_as_expected` requires at
least one parsed non-`finish`/non-`think` tool action before the SDK run reaches
a finished state. For records whose converted sample has no environment tool
calls, reaching a finished state is considered expected.
