# OpenHands Software Agent SDK Converter

This converter emits an OpenHands SDK V1-style SFT record. Unlike
`openhands_v0`, it does not serialize tool use as XML in a special
`function_call` role. It uses OpenAI chat-completions style messages, matching
the SDK's logged LLM call shape:

```json
{
  "id": "trajectory-id",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "I'll inspect the repository."}],
      "tool_calls": [
        {
          "id": "call_000001",
          "type": "function",
          "function": {
            "name": "terminal",
            "arguments": "{\"command\": \"ls\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_000001",
      "name": "terminal",
      "content": [{"type": "text", "text": "..."}]
    }
  ],
  "tools": [
    {"type": "function", "function": {"name": "terminal", "...": "..."}}
  ]
}
```

The converter keeps the SDK default tool set (`terminal`, `file_editor`,
`task_tracker`, `finish`, and `think`) in each record, matching the current SDK
LLM request shape observed with `openhands/minimax-m2.7`.

The bundled `system_prompt.txt` is a stable snapshot of the SDK system prompt
used for generated fixtures. To regenerate with an exact prompt from a specific
SDK checkout, pass `--system-prompt-file`.

```bash
export MY_DATASET=swe-smith
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_std.json \
  | python scripts/json_to_jsonl.py \
  | python agents/openhands_sdk/std_to_sft.py \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
```
