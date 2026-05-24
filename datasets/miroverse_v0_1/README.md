# MiroVerse v0.1

This converter adds the SFT portion of [`miromind-ai/MiroVerse-v0.1`](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1), a large full-trajectory deep-research agent dataset from MiroMind.

## Source

- **Dataset:** `miromind-ai/MiroVerse-v0.1`
- **Task type:** question answering with full agent rollout trajectories, including MCP-style tool calls and tool observations
- **Size:** 147,985 SFT samples across the JSONL configs listed by the dataset card; the issue's HF viewer metadata reported approximately 227,584 rows including additional configurations
- **Split used:** train JSONL SFT configs (`MiroVerse-Voyager1.0`, `MiroVerse-MuSiQue`, `MiroVerse-HotpotQA`, `MiroVerse-WebWalkerQA-Silver`, `MiroVerse-MegaScience`, `MiroVerse-TaskCraft`, `MiroVerse-QA-Expert-Multi-Hop-V1.0`, `MiroVerse-OneGen-TrainDataset-MultiHopQA`, `MiroVerse-2WikiMultihopQA`, `MiroVerse-WikiTables`, `MiroVerse-WebShaper`, and `MiroVerse-WebDancer`)
- **License:** hybrid; trace data is CC-BY-NC-4.0 and query/answer data retains the original source licenses according to the dataset card

The source Hugging Face repository is gated. Run `extract_raw.py` after accepting the dataset terms and setting an authorized `HF_TOKEN`.

## Schema mapping

Raw rows contain a `messages` list with OpenAI-style `system`, `user`, and `assistant` messages plus a `split` name. The extractor also parses each row's system prompt and stores the row-specific MCP tools in `available_tools`.

- Raw `system` messages are preserved in `Trajectory.details["system_prompt"]` rather than emitted as a conversation turn.
- Tool declarations are parsed from the `## Server name` / `### Tool name` JSON-schema blocks in the system prompt, sanitized into direct Python-callable names such as `tool_google_search__scrape`, and recorded on the top-level `Trajectory.available_apis` field. The dataset's `api.py` ships matching stubs for every advertised tool so the OpenHands SFT converter can expand only the per-trajectory subset via `include_apis`.
- Raw `user` messages become `TextObservation(source="user")`, except the user message immediately following a parsed tool call becomes `TextObservation(source="environment")` because MiroVerse stores MCP tool results as user-role messages.
- Assistant messages containing `<use_mcp_tool>...</use_mcp_tool>` become direct `ApiAction` calls to the parsed tool function; the assistant reasoning before the XML call is preserved as the action description.
- Other assistant messages become `MessageAction`. The final assistant answer is wrapped as a finish message during standardization so the OpenHands SFT sample has an explicit completion action.

## Sample generation

```bash
export MY_DATASET=miroverse_v0_1
export PYTHONPATH=`pwd`:$PYTHONPATH

MIROVERSE_SOURCE_DATASET=WaltonFuture/agentic-sft-new \
MIROVERSE_FLAT_LAYOUT=1 \
MIROVERSE_MAX_PER_CONFIG=1 \
MIROVERSE_CONFIGS="MiroVerse-HotpotQA,MiroVerse-WebWalkerQA-Silver,MiroVerse-WikiTables" \
python datasets/$MY_DATASET/extract_raw.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json
```

If you are regenerating the committed sample without access to the gated source repository, you can point `MIROVERSE_SOURCE_DATASET` at a repository containing the same JSONL files and set `MIROVERSE_FLAT_LAYOUT=1` when those files are stored at the repository root.
