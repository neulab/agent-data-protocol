# AgentNet

## Overview

AgentNet is a large-scale human-annotated dataset for desktop GUI automation, covering Windows, macOS, and Ubuntu platforms across 140+ applications.

- **Paper**: [OpenCUA: A Unified Benchmark and Toolkit for Computer Use Agents](https://arxiv.org/abs/2508.09123)
- **Source**: [HuggingFace - xlangai/AgentNet](https://huggingface.co/datasets/xlangai/AgentNet)
- **License**: MIT
- **Size**: ~22.6K trajectories (5K Ubuntu + 18K Windows/macOS)

## Action Space

Actions use PyAutoGUI with **normalized coordinates** (0-1 range):

| Action | Example |
|--------|---------|
| `click(x, y)` | `pyautogui.click(x=0.16, y=0.27)` |
| `double_click(x, y)` | `pyautogui.double_click(x=0.5, y=0.3)` |
| `right_click(x, y)` | `pyautogui.right_click(x=0.5, y=0.5)` |
| `write(text)` | `pyautogui.write("hello")` |
| `press(key)` | `pyautogui.press("enter")` |
| `hotkey(*keys)` | `pyautogui.hotkey("ctrl", "c")` |
| `scroll(clicks)` | `pyautogui.scroll(-5)` |
| `drag(x1, y1, x2, y2)` | `pyautogui.drag(0.1, 0.2, 0.3, 0.4)` |

## Quality Annotations

Per-trajectory scores (0-10):
- `alignment_score`: How well actions aligned with the task objective
- `efficiency_score`: How few redundant steps were taken
- `task_difficulty`: Inherent complexity of the task
- `task_completed`: Boolean success flag

Per-step flags:
- `last_step_correct`: Whether the step was correct
- `last_step_redundant`: Whether the step was unnecessary

## Usage

```bash
# Extract raw data (JSONL only):
python datasets/agentnet/extract_raw.py --stats-only

# Extract with quality filtering:
python datasets/agentnet/extract_raw.py --completed-only --min-alignment=7

# Full pipeline:
PYTHONPATH=$(pwd):$PYTHONPATH
cat datasets/agentnet/sample_raw.json \
    | python scripts/json_to_jsonl.py \
    | python datasets/agentnet/raw_to_standardized.py \
    | python scripts/jsonl_to_json.py > datasets/agentnet/sample_std.json
```

## Data Files

- `agentnet_ubuntu_5k.jsonl` — Ubuntu trajectories
- `agentnet_win_mac_18k.jsonl` — Windows/macOS trajectories
- Images in split zip archives (~200GB total)
