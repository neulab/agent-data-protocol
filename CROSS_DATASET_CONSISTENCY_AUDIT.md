# Cross-Dataset Consistency Audit

Manual audit date: 2026-07-05

Scope: I audited the current 56 dataset directories in the repo, using a local
inspection worktree that merged the open normalization PRs (#285-#303) plus the
new Toucan and AlienKevin updates. Earlier reports cited 54 datasets, but the
repo currently contains 56 after recent additions, so I included all 56.

For each target dataset I picked three similar datasets by task family, manually
looked at the target and comparators' standardized samples/converters, and
checked for inconsistencies in role layout, system/task separation, tool-call
structure, observation attachment, reasoning cleanup, and sample regeneration.

## Fixes Opened Or Updated

- `AlienKevin_SWE-ZERO-12M-trajectories`: inconsistent with comparable
  SWE/terminal traces because it dropped a rich system prompt. Fixed in
  PR #304 by preserving the system step.
- `toucan_1_5m`: inconsistent with other function-calling datasets because it
  had tool definitions and function observations but no structured tool calls.
  Fixed in PR #291 by recovering legacy `function_call` records and linking
  observations.
- `litecoder-terminal-sft`, `nemotron_terminal_corpus`, and the shared terminal
  helper: terminal completion `finish` calls now include a non-empty `message`
  argument. Updated in PRs #293, #292, and #301.

## Dataset-by-Dataset Comparison

| Dataset | Three closest comparators | Manual consistency judgment |
| --- | --- | --- |
| `AlienKevin_SWE-ZERO-12M-trajectories` | `nebius_SWE-agent-trajectories`, `litecoder-terminal-sft`, `openthoughts_agent_sft` | Found a real inconsistency: the target dropped its system prompt while comparators preserve task-shaping terminal/system instructions. PR #304 fixes this. |
| `CharlieDreemur_OpenManus-RL` | `agenttuning_alfworld`, `eto`, `agenttuning_webshop` | Consistent for text environment tasks. It keeps environment/task framing and does not invent generic browser/terminal tools for AlfWorld-like actions. |
| `SALT-NLP_SWE-chat` | `coderforge_preview`, `logicstar_swe-star`, `swe-play-trajectories` | Consistent SWE trace normalization: user task remains user-authored, terminal/editor aliases are canonicalized, and explicit `think` remains a tool. |
| `agenttuning_alfworld` | `CharlieDreemur_OpenManus-RL`, `agenttuning_webshop`, `eto` | Consistent with environment-action datasets. Domain actions such as `go`, `take`, and `open` are preserved rather than forced into generic tools. |
| `agenttuning_db` | `agenttuning_kg`, `orca_agentinstruct`, `code_feedback` | Consistent as text-only instruction/response data. SQL actions are embedded transcript text in this dataset family, not structured calls. |
| `agenttuning_kg` | `agenttuning_db`, `cognitivekernel_pro_sft`, `orca_agentinstruct` | Consistent as text-only reasoning data. The multi-turn question/answer flow matches the comparable non-tool instruction datasets. |
| `agenttuning_mind2web` | `screenagent`, `synatra`, `llava_plus` | Consistent for simplified text/multimodal task data. It does not have structured browser actions, unlike full browser traces, and that matches its source shape. |
| `agenttuning_os` | `litecoder-terminal-sft`, `openthoughts_agent_sft`, `nemotron_terminal_corpus` | Consistent after comparing terminal structure. It preserves the Linux task and uses canonical terminal calls; no separate system split is needed for its already-clean prompt form. |
| `agenttuning_webshop` | `eto`, `agenttuning_alfworld`, `CharlieDreemur_OpenManus-RL` | Consistent. WebShop actions remain domain-specific `search`/`click`, comparable to other environment-specific action sets. |
| `allenai_Sera-4.6-Lite-T2` | `swe-smith`, `logicstar_swe-star`, `nvidia_SWE-Zero-openhands-trajectories` | Consistent with strong SWE/OpenHands-style traces: system prompt, user issue, canonical terminal/editor/finish tools, and reasoning extraction. |
| `android_in_the_wild` | `androidcontrol`, `omniact`, `screenagent` | Consistent after PR #285: mobile task is user-authored and image observations/actions are preserved. Remaining observation-only convention is documented in PR #303. |
| `androidcontrol` | `android_in_the_wild`, `omniact`, `mind2web` | Consistent after PR #286 for task role. Android actions remain dataset-specific, which matches comparable mobile/browser control datasets. |
| `code_feedback` | `orca_agentinstruct`, `agenttuning_db`, `openthoughts_tb_dev` | Consistent after stale fixture refresh in PR #294. It is text feedback data with no expected structured tool calls. |
| `codeactinstruct` | `dolci_instruct_sft_tool_use`, `jupyter-agent-dataset`, `toolmind` | Consistent. Its executable code snippets correctly normalize to `python`, unlike domain-specific API functions in Dolci/ToolMind. |
| `coderforge_preview` | `logicstar_swe-star`, `swe-play-trajectories`, `hybrid-gym` | Consistent SWE normalization: terminal/editor/finish are canonical, and explicit `think` calls remain tools. |
| `codescout` | `SALT-NLP_SWE-chat`, `gair_davinci_dev`, `swe-smith` | Consistent for localization/SWE traces. Its localization finish behavior is normalized while preserving repository metadata. |
| `cognitivekernel_pro_sft` | `orca_agentinstruct`, `openthoughts_tb_dev`, `finch_collection` | Consistent as text-only reasoning/instruction data. No structured tool behavior is implied by the sample format. |
| `dolci_instruct_sft_tool_use` | `toolmind`, `toucan_1_5m`, `enterpriselab` | Consistent. Dataset-specific function APIs remain named APIs, and observations attach to structured calls. Toucan was brought into this pattern in PR #291. |
| `enterpriselab` | `dolci_instruct_sft_tool_use`, `toolmind`, `openresearcher` | Consistent. It has system tool instructions, user tasks, structured function calls, and linked observations. No PR needed. |
| `eto` | `agenttuning_webshop`, `agenttuning_alfworld`, `turkingbench` | Consistent. Environment actions are preserved as task-domain calls and not over-normalized. |
| `finch_collection` | `cognitivekernel_pro_sft`, `openthoughts_tb_dev`, `orca_agentinstruct` | Consistent. It keeps system math/evolution context and extracts inline thinking, unlike simpler text-only datasets without `<think>`. |
| `gair_davinci_dev` | `swe-smith`, `codescout`, `SALT-NLP_SWE-chat` | Consistent SWE trace normalization with canonical terminal/editor/finish tools. |
| `go-browse-wa` | `mind2web`, `turkingbench`, `webarena_successful` | Consistent browser-agent data. It uses browser action tools and standalone observations under the documented current convention. |
| `hybrid-gym` | `coderforge_preview`, `swe-gym_openhands_sampled_trajectories`, `swe-play-trajectories` | Consistent after stale fixture refresh. Tool names and role structure match comparable SWE trajectories. |
| `jupyter-agent-dataset` | `codeactinstruct`, `openhands`, `SALT-NLP_SWE-chat` | Consistent. Notebook execution becomes canonical `python`, and finalization becomes `finish`. |
| `kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k` | `mini-coder`, `agenttuning_kg`, `code_feedback` | Consistent with text-transcript SWE data. It lacks structured tool events in the sample, so preserving text turns is appropriate. |
| `litecoder-terminal-sft` | `openthoughts_agent_sft`, `nemotron_terminal_corpus`, `agenttuning_os` | Consistent after PR #293: terminal instructions are split, terminal calls are structured, and completion JSON becomes `finish` with a message. |
| `llava_plus` | `screenagent`, `agenttuning_mind2web`, `synatra` | Consistent multimodal/text task data. The sample does not expose formal tools, so no tool normalization is expected. |
| `logicstar_swe-star` | `coderforge_preview`, `swe-play-trajectories`, `nvidia_SWE-Zero-openhands-trajectories` | Consistent SWE trace normalization with canonical terminal/editor/finish and preserved explicit `think` tools. |
| `mind2web` | `weblinx`, `go-browse-wa`, `turkingbench` | Consistent after fixture refresh. Browser actions and observations follow the current browser dataset convention. |
| `mini-coder` | `kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k`, `code_feedback`, `openthoughts_tb_dev` | Consistent text-coded transcript data. No structured tools are present in the sample source shape. |
| `miroverse_v0_1` | `dolci_instruct_sft_tool_use`, `toolmind`, `toucan_1_5m` | Consistent. Tool definitions are preserved at the agent level; the audited sample itself does not contain function calls. |
| `nebius_SWE-agent-trajectories` | `AlienKevin_SWE-ZERO-12M-trajectories`, `agenttuning_os`, `litecoder-terminal-sft` | Consistent after PR #288 removes empty system placeholders. Unlike AlienKevin, there is no rich system prompt to preserve in the sample. |
| `nebius_SWE-rebench-openhands-trajectories` | `swe-smith`, `nvidia_SWE-Zero-openhands-trajectories`, `scale_swe_distilled` | Consistent OpenHands-style SWE trace normalization with system/user/tool separation. |
| `nemotron_terminal_corpus` | `litecoder-terminal-sft`, `openthoughts_agent_sft`, `agenttuning_os` | Consistent after PR #292: terminal instructions are split, empty turns removed, and completion handler matches LiteCoder/shared helper. |
| `nnetnav-live` | `nnetnav-wa`, `weblinx`, `mind2web` | Consistent for text-only navigation reasoning samples. It keeps observations as user text because no structured tool events are present. |
| `nnetnav-wa` | `nnetnav-live`, `weblinx`, `turkingbench` | Consistent with `nnetnav-live`; both are text reasoning navigation data rather than structured browser-action trajectories. |
| `nvidia_SWE-Zero-openhands-trajectories` | `swe-smith`, `scale_swe_distilled`, `nebius_SWE-rebench-openhands-trajectories` | Consistent OpenHands-style SWE normalization with canonical tools and preserved explicit thinking tools. |
| `omniact` | `android_in_the_wild`, `androidcontrol`, `jupyter-agent-dataset` | Consistent after PR #287: task precedes image observation/action, and Python execution is canonicalized. |
| `openhands` | `jupyter-agent-dataset`, `swe-smith`, `webarena_successful` | Consistent mixed-tool trace normalization. Browser, Python, terminal, and finish events are kept as structured calls. |
| `openresearcher` | `toucan_1_5m`, `enterpriselab`, `dolci_instruct_sft_tool_use` | Consistent after PR #289 structures search/open actions. `browser.open` remains a generic/domain tool because its cursor/id semantics are not the same as browser `goto`. |
| `openthoughts_agent_sft` | `litecoder-terminal-sft`, `nemotron_terminal_corpus`, `agenttuning_os` | Consistent after PR #301 shares terminal prompt/completion helpers. Terminal JSON actions are structured where safely parseable. |
| `openthoughts_tb_dev` | `cognitivekernel_pro_sft`, `mini-coder`, `code_feedback` | Consistent after fixture refresh. This is a compact text/code artifact sample, not a structured tool trace. |
| `orca_agentinstruct` | `agenttuning_db`, `code_feedback`, `cognitivekernel_pro_sft` | Consistent after PR #290 removes empty system placeholders and refreshes samples. |
| `scale_swe_distilled` | `swe-smith`, `nebius_SWE-rebench-openhands-trajectories`, `nvidia_SWE-Zero-openhands-trajectories` | Consistent after fixture refresh. Canonical terminal/editor tools match other SWE traces. |
| `screenagent` | `android_in_the_wild`, `llava_plus`, `agenttuning_mind2web` | Consistent as text/screen-agent sample data. It does not expose formal action events in the committed sample. |
| `swe-gym_openhands_sampled_trajectories` | `hybrid-gym`, `swe-smith`, `scale_swe_distilled` | Consistent after fixture refresh. Tool and role conventions match OpenHands-style SWE traces. |
| `swe-play-trajectories` | `coderforge_preview`, `logicstar_swe-star`, `hybrid-gym` | Consistent after fixture refresh. Explicit `think` remains a tool and execution/editing calls are canonicalized. |
| `swe-smith` | `scale_swe_distilled`, `nvidia_SWE-Zero-openhands-trajectories`, `allenai_Sera-4.6-Lite-T2` | Consistent canonical SWE/OpenHands sample with system, user issue, editor/terminal/finish tools. |
| `synatra` | `agenttuning_mind2web`, `screenagent`, `openthoughts_tb_dev` | Consistent after fixture refresh. The sample is text/action-description data without formal tool calls. |
| `toolmind` | `dolci_instruct_sft_tool_use`, `toucan_1_5m`, `enterpriselab` | Consistent function-calling data. It has structured tool calls and reasoning extraction where present. |
| `toucan_1_5m` | `dolci_instruct_sft_tool_use`, `toolmind`, `enterpriselab` | Found a real inconsistency: legacy `function_call` messages were not structured. PR #291 now recovers tool calls and links observations. |
| `turkingbench` | `webarena_successful`, `mind2web`, `go-browse-wa` | Consistent browser-task data. Browser action calls and standalone observations follow the documented current convention. |
| `webarena_successful` | `turkingbench`, `mind2web`, `weblinx` | Consistent browser task data. `stop`/completion is normalized to `finish`; observations follow current convention. |
| `weblinx` | `wonderbread`, `mind2web`, `turkingbench` | Consistent full web trajectory data. It has structured browser actions and many standalone observations under the documented convention. |
| `wonderbread` | `weblinx`, `mind2web`, `go-browse-wa` | Consistent full web trajectory data. It preserves screenshots/SOP/state and uses structured browser actions. |

## Expanded Manual Examples

- `AlienKevin_SWE-ZERO-12M-trajectories`: I compared it with
  `nebius_SWE-agent-trajectories`, `litecoder-terminal-sft`, and
  `openthoughts_agent_sft`. The target and `nebius` both use one terminal
  command per agent turn, but AlienKevin's raw system prompt includes strict
  command-format and tool-availability rules. That is more like LiteCoder and
  OpenThoughts, where the terminal contract is retained as system context, so
  dropping it was inconsistent and PR #304 preserves it.
- `CharlieDreemur_OpenManus-RL`: I compared it with `agenttuning_alfworld`,
  `eto`, and `agenttuning_webshop`. All four are environment-task datasets where
  actions are domain primitives rather than shell/browser APIs. CharlieDreemur's
  household-style text actions stay as transcript content, while AlfWorld keeps
  `go`/`take`/`put`; this is consistent because the target sample does not expose
  the same structured action fields.
- `SALT-NLP_SWE-chat`: I compared it with `coderforge_preview`,
  `logicstar_swe-star`, and `swe-play-trajectories`. In all four, repository
  editing and command execution are normalized to `file_editor` and `terminal`,
  while explicit `think` tool calls are preserved as tool events rather than
  converted into `reasoning_content`.
- `agenttuning_alfworld`: I compared it with `CharlieDreemur_OpenManus-RL`,
  `agenttuning_webshop`, and `eto`. The target exposes environment actions such
  as `go`, `take`, and `open`; WebShop/ETO expose `search` and `click`. I judged
  these consistent because each preserves its native environment API instead of
  forcing unrelated canonical browser or terminal names.
- `agenttuning_db`: I compared it with `agenttuning_kg`, `orca_agentinstruct`,
  and `code_feedback`. The target describes SQL operations in natural language
  turns, similar to KG and code-feedback text traces. Since the raw sample does
  not expose executable SQL calls, leaving SQL as message text is consistent.
- `agenttuning_kg`: I compared it with `agenttuning_db`,
  `cognitivekernel_pro_sft`, and `orca_agentinstruct`. The target has alternating
  user/agent text reasoning without tool metadata. That matches the other
  text-only reasoning datasets, so there is no basis to invent KG lookup tools.
- `agenttuning_mind2web`: I compared it with `screenagent`, `synatra`, and
  `llava_plus`. The target is a compact text representation of a web task, not a
  full browser-action trace like `mind2web` or `weblinx`. Its lack of structured
  browser calls is therefore consistent with screen/text datasets, not a
  normalization failure.
- `agenttuning_os`: I compared it with `litecoder-terminal-sft`,
  `openthoughts_agent_sft`, and `nemotron_terminal_corpus`. All are OS/terminal
  tasks. The target already exposes a clean task plus `terminal` calls, whereas
  LiteCoder/Nemotron needed prompt splitting because their first user turn mixed
  reusable terminal instructions with the task.
- `agenttuning_webshop`: I compared it with `eto`, `agenttuning_alfworld`, and
  `CharlieDreemur_OpenManus-RL`. The target and ETO both preserve WebShop
  `search`/`click` as domain tools; this is analogous to AlfWorld preserving
  `go`/`take`, and is consistent.
- `allenai_Sera-4.6-Lite-T2`: I compared it with `swe-smith`,
  `logicstar_swe-star`, and `nvidia_SWE-Zero-openhands-trajectories`. All keep a
  system prompt before the user issue and normalize repo actions to
  `file_editor`, `terminal`, and `finish`. The target also extracts inline
  thinking into `reasoning_content`, matching the strongest normalized SWE
  samples.
- `android_in_the_wild`: I compared it with `androidcontrol`, `omniact`, and
  `screenagent`. The target now starts with the mobile task as `user`, matching
  AndroidControl and OmniAct after their fixes. Its `touch_and_lift` action is
  preserved because AndroidControl similarly keeps native mobile actions such as
  `open_app`, `scroll`, and `back`.
- `androidcontrol`: I compared it with `android_in_the_wild`, `omniact`, and
  `mind2web`. The target now labels goals as user tasks, matching the mobile/web
  task convention. Its Android-specific tool names are consistent with the way
  browser datasets keep `click`/`type`/`goto` rather than mapping everything to a
  generic tool.
- `code_feedback`: I compared it with `orca_agentinstruct`, `agenttuning_db`,
  and `openthoughts_tb_dev`. All are text-only or artifact-text datasets. The
  target's user feedback and assistant revisions remain messages, which is
  consistent because none of the comparators expose formal tool-call fields.
- `codeactinstruct`: I compared it with `dolci_instruct_sft_tool_use`,
  `jupyter-agent-dataset`, and `toolmind`. The target uses executable code
  snippets, so normalizing to `python` matches Jupyter. Dolci/ToolMind keep
  domain API names because those are externally defined functions, not generic
  code execution.
- `coderforge_preview`: I compared it with `logicstar_swe-star`,
  `swe-play-trajectories`, and `hybrid-gym`. All are SWE traces with
  editor/shell calls. The target's `terminal`, `file_editor`, and preserved
  `think` tools match the peers' design decisions.
- `codescout`: I compared it with `SALT-NLP_SWE-chat`, `gair_davinci_dev`, and
  `swe-smith`. The target's repository metadata and localization flow are
  preserved, while terminal/finalization are normalized. That matches the SWE
  comparators' split between canonical execution tools and dataset-specific
  provenance.
- `cognitivekernel_pro_sft`: I compared it with `orca_agentinstruct`,
  `openthoughts_tb_dev`, and `finch_collection`. It is a simple text reasoning
  sample. Unlike Finch, it has no `<think>` blocks to extract, and unlike
  tool-call datasets it has no function metadata to structure.
- `dolci_instruct_sft_tool_use`: I compared it with `toolmind`, `toucan_1_5m`,
  and `enterpriselab`. The target uses declared domain APIs and attaches
  observations to tool calls. After the Toucan fix, all four share the same
  basic function-calling design: declared tools stay as named APIs and results
  attach to calls.
- `enterpriselab`: I compared it with `dolci_instruct_sft_tool_use`,
  `toolmind`, and `openresearcher`. The target keeps a long system tool list,
  user enterprise tasks, and structured calls such as `search_repositories`.
  That is consistent with function-calling peers that preserve domain APIs.
- `eto`: I compared it with `agenttuning_webshop`, `agenttuning_alfworld`, and
  `turkingbench`. ETO/WebShop use task-environment actions, while TurkingBench
  uses browser navigation actions. Keeping ETO's WebShop-style `search`/`click`
  names is consistent with preserving native environment APIs.
- `finch_collection`: I compared it with `cognitivekernel_pro_sft`,
  `openthoughts_tb_dev`, and `orca_agentinstruct`. Finch keeps math/evolution
  setup as system context and extracts `<think>` text; the simpler comparators
  lack those features. The difference is justified by Finch's richer raw format.
- `gair_davinci_dev`: I compared it with `swe-smith`, `codescout`, and
  `SALT-NLP_SWE-chat`. The target normalizes file edits and shell commands
  exactly like the other SWE traces, and keeps issue/task text in user turns.
- `go-browse-wa`: I compared it with `mind2web`, `turkingbench`, and
  `webarena_successful`. All are browser/web tasks with action/observation
  alternation. The target uses `click`, `fill`, `select_option`, and `finish`;
  the naming differs slightly from Mind2Web/WebLINX but reflects its source API.
- `hybrid-gym`: I compared it with `coderforge_preview`,
  `swe-gym_openhands_sampled_trajectories`, and `swe-play-trajectories`. The
  target's stale fixture was refreshed, and its canonical editor/terminal tools
  now line up with the other OpenHands/SWE-style traces.
- `jupyter-agent-dataset`: I compared it with `codeactinstruct`, `openhands`,
  and `SALT-NLP_SWE-chat`. Notebook execution is normalized to `python`, which
  matches CodeAct's executable-code normalization and OpenHands' IPython
  handling. Final answers become `finish`.
- `kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k`: I compared it
  with `mini-coder`, `agenttuning_kg`, and `code_feedback`. The target is a
  text transcript without structured tool-call fields, so preserving alternating
  text turns is consistent with those text-style peers.
- `litecoder-terminal-sft`: I compared it with `openthoughts_agent_sft`,
  `nemotron_terminal_corpus`, and `agenttuning_os`. It now follows the same
  terminal pattern: reusable terminal instructions are system context, the
  actual task is user content, command JSON becomes `terminal`, and completion
  JSON becomes `finish` with a message.
- `llava_plus`: I compared it with `screenagent`, `agenttuning_mind2web`, and
  `synatra`. The target's multimodal prompt/response data does not expose
  formal tool events. That is consistent with the closest text/screen datasets,
  while image/provenance remains preserved.
- `logicstar_swe-star`: I compared it with `coderforge_preview`,
  `swe-play-trajectories`, and `nvidia_SWE-Zero-openhands-trajectories`. The
  target uses the same canonical SWE tool surface and preserves explicit
  `think` tool calls rather than treating them as hidden reasoning.
- `mind2web`: I compared it with `weblinx`, `go-browse-wa`, and `turkingbench`.
  The target has browser actions (`goto`, `click`, `type`, `select`) and
  standalone observations, matching the current full-browser trajectory
  convention documented in PR #303.
- `mini-coder`: I compared it with
  `kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k`, `code_feedback`,
  and `openthoughts_tb_dev`. The target has text-only code-solving turns. Since
  no tool-call metadata is present, it should not be normalized like SWE traces
  with explicit shell/editor events.
- `miroverse_v0_1`: I compared it with `dolci_instruct_sft_tool_use`,
  `toolmind`, and `toucan_1_5m`. The target preserves tool definitions at the
  agent level, but the audited sample has no actual function-call turns. That is
  consistent with keeping declared capabilities without inventing calls.
- `nebius_SWE-agent-trajectories`: I compared it with
  `AlienKevin_SWE-ZERO-12M-trajectories`, `agenttuning_os`, and
  `litecoder-terminal-sft`. After empty-system cleanup, it starts with the user
  issue and then terminal calls. Unlike AlienKevin, its sample had no meaningful
  system instruction to preserve.
- `nebius_SWE-rebench-openhands-trajectories`: I compared it with `swe-smith`,
  `nvidia_SWE-Zero-openhands-trajectories`, and `scale_swe_distilled`. It keeps
  OpenHands-style system/user separation and canonical `terminal`/`file_editor`
  tools, matching the peers.
- `nemotron_terminal_corpus`: I compared it with `litecoder-terminal-sft`,
  `openthoughts_agent_sft`, and `agenttuning_os`. It now shares the same
  terminal prompt split and safe terminal JSON handling as LiteCoder and
  OpenThoughts, while preserving extracted reasoning.
- `nnetnav-live`: I compared it with `nnetnav-wa`, `weblinx`, and `mind2web`.
  The target keeps observations as user text and has no structured browser
  calls. That is consistent with `nnetnav-wa`; it is a reasoning/navigation text
  sample, not a WebLINX-style action log.
- `nnetnav-wa`: I compared it with `nnetnav-live`, `weblinx`, and
  `turkingbench`. Like `nnetnav-live`, it keeps the browser observation in text
  because the raw sample lacks formal action/observation events. It should not
  be forced into WebLINX-style tool calls.
- `nvidia_SWE-Zero-openhands-trajectories`: I compared it with `swe-smith`,
  `scale_swe_distilled`, and `nebius_SWE-rebench-openhands-trajectories`. The
  target uses canonical OpenHands-style tools and keeps `think` as an explicit
  tool, consistent with similar SWE traces.
- `omniact`: I compared it with `android_in_the_wild`, `androidcontrol`, and
  `jupyter-agent-dataset`. After PR #287, it starts with the user task before
  image observation/execution. Python actions normalize to `python`, consistent
  with Jupyter/CodeAct-style execution.
- `openhands`: I compared it with `jupyter-agent-dataset`, `swe-smith`, and
  `webarena_successful`. It is mixed-tool data, and its browser, Python,
  terminal, and finish calls stay structured. That matches each comparator's
  relevant tool family rather than collapsing everything into one API.
- `openresearcher`: I compared it with `toucan_1_5m`, `enterpriselab`, and
  `dolci_instruct_sft_tool_use`. After PR #289, search/open actions are
  structured. I kept `browser.open` as a domain/generic tool because its
  cursor/id arguments are not equivalent to a browser-navigation `goto` URL.
- `openthoughts_agent_sft`: I compared it with `litecoder-terminal-sft`,
  `nemotron_terminal_corpus`, and `agenttuning_os`. It already had the desired
  terminal split pattern, and PR #301 factors that behavior into shared helpers
  used by the terminal-family datasets.
- `openthoughts_tb_dev`: I compared it with `cognitivekernel_pro_sft`,
  `mini-coder`, and `code_feedback`. The target is compact code/artifact text
  data and does not expose shell/editor tool events. Keeping it text-only is
  consistent with those comparators.
- `orca_agentinstruct`: I compared it with `agenttuning_db`, `code_feedback`,
  and `cognitivekernel_pro_sft`. After removing empty system placeholders, it is
  a plain instruction/answer dataset like the comparators, with no expected
  tools.
- `scale_swe_distilled`: I compared it with `swe-smith`,
  `nebius_SWE-rebench-openhands-trajectories`, and
  `nvidia_SWE-Zero-openhands-trajectories`. Its `terminal`/`file_editor` tools
  and system/user issue split match the OpenHands-style SWE group.
- `screenagent`: I compared it with `android_in_the_wild`, `llava_plus`, and
  `agenttuning_mind2web`. The target's committed sample is text/screen-agent
  response data, not a structured action log. This is consistent with the
  nearest text/multimodal samples.
- `swe-gym_openhands_sampled_trajectories`: I compared it with `hybrid-gym`,
  `swe-smith`, and `scale_swe_distilled`. After fixture refresh, it follows the
  same OpenHands-style conventions for system context, user issue text, and
  canonical editor/terminal calls.
- `swe-play-trajectories`: I compared it with `coderforge_preview`,
  `logicstar_swe-star`, and `hybrid-gym`. The target keeps explicit `think`
  calls as tools and normalizes execution/editing calls, matching those SWE
  peers.
- `swe-smith`: I compared it with `scale_swe_distilled`,
  `nvidia_SWE-Zero-openhands-trajectories`, and `allenai_Sera-4.6-Lite-T2`. It
  is a reference-style OpenHands/SWE sample: system prompt, user issue, and
  canonical editor/terminal/finish calls all match the comparators.
- `synatra`: I compared it with `agenttuning_mind2web`, `screenagent`, and
  `openthoughts_tb_dev`. The sample is synthetic text/action-description data
  without formal tool metadata, so keeping it text-only is consistent.
- `toolmind`: I compared it with `dolci_instruct_sft_tool_use`,
  `toucan_1_5m`, and `enterpriselab`. It has real structured function calls and
  linked observations, which now matches Toucan after PR #291 and the other
  function-calling datasets.
- `toucan_1_5m`: I compared it with `dolci_instruct_sft_tool_use`, `toolmind`,
  and `enterpriselab`. The original inconsistency was concrete: target tool
  definitions and function observations existed, but assistant `function_call`
  records were not structured. PR #291 now makes Toucan match its peers.
- `turkingbench`: I compared it with `webarena_successful`, `mind2web`, and
  `go-browse-wa`. It uses browser actions (`goto`, `click`) and standalone
  observations under the same convention as the other browser datasets.
- `webarena_successful`: I compared it with `turkingbench`, `mind2web`, and
  `weblinx`. It normalizes final stop actions to `finish`, while browser
  observations use the same standalone-observation convention as the peers.
- `weblinx`: I compared it with `wonderbread`, `mind2web`, and `turkingbench`.
  It has full browser trajectories with structured `goto`/`click`/`type` and
  many observations, matching Wonderbread/Mind2Web design rather than text-only
  navigation datasets.
- `wonderbread`: I compared it with `weblinx`, `mind2web`, and `go-browse-wa`.
  It preserves browser actions plus screenshot/SOP/state provenance. The action
  and observation layout is consistent with the full web trajectory family.

## Remaining Judgment Calls

- Observation-only `agent` steps are still a cross-family schema convention for
  browser/mobile/web traces. PR #303 documents this, and issue #295 tracks a
  possible future schema/normalizer change.
- `openresearcher` keeps cursor-based `browser.open` as a domain/generic tool
  rather than forcing it into `goto`; I judged that consistent because the
  arguments and semantics differ from web-navigation `goto`.
- Text-only datasets that contain SQL, KG, or web instructions in plain text
  were not converted into tools unless the source sample had explicit tool-call
  structure. This matches the treatment of their closest comparators.
