from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence, Self

from pydantic import SecretStr

from openhands.sdk import ImageContent, LLM, Message, Observation, TextContent, ToolDefinition
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.tool import Action, ToolAnnotations, ToolExecutor
from openhands.sdk.tool.builtins.finish import FinishTool
from openhands.sdk.tool.builtins.think import ThinkTool
from openhands.tools.browser_use.definition import (
    BROWSER_CLICK_DESCRIPTION,
    BROWSER_GET_CONTENT_DESCRIPTION,
    BROWSER_GET_STATE_DESCRIPTION,
    BROWSER_GO_BACK_DESCRIPTION,
    BROWSER_NAVIGATE_DESCRIPTION,
    BROWSER_SCROLL_DESCRIPTION,
    BROWSER_TYPE_DESCRIPTION,
    BrowserClickAction,
    BrowserClickTool,
    BrowserGetContentAction,
    BrowserGetContentTool,
    BrowserGetStateAction,
    BrowserGetStateTool,
    BrowserGoBackAction,
    BrowserGoBackTool,
    BrowserNavigateAction,
    BrowserNavigateTool,
    BrowserObservation,
    BrowserScrollAction,
    BrowserScrollTool,
    BrowserTypeAction,
    BrowserTypeTool,
)
from openhands.tools.file_editor.definition import (
    TOOL_DESCRIPTION as FILE_EDITOR_DESCRIPTION,
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
)
from openhands.tools.task_tracker.definition import (
    TASK_TRACKER_DESCRIPTION,
    TaskTrackerAction,
    TaskTrackerObservation,
    TaskTrackerTool,
)
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation, TerminalTool
from openhands.tools.terminal.descriptions import UNIX_TOOL_DESCRIPTION


DATASET_NAME = 'nnetnav-wa'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'webarena_openended_5777'
ASSISTANT_INDEX = 3
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "You are given the following objective: Search for information on how to use the \\"Software Architecture Group / My Machine Learning Project\\".\\nExamine the current web page and perform the next appropriate action to move toward completing the objective.", "type": "text"}], "role": "user"}, {"content": [{"text": "URL: http://ec2-18-116-228-190.us-east-2.compute.amazonaws.com:8023/software-architecture-grsoftware-architecture-groupoup/my-machine-learning-projectmy-machine-learning-project/-/blob/main/README.md\\n\\nTab 0 (current): README.md · main · Software Architecture Group / My Machine Learning Project · GitLab\\n\\n[4310] RootWebArea \'README.md · main · Software Architecture Group / My Machine Learning Project · GitLab\' focused: True\\n\\t[4364] link \'Skip to content\'\\n\\t[4506] link \'Dashboard\'\\n\\t[4652] button \'\' hasPopup: menu expanded: False\\n\\t[5755] textbox \'Search GitLab\' required: False\\n\\t[5564] generic \'Use the shortcut key / to start a search\'\\n\\t[4659] link \'Create new...\'\\n\\t[4661] link \'Issues\'\\n\\t\\t[5042] generic \'14 assigned issues\'\\n\\t[4662] link \'Merge requests\'\\n\\t\\t[5044] generic \'9 merge requests\'\\n\\t[4664] link \'To-Do List\'\\n\\t\\t[5060] generic \'Todos count\'\\n\\t[4665] link \'Help\'\\n\\t[4667] link \'Byte BlazeByte BlazeByte Blaze\'\\n\\t\\t[5094] img \'Byte BlazeByte BlazeByte Blaze\'\\n\\t[4365] complementary \'Project navigation\'\\n\\t\\t[4541] listitem \'My Machine Learning Project\'\\n\\t\\t\\t[4669] link \'My Machine Learning Project\'\\n\\t\\t[4670] link \'Project information\'\\n\\t\\t[4672] link \'Repository\'\\n\\t\\t[5604] link \'Files\'\\n\\t\\t[5605] link \'Commits\'\\n\\t\\t[5606] link \'Branches\'\\n\\t\\t[5607] link \'Tags\'\\n\\t\\t[5608] link \'Contributors\'\\n\\t\\t[5609] link \'Graph\'\\n\\t\\t[5610] link \'Compare\'\\n\\t\\t[4674] link \'Issues\'\\n\\t\\t[4676] link \'Merge requests\'\\n\\t\\t[4678] link \'CI/CD\'\\n\\t\\t[4680] link \'Security & Compliance\'\\n\\t\\t[4682] link \'Deployments\'\\n\\t\\t[4684] link \'Packages and registries\'\\n\\t\\t[4686] link \'Infrastructure\'\\n\\t\\t[4688] link \'Monitor\'\\n\\t\\t[4690] link \'Analytics\'\\n\\t\\t[4692] link \'Wiki\'\\n\\t\\t[4694] link \'Snippets\'\\n\\t\\t[4696] link \'Settings\'\\n\\t\\t[4497] button \'Collapse sidebar\'\\n\\t[4369] navigation \'Breadcrumbs\'\\n\\t\\t[4713] link \'Software Architecture Group\'\\n\\t\\t[4715] link \'My Machine Learning Project\'\\n\\t\\t[4717] link \'Repository\'\\n\\t[4316] main \'\'\\n\\t\\t[5832] link \'\'\\n\\t\\t[5839] link \'Initial commit\'\\n\\t\\t[5842] link \'Byte BlazeByte BlazeByte Blaze\'\\n\\t\\t[5843] StaticText\'authored \'\\n\\t\\t[5844] time \'Sep 18, 2024 7:14pm PDT\'\\n\\t\\t\\t[6026] StaticText \'5 hours ago\'\\n\\t\\t[5845] group \'\'\\n\\t\\t\\t[6028] StaticText \'a0714ddc\'\\n\\t\\t\\t[5847] button \'Copy commit SHA\' live: polite atomic: False relevant: additions text\\n\\t\\t[5433] button\'main\' hasPopup: menu expanded: False\\n\\t\\t[4320] navigation \'Files breadcrumb\'\\n\\t\\t\\t[5690] link\'my-machine-learning-projectmy-machine-learning-project\'\\n\\t\\t\\t[6249] StaticText \'/\'\\n\\t\\t\\t[6245] link \'README.md\'\\n\\t\\t[6030] link \'Find file\'\\n\\t\\t[6031] link \'Blame\'\\n\\t\\t[6032] link \'History\'\\n\\t\\t[6033] link \'Permalink\'\\n\\t\\t[5852] button \'\' hasPopup: menu expanded: False\\n\\t\\t[5861] StaticText \'README.md\'\\n\\t\\t[5855] button \'Copy file path\' live: polite atomic: False relevant: additions text\\n\\t\\t[5863] StaticText \'6.33 KiB\'\\n\\t\\t[5864] group \'\'\\n\\t\\t\\t[5868] button \'Display source\'\\n\\t\\t\\t[5869] button \'Display rendered file\'\\n\\t\\t[6097] link \'Open in Web IDE\'\\n\\t\\t[6098] button \'Toggle dropdown\' hasPopup: menu expanded: False\\n\\t\\t[5872] group \'\'\\n\\t\\t\\t[5874] button \'Replace\'\\n\\t\\t\\t[5875] button \'Delete\'\\n\\t\\t[5867] group \'\'\\n\\t\\t\\t[5876] button \'Copy file contents\' disabled: True\\n\\t\\t\\t[5877] link \'Open raw\'\\n\\t\\t\\t[5878] link \'Download\'\\n\\t\\t[5882] heading \'My Machine Learning Project\'\\n\\t\\t\\t[5927] link \'\' hidden: True\\n\\t\\t[5883] heading \'Getting started\'\\n\\t\\t\\t[5930] link \'\' hidden: True\\n\\t\\t[5933] StaticText \\"To make it easy for you to get started with GitLab, here\'s a list of recommended next steps.\\"\\n\\t\\t[5934] StaticText \'Already a pro? Just edit this README.md and make it your own. Want to make it easy? \'\\n\\t\\t[5935] link \'Use the template at the bottom\'\\n\\t\\t[5936] StaticText \'!\'\\n\\t\\t[5886] heading \'Add your files\'\\n\\t\\t\\t[5937] link \'\' hidden: True\\n\\t\\t[6108] checkbox \'\' disabled: True checked: false\\n\\t\\t[6109] link \'Create\'\\n\\t\\t[6110] StaticText\'or \'\\n\\t\\t[6111] link \'upload\'\\n\\t\\t[6112] StaticText\'files\'\\n\\t\\t[6113] checkbox \'\' disabled: True checked: false\\n\\t\\t[6114] link \'Add files using the command line\'\\n\\t\\t[6115] StaticText\'or push an existing Git repository with the following command:\'\\n\\t\\t[5942] Pre \'\'\\n\\t\\t\\t[6211] StaticText \'cd existing_repo\'\\n\\t\\t\\t[6118] StaticText \'\\\\n\'\\n\\t\\t\\t[6212] StaticText \'git remote add origin http://ec2-18-116-228-190.us-east-2.compute.amazonaws.com:8023/software-architecture-grsoftware-architecture-groupoup/my-machine-learning-projectmy-machine-learning-project.git\'\\n\\t\\t\\t[6213] StaticText \'git branch -M main\'\\n\\t\\t\\t[6214] StaticText \'git push -uf origin main\'\\n\\t\\t[6124] button \'\'\\n\\t\\t[5889] heading \'Integrate with your tools\'\\n\\t\\t\\t[5944] link \'\' hidden: True\\n\\t\\t[6125] checkbox \'\' disabled: True checked: false\\n\\t\\t[6126] link \'Set up project integrations\'\\n\\t\\t[5891] heading \'Collaborate with your team\'\\n\\t\\t\\t[5948] link \'\' hidden: True\\n\\t\\t[6127] checkbox \'\' disabled: True checked: false\\n\\t\\t[6128] link \'Invite team members and collaborators\'\\n\\t\\t[6129] checkbox \'\' disabled: True checked: false\\n\\t\\t[6130] link \'Create a new merge request\'\\n\\t\\t[6131] checkbox \'\' disabled: True checked: false\\n\\t\\t[6132] link \'Automatically close issues from merge requests\'\\n\\t\\t[6133] checkbox \'\' disabled: True checked: false\\n\\t\\t[6134] link \'Enable merge request approvals\'\\n\\t\\t[6135] checkbox \'\' disabled: True checked: false\\n\\t\\t[6136] link \'Automatically merge when pipeline succeeds\'\\n\\t\\t[5893] heading \'Test and Deploy\'\\n\\t\\t\\t[5956] link \'\' hidden: True\\n\\t\\t[5959] StaticText \'Use the built-in continuous integration in GitLab.\'\\n\\t\\t[6137] checkbox \'\' disabled: True checked: false\\n\\t\\t[6138] link \'Get started with GitLab CI/CD\'\\n\\t\\t[6139] checkbox \'\' disabled: True checked: false\\n\\t\\t[6140] link \'Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)\'\\n\\t\\t[6141] checkbox \'\' disabled: True checked: false\\n\\t\\t[6142] link \'Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy\'\\n\\t\\t[6143] checkbox \'\' disabled: True checked: false\\n\\t\\t[6144] link \'Use pull-based deployments for improved Kubernetes management\'\\n\\t\\t[6145] checkbox \'\' disabled: True checked: false\\n\\t\\t[6146] link \'Set up protected environments\'\\n\\t\\t[5896] separator \'\' orientation: horizontal\\n\\t\\t[5897] heading \'Editing this README\'\\n\\t\\t\\t[5965] link \'\' hidden: True\\n\\t\\t[5968] StaticText \\"When you\'re ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to \\"\\n\\t\\t[5969] link\'makeareadme.com\'\\n\\t\\t[5970] StaticText\'for this template.\'\\n\\t\\t[5899] heading \'Suggestions for a good README\'\\n\\t\\t\\t[5971] link \'\' hidden: True\\n\\t\\t[5974] StaticText \'Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.\'\\n\\t\\t[5901] heading \'Name\'\\n\\t\\t\\t[5975] link \'\' hidden: True\\n\\t\\t[5978] StaticText \'Choose a self-explaining name for your project.\'\\n\\t\\t[5903] heading \'Description\'\\n\\t\\t\\t[5979] link \'\' hidden: True\\n\\t\\t[5982] StaticText \'Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.\'\\n\\t\\t[5905] heading \'Badges\'\\n\\t\\t\\t[5983] link \'\' hidden: True\\n\\t\\t[5986] StaticText \'On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.\'\\n\\t\\t[5907] heading \'Visuals\'\\n\\t\\t\\t[5987] link \'\' hidden: True\\n\\t\\t[5990] StaticText \\"Depending on what you are making, it can be a good idea to include screenshots or even a video (you\'ll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.\\"\\n\\t\\t[5909] heading \'Installation\'\\n\\t\\t\\t[5991] link \'\' hidden: True\\n\\t\\t[5994] StaticText \'Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.\'\\n\\t\\t[5911] heading \'Usage\'\\n\\t\\t\\t[5995] link \'\' hidden: True\\n\\t\\t[5998] StaticText \\"Use examples liberally, and show the expected output if you can. It\'s helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.\\"\\n\\t\\t[5913] heading \'Support\'\\n\\t\\t\\t[5999] link \'\' hidden: True\\n\\t\\t[6002] StaticText \'Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.\'\\n\\t\\t[5915] heading \'Roadmap\'\\n\\t\\t\\t[6003] link \'\' hidden: True\\n\\t\\t[6006] StaticText \'If you have ideas for releases in the future, it is a good idea to list them in the README.\'\\n\\t\\t[5917] heading \'Contributing\'\\n\\t\\t\\t[6007] link \'\' hidden: True\\n\\t\\t[6010] StaticText \'State if you are open to contributions and what your requirements are for accepting them.\'\\n\\t\\t[6011] StaticText \\"For people who want to make changes to your project, it\'s helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.\\"\\n\\t\\t[6012] StaticText \'You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.\'\\n\\t\\t[5921] heading \'Authors and acknowledgment\'\\n\\t\\t\\t[6013] link \'\' hidden: True\\n\\t\\t[6016] StaticText \'Show your appreciation to those who have contributed to the project.\'\\n\\t\\t[5923] heading \'License\'\\n\\t\\t\\t[6017] link \'\' hidden: True\\n\\t\\t[6020] StaticText \'For open source projects, say how it is licensed.\'\\n\\t\\t[5925] heading \'Project status\'\\n\\t\\t\\t[6021] link \'\' hidden: True\\n\\t\\t[6024] StaticText \'If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.\'", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "Let\'s think step-by-step. Currently, I am in the README.md file of the \\"Software Architecture Group / My Machine Learning Project\\" repository. I see a section titled \'Integrate with your tools\', which I believe may have useful information for integrating this project with other tools I am using. I will click on this heading to expand it and view its contents. The link has bid 5944.", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"index\\": 5944, \\"security_risk\\": \\"MEDIUM\\", \\"summary\\": \\"Let\'s think step-by-step. Currently, I am in the README.md file of the \\\\\\"Software Architecture Group / My Machine Learnin\\"}", "name": "browser_click"}, "id": "call_000001_browser_click", "type": "function"}]}'
TOOL_SPECS_JSON = '[{"function": {"description": "Click an element on the page by its current browser state index.", "name": "browser_click", "parameters": {"properties": {"index": {"description": "Index from the browser state output.", "type": "integer"}, "new_tab": {"description": "Whether to open navigation in a new tab.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["index"], "type": "object"}}, "type": "function"}, {"function": {"description": "View, create, and edit files in plain-text format.", "name": "file_editor", "parameters": {"properties": {"command": {"enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "file_text": {"description": "Content for create operations.", "type": "string"}, "insert_line": {"description": "Line before which to insert.", "type": "integer"}, "new_str": {"description": "Replacement text.", "type": "string"}, "old_str": {"description": "Text to replace.", "type": "string"}, "path": {"description": "Absolute file path.", "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "view_range": {"description": "Optional inclusive line range for view.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Create, update, and inspect the task list for the current conversation.", "name": "task_tracker", "parameters": {"properties": {"command": {"enum": ["plan", "update", "view"], "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task_list": {"description": "Markdown task list content.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}, {"function": {"description": "Log a thought without obtaining new information or changing the environment.", "name": "think", "parameters": {"properties": {"summary": {"description": "Concise summary of what this action does.", "type": "string"}, "thought": {"description": "The thought to log.", "type": "string"}}, "required": ["thought"], "type": "object"}}, "type": "function"}]'


class DatasetToolObservation(Observation):
    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.content:
            return self.content
        return [TextContent(text="Dataset validation tool result.")]


class DatasetToolExecutor(ToolExecutor):
    def __call__(self, action: Action, conversation=None) -> DatasetToolObservation:  # noqa: ARG002
        return DatasetToolObservation.from_text(
            f"Executed {action.__class__.__name__} for {DATASET_NAME}."
        )


class DatasetTool(ToolDefinition):
    @classmethod
    def create(cls, *args, **kwargs) -> list[Self]:
        return []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def class_name(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    text = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not text or text[0].isdigit():
        text = "Dataset" + text
    return text


def sdk_content(content: Any) -> list[TextContent | ImageContent]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContent(text=content)]
    blocks: list[TextContent | ImageContent] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            blocks.append(TextContent(text=item.get("text", "")))
        elif item.get("type") in {"image", "image_url"}:
            url = item.get("image_url", {}).get("url") if isinstance(item.get("image_url"), dict) else None
            if url:
                blocks.append(ImageContent(image_urls=[url]))
    return blocks


def sdk_message(raw: dict) -> Message:
    tool_calls = None
    if raw.get("tool_calls"):
        tool_calls = [
            MessageToolCall(
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                arguments=tool_call["function"].get("arguments") or "{}",
                origin="completion",
            )
            for tool_call in raw["tool_calls"]
        ]
    return Message(
        role=raw["role"],
        content=sdk_content(raw.get("content")),
        tool_calls=tool_calls,
        tool_call_id=raw.get("tool_call_id"),
        name=raw.get("name"),
        reasoning_content=raw.get("reasoning_content"),
    )


def native_tool(name: str) -> ToolDefinition | None:
    if name == "finish":
        return FinishTool.create()[0]
    if name == "think":
        return ThinkTool.create()[0]
    if name == "terminal":
        return TerminalTool(
            description=UNIX_TOOL_DESCRIPTION,
            action_type=TerminalAction,
            observation_type=TerminalObservation,
            annotations=ToolAnnotations(
                title="terminal",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )
    if name == "file_editor":
        return FileEditorTool(
            description=FILE_EDITOR_DESCRIPTION,
            action_type=FileEditorAction,
            observation_type=FileEditorObservation,
            annotations=ToolAnnotations(
                title="file_editor",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=False,
            ),
        )
    if name == "task_tracker":
        return TaskTrackerTool(
            description=TASK_TRACKER_DESCRIPTION,
            action_type=TaskTrackerAction,
            observation_type=TaskTrackerObservation,
            annotations=ToolAnnotations(
                title="task_tracker",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
    browser_tools: dict[str, tuple[type[ToolDefinition], type[Action], str, bool]] = {
        "browser_navigate": (BrowserNavigateTool, BrowserNavigateAction, BROWSER_NAVIGATE_DESCRIPTION, False),
        "browser_click": (BrowserClickTool, BrowserClickAction, BROWSER_CLICK_DESCRIPTION, False),
        "browser_type": (BrowserTypeTool, BrowserTypeAction, BROWSER_TYPE_DESCRIPTION, False),
        "browser_get_state": (BrowserGetStateTool, BrowserGetStateAction, BROWSER_GET_STATE_DESCRIPTION, True),
        "browser_get_content": (BrowserGetContentTool, BrowserGetContentAction, BROWSER_GET_CONTENT_DESCRIPTION, True),
        "browser_scroll": (BrowserScrollTool, BrowserScrollAction, BROWSER_SCROLL_DESCRIPTION, False),
        "browser_go_back": (BrowserGoBackTool, BrowserGoBackAction, BROWSER_GO_BACK_DESCRIPTION, False),
    }
    if name in browser_tools:
        tool_cls, action_type, description, read_only = browser_tools[name]
        return tool_cls(
            description=description,
            action_type=action_type,
            observation_type=BrowserObservation,
            annotations=ToolAnnotations(
                title=name,
                readOnlyHint=read_only,
                destructiveHint=False,
                idempotentHint=read_only,
                openWorldHint=True,
            ),
        )
    return None


def custom_tool(spec: dict) -> ToolDefinition:
    function = spec["function"]
    name = function["name"]
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    action_type = Action.from_mcp_schema(f"{class_name(name)}Action", parameters)
    tool_cls = type(f"{class_name(name)}Tool", (DatasetTool,), {"name": name})
    return tool_cls(
        description=function.get("description") or f"Dataset-specific tool {name}.",
        action_type=action_type,
        observation_type=DatasetToolObservation,
        executor=DatasetToolExecutor(),
    )


def validation_tools() -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    seen: set[str] = set()
    for spec in json.loads(TOOL_SPECS_JSON):
        function = spec.get("function") or {}
        name = function.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        tools.append(native_tool(name) or custom_tool(spec))
    return tools


def latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise RuntimeError("SDK did not write a completion log")
    return logs[-1]


def main() -> None:
    root = repo_root()
    load_env_file(root / ".env")
    load_env_file(Path.home() / "work" / "agent-data-protocol" / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    prefix_messages = [sdk_message(message) for message in json.loads(PREFIX_MESSAGES_JSON)]
    log_dir = Path(tempfile.mkdtemp(prefix=f"{DATASET_NAME}-completion-"))
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url=os.getenv("LLM_BASE_URL"),
        log_completions=True,
        log_completions_folder=str(log_dir),
        max_output_tokens=512,
    )
    llm.completion(
        messages=prefix_messages,
        tools=validation_tools(),
        add_security_risk_prediction=True,
        temperature=0,
    )
    shutil.copyfile(latest_log(log_dir), Path(__file__).with_name("completion.json"))


if __name__ == "__main__":
    main()
