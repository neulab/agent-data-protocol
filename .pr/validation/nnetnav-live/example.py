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


DATASET_NAME = 'nnetnav-live'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'openweb_6442'
ASSISTANT_INDEX = 3
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "You are given the following objective: Evaluate the limit of the expression (sin x - x)/x^3 as x approaches 0 using Wolfram Alpha.\\nExamine the current web page and perform the next appropriate action to move toward completing the objective.", "type": "text"}], "role": "user"}, {"content": [{"text": "URL: https://www.wolframalpha.com/\\n\\nRootWebArea \'Wolfram|Alpha: Computational Intelligence\', focused, url=\'https://www.wolframalpha.com/\'\\n\\t[51] banner \'\', visible\\n\\t\\t[52] navigation \'\', visible\\n\\t\\t\\t[53] list \'\', visible\\n\\t\\t\\t\\t[54] listitem \'\', visible\\n\\t\\t\\t\\t\\t[55] button \'UPGRADE TO PRO\', clickable, visible, hasPopup=\'menu\', expanded=False\\n\\t\\t\\t\\t\\t\\tStaticText \'UPGRADE TO PRO\'\\n\\t\\t\\t\\t\\t\\t[57] image \'\', visible\\n\\t\\t\\t\\t[58] listitem \'\', visible\\n\\t\\t\\t\\t\\t[59] button \'APPS\', clickable, visible, hasPopup=\'menu\', expanded=False\\n\\t\\t\\t\\t\\t\\tStaticText \'APPS\'\\n\\t\\t\\t\\t\\t\\t[61] image \'\', visible\\n\\t\\t\\t\\t[62] listitem \'\', visible\\n\\t\\t\\t\\t\\t[63] link \'TOUR\', clickable, visible, url=\'https://www.wolframalpha.com/tour\'\\n\\t\\t\\t\\t\\t\\tStaticText \'TOUR\'\\n\\t\\t\\t\\t[65] listitem \'\', visible\\n\\t\\t\\t\\t\\t[66] button \'Sign in\', clickable, visible\\n\\t\\t\\t\\t\\t\\tStaticText \'Sign in\'\\n\\t\\t\\t\\t[68] listitem \'\', visible\\n\\t\\t\\t\\t\\t[69] listitem \'\', visible\\n\\t\\t\\t\\t\\t\\t[71] button \'Language and theme selector\', clickable, visible\\n\\t\\t\\t\\t\\t\\t\\t[72] image \'\', visible\\n\\t\\t\\t\\t\\t\\t\\t[73] image \'\', visible\\n\\t[74] main \'\', visible\\n\\t\\t[78] SvgRoot \'From the makers of Wolfram Language and Mathematica\', visible\\n\\t\\t\\tgraphics-symbol \'\'\\n\\t\\t\\tgraphics-symbol \'\'\\n\\t\\t[79] link \'WolframAlpha logo\', clickable, visible, url=\'https://www.wolframalpha.com/\'\\n\\t\\t\\t[80] image \'\', visible\\n\\t\\tStaticText \'Enter what you want to calculate or know about\'\\n\\t\\t[89] textbox \'WolframAlpha input field\', clickable, visible, focused\\n\\t\\t[91] button \'Compute input button\', clickable, visible\\n\\t\\t\\t[92] image \'\', visible\\n\\t\\t[94] list \'\', visible\\n\\t\\t\\t[95] listitem \'\', visible\\n\\t\\t\\t\\t[96] button \'NATURAL LANGUAGE\', clickable, visible\\n\\t\\t\\t\\t\\t[97] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'NATURAL LANGUAGE\'\\n\\t\\t\\t[99] listitem \'\', visible\\n\\t\\t\\t\\t[100] button \'MATH INPUT\', clickable, visible\\n\\t\\t\\t\\t\\t[101] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'MATH INPUT\'\\n\\t\\t[104] navigation \'\', visible\\n\\t\\t\\t[105] button \'EXTENDED KEYBOARD\', clickable, visible\\n\\t\\t\\t\\t[106] image \'\', visible\\n\\t\\t\\t\\tStaticText \'EXTENDED KEYBOARD\'\\n\\t\\t\\t[108] link \'EXAMPLES\', clickable, visible, url=\'https://www.wolframalpha.com/examples\'\\n\\t\\t\\t\\t[109] image \'\', visible\\n\\t\\t\\t\\tStaticText \'EXAMPLES\'\\n\\t\\t\\t[111] button \'UPLOAD\', clickable, visible\\n\\t\\t\\t\\t[112] image \'\', visible\\n\\t\\t\\t\\tStaticText \'UPLOAD\'\\n\\t\\t\\t[114] button \'RANDOM\', clickable, visible\\n\\t\\t\\t\\t[115] image \'\', visible\\n\\t\\t\\t\\tStaticText \'RANDOM\'\\n\\t\\t[118] paragraph \'\', visible\\n\\t\\t\\tStaticText \'Compute expert-level answers using Wolfram’s breakthrough\'\\n\\t\\t\\tStaticText \'algorithms, knowledgebase and AI technology\'\\n\\t\\t[119] link \'Now available through Wolfram GPT\', clickable, visible, url=\'https://wolfr.am/chatgpt-plugin\'\\n\\t\\t\\t[120] image \'Now available through Wolfram GPT\', visible, url=\'https://www.wolframalpha.com/_next/static/images/chatGPT_1ht2FAQ_.png\'\\n\\t\\t[123] heading \'Mathematics\', visible\\n\\t\\t\\t[124] link \'Mathematics\', clickable, visible, url=\'https://www.wolframalpha.com/examples/mathematics\'\\n\\t\\t\\t\\tStaticText \'Mathematics\'\\n\\t\\t[127] list \'\'\\n\\t\\t\\t[128] listitem \'\', visible\\n\\t\\t\\t\\t[129] link \'Step-by-Step Solutions\', clickable, visible, url=\'https://www.wolframalpha.com/examples/pro-features/step-by-step-solutions\'\\n\\t\\t\\t\\t\\t[130] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Step-by-Step Solutions\'\\n\\t\\t\\t[132] listitem \'\', visible\\n\\t\\t\\t\\t[133] link \'Elementary Math\', clickable, visible, url=\'https://www.wolframalpha.com/examples/mathematics/elementary-math\'\\n\\t\\t\\t\\t\\t[134] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Elementary Math\'\\n\\t\\t\\t[136] listitem \'\', visible\\n\\t\\t\\t\\t[137] link \'Algebra\', clickable, visible, url=\'https://www.wolframalpha.com/examples/mathematics/algebra\'\\n\\t\\t\\t\\t\\t[138] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Algebra\'\\n\\t\\t\\t[140] listitem \'\', visible\\n\\t\\t\\t\\t[141] link \'Plotting & Graphics\', clickable, visible, url=\'https://www.wolframalpha.com/examples/mathematics/plotting-and-graphics\'\\n\\t\\t\\t\\t\\t[142] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Plotting & Graphics\'\\n\\t\\t\\t[144] listitem \'\'\\n\\t\\t\\t\\t[145] link \'Calculus & Analysis\', clickable, url=\'https://www.wolframalpha.com/examples/mathematics/calculus-and-analysis\'\\n\\t\\t\\t\\t\\t[146] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Calculus & Analysis\'\\n\\t\\t\\t[148] listitem \'\'\\n\\t\\t\\t\\t[149] link \'Geometry\', clickable, url=\'https://www.wolframalpha.com/examples/mathematics/geometry\'\\n\\t\\t\\t\\t\\t[150] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Geometry\'\\n\\t\\t\\t[152] listitem \'\'\\n\\t\\t\\t\\t[153] link \'Differential Equations\', clickable, url=\'https://www.wolframalpha.com/examples/mathematics/differential-equations\'\\n\\t\\t\\t\\t\\t[154] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Differential Equations\'\\n\\t\\t\\t[156] listitem \'\'\\n\\t\\t\\t\\t[157] link \'Statistics\', clickable, url=\'https://www.wolframalpha.com/examples/mathematics/statistics\'\\n\\t\\t\\t\\t\\t[158] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Statistics\'\\n\\t\\t\\t[160] listitem \'\'\\n\\t\\t\\t\\t[161] link \'More Topics »\', clickable, url=\'https://www.wolframalpha.com/examples/mathematics\'\\n\\t\\t\\t\\t\\t[162] image \'\'\\n\\t\\t[164] heading \'Science & Technology\', visible\\n\\t\\t\\t[165] link \'Science & Technology\', clickable, visible, url=\'https://www.wolframalpha.com/examples/science-and-technology\'\\n\\t\\t\\t\\tStaticText \'Science & Technology\'\\n\\t\\t[168] list \'\'\\n\\t\\t\\t[169] listitem \'\', visible\\n\\t\\t\\t\\t[170] link \'Units & Measures\', clickable, visible, url=\'https://www.wolframalpha.com/examples/science-and-technology/units-and-measures\'\\n\\t\\t\\t\\t\\t[171] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Units & Measures\'\\n\\t\\t\\t[173] listitem \'\', visible\\n\\t\\t\\t\\t[174] link \'Physics\', clickable, visible, url=\'https://www.wolframalpha.com/examples/science-and-technology/physics\'\\n\\t\\t\\t\\t\\t[175] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Physics\'\\n\\t\\t\\t[177] listitem \'\', visible\\n\\t\\t\\t\\t[178] link \'Chemistry\', clickable, visible, url=\'https://www.wolframalpha.com/examples/science-and-technology/chemistry\'\\n\\t\\t\\t\\t\\t[179] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Chemistry\'\\n\\t\\t\\t[181] listitem \'\', visible\\n\\t\\t\\t\\t[182] link \'Engineering\', clickable, visible, url=\'https://www.wolframalpha.com/examples/science-and-technology/engineering\'\\n\\t\\t\\t\\t\\t[183] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Engineering\'\\n\\t\\t\\t[185] listitem \'\'\\n\\t\\t\\t\\t[186] link \'Computational Sciences\', clickable, url=\'https://www.wolframalpha.com/examples/science-and-technology/computational-sciences\'\\n\\t\\t\\t\\t\\t[187] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Computational Sciences\'\\n\\t\\t\\t[189] listitem \'\'\\n\\t\\t\\t\\t[190] link \'Earth Sciences\', clickable, url=\'https://www.wolframalpha.com/examples/science-and-technology/earth-sciences\'\\n\\t\\t\\t\\t\\t[191] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Earth Sciences\'\\n\\t\\t\\t[193] listitem \'\'\\n\\t\\t\\t\\t[194] link \'Materials\', clickable, url=\'https://www.wolframalpha.com/examples/science-and-technology/materials\'\\n\\t\\t\\t\\t\\t[195] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Materials\'\\n\\t\\t\\t[197] listitem \'\'\\n\\t\\t\\t\\t[198] link \'Transportation\', clickable, url=\'https://www.wolframalpha.com/examples/science-and-technology/transportation\'\\n\\t\\t\\t\\t\\t[199] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Transportation\'\\n\\t\\t\\t[201] listitem \'\'\\n\\t\\t\\t\\t[202] link \'More Topics »\', clickable, url=\'https://www.wolframalpha.com/examples/science-and-technology\'\\n\\t\\t\\t\\t\\t[203] image \'\'\\n\\t\\t[205] heading \'Society & Culture\', visible\\n\\t\\t\\t[206] link \'Society & Culture\', clickable, visible, url=\'https://www.wolframalpha.com/examples/society-and-culture\'\\n\\t\\t\\t\\tStaticText \'Society & Culture\'\\n\\t\\t[209] list \'\'\\n\\t\\t\\t[210] listitem \'\', visible\\n\\t\\t\\t\\t[211] link \'People\', clickable, visible, url=\'https://www.wolframalpha.com/examples/society-and-culture/people\'\\n\\t\\t\\t\\t\\t[212] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'People\'\\n\\t\\t\\t[214] listitem \'\', visible\\n\\t\\t\\t\\t[215] link \'Arts & Media\', clickable, visible, url=\'https://www.wolframalpha.com/examples/society-and-culture/arts-and-media\'\\n\\t\\t\\t\\t\\t[216] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Arts & Media\'\\n\\t\\t\\t[218] listitem \'\', visible\\n\\t\\t\\t\\t[219] link \'Dates & Times\', clickable, visible, url=\'https://www.wolframalpha.com/examples/society-and-culture/dates-and-times\'\\n\\t\\t\\t\\t\\t[220] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Dates & Times\'\\n\\t\\t\\t[222] listitem \'\', visible\\n\\t\\t\\t\\t[223] link \'Words & Linguistics\', clickable, visible, url=\'https://www.wolframalpha.com/examples/society-and-culture/words-and-linguistics\'\\n\\t\\t\\t\\t\\t[224] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Words & Linguistics\'\\n\\t\\t\\t[226] listitem \'\'\\n\\t\\t\\t\\t[227] link \'Money & Finance\', clickable, url=\'https://www.wolframalpha.com/examples/society-and-culture/money-and-finance\'\\n\\t\\t\\t\\t\\t[228] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Money & Finance\'\\n\\t\\t\\t[230] listitem \'\'\\n\\t\\t\\t\\t[231] link \'Food & Nutrition\', clickable, url=\'https://www.wolframalpha.com/examples/society-and-culture/food-and-nutrition\'\\n\\t\\t\\t\\t\\t[232] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Food & Nutrition\'\\n\\t\\t\\t[234] listitem \'\'\\n\\t\\t\\t\\t[235] link \'Political Geography\', clickable, url=\'https://www.wolframalpha.com/examples/society-and-culture/political-geography\'\\n\\t\\t\\t\\t\\t[236] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Political Geography\'\\n\\t\\t\\t[238] listitem \'\'\\n\\t\\t\\t\\t[239] link \'History\', clickable, url=\'https://www.wolframalpha.com/examples/society-and-culture/history\'\\n\\t\\t\\t\\t\\t[240] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'History\'\\n\\t\\t\\t[242] listitem \'\'\\n\\t\\t\\t\\t[243] link \'More Topics »\', clickable, url=\'https://www.wolframalpha.com/examples/society-and-culture\'\\n\\t\\t\\t\\t\\t[244] image \'\'\\n\\t\\t[246] heading \'Everyday Life\', visible\\n\\t\\t\\t[247] link \'Everyday Life\', clickable, visible, url=\'https://www.wolframalpha.com/examples/everyday-life\'\\n\\t\\t\\t\\tStaticText \'Everyday Life\'\\n\\t\\t[250] list \'\'\\n\\t\\t\\t[251] listitem \'\', visible\\n\\t\\t\\t\\t[252] link \'Personal Health\', clickable, visible, url=\'https://www.wolframalpha.com/examples/everyday-life/personal-health\'\\n\\t\\t\\t\\t\\t[253] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Personal Health\'\\n\\t\\t\\t[255] listitem \'\', visible\\n\\t\\t\\t\\t[256] link \'Personal Finance\', clickable, visible, url=\'https://www.wolframalpha.com/examples/everyday-life/personal-finance\'\\n\\t\\t\\t\\t\\t[257] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Personal Finance\'\\n\\t\\t\\t[259] listitem \'\', visible\\n\\t\\t\\t\\t[260] link \'Surprises\', clickable, visible, url=\'https://www.wolframalpha.com/examples/everyday-life/surprises\'\\n\\t\\t\\t\\t\\t[261] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Surprises\'\\n\\t\\t\\t[263] listitem \'\', visible\\n\\t\\t\\t\\t[264] link \'Entertainment\', clickable, visible, url=\'https://www.wolframalpha.com/examples/everyday-life/entertainment\'\\n\\t\\t\\t\\t\\t[265] image \'\', visible\\n\\t\\t\\t\\t\\tStaticText \'Entertainment\'\\n\\t\\t\\t[267] listitem \'\'\\n\\t\\t\\t\\t[268] link \'Household Science\', clickable, url=\'https://www.wolframalpha.com/examples/everyday-life/household-science\'\\n\\t\\t\\t\\t\\t[269] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Household Science\'\\n\\t\\t\\t[271] listitem \'\'\\n\\t\\t\\t\\t[272] link \'Household Math\', clickable, url=\'https://www.wolframalpha.com/examples/everyday-life/household-math\'\\n\\t\\t\\t\\t\\t[273] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Household Math\'\\n\\t\\t\\t[275] listitem \'\'\\n\\t\\t\\t\\t[276] link \'Hobbies\', clickable, url=\'https://www.wolframalpha.com/examples/everyday-life/hobbies\'\\n\\t\\t\\t\\t\\t[277] image \'\'\\n\\t\\t\\t\\t\\tStaticText \'Hobbies\'\\n\\t\\t\\t[279] listitem \'\'\\n\\t\\t\\t\\t[280] link \\"Today\'s World\\", clickable, url=\'https://www.wolframalpha.com/examples/everyday-life/todays-world\'\\n\\t\\t\\t\\t\\t[281] image \'\'\\n\\t\\t\\t\\t\\tStaticText \\"Today\'s World\\"\\n\\t\\t\\t[283] listitem \'\'\\n\\t\\t\\t\\t[284] link \'More Topics »\', clickable, url=\'https://www.wolframalpha.com/examples/everyday-life\'\\n\\t\\t\\t\\t\\t[285] image \'\'\\n\\t\\t[286] link \'Made possible by the Wolfram Language —building on 35+ years of research\\\\xa0&\\\\xa0development » Wolfram|Alpha Wolfram|Alpha Natural Language Understanding Curated Data & Knowledge Dynamic Algorithmic Computation Computed Visual Computation\', clickable, url=\'https://www.wolframalpha.com/tour\'\\n\\t\\t\\tStaticText \'Made possible by the\'\\n\\t\\t\\t[288] link \'Wolfram Language\', clickable\\n\\t\\t\\t\\tStaticText \'Wolfram Language\'\\n\\t\\t\\tStaticText \'—building on 35+ years of\'\\n\\t\\t\\tStaticText\'research\\\\xa0&\\\\xa0development »\'\\n\\t\\t\\t[292] image \'Wolfram|Alpha\', url=\'https://www.wolframalpha.com/_next/static/images/spikey_15JBFdMn.svg\'\\n\\t\\t\\tStaticText \'Wolfram|Alpha\'\\n\\t\\t\\t[294] image \'\'\\n\\t\\t\\t[295] image \'\'\\n\\t\\t\\tStaticText \'Natural Language Understanding\'\\n\\t\\t\\t[297] image \'\'\\n\\t\\t\\t[298] image \'\'\\n\\t\\t\\tStaticText \'Curated Data & Knowledge\'\\n\\t\\t\\t[300] image \'\'\\n\\t\\t\\t[301] image \'\'\\n\\t\\t\\tStaticText \'Dynamic Algorithmic Computation\'\\n\\t\\t\\t[303] image \'\'\\n\\t\\t\\t[304] image \'\'\\n\\t\\t\\tStaticText \'Computed Visual Computation\'\\n\\t[306] contentinfo \'\'\\n\\t\\t[307] list \'\'\\n\\t\\t\\t[308] listitem \'\'\\n\\t\\t\\t\\t[309] link \'Pro\', clickable, url=\'https://www.wolframalpha.com/pro\'\\n\\t\\t\\t\\t\\tStaticText \'Pro\'\\n\\t\\t\\t[311] listitem \'\'\\n\\t\\t\\t\\t[312] link \'Mobile Apps\', clickable, url=\'https://products.wolframalpha.com/mobile\'\\n\\t\\t\\t\\t\\tStaticText \'Mobile Apps\'\\n\\t\\t\\t[314] listitem \'\'\\n\\t\\t\\t\\t[315] link \'Products\', clickable, url=\'https://products.wolframalpha.com/\'\\n\\t\\t\\t\\t\\tStaticText \'Products\'\\n\\t\\t\\t[317] listitem \'\'\\n\\t\\t\\t\\t[318] link \'Business\', clickable, url=\'https://products.wolframalpha.com/enterprise\'\\n\\t\\t\\t\\t\\tStaticText \'Business\'\\n\\t\\t\\t[320] listitem \'\'\\n\\t\\t\\t\\t[321] link \'API & Developer Solutions\', clickable, url=\'https://products.wolframalpha.com/api\'\\n\\t\\t\\t\\t\\tStaticText \'API & Developer Solutions\'\\n\\t\\t\\t[323] listitem \'\'\\n\\t\\t\\t\\t[324] link \'LLM Solutions\', clickable, url=\'https://www.wolfram.com/resources/tools-for-AIs/\'\\n\\t\\t\\t\\t\\tStaticText \'LLM Solutions\'\\n\\t\\t[326] list \'\'\\n\\t\\t\\t[327] listitem \'\'\\n\\t\\t\\t\\t[328] link \'Resources & Tools\', clickable, url=\'https://www.wolframalpha.com/resources\'\\n\\t\\t\\t\\t\\tStaticText \'Resources & Tools\'\\n\\t\\t\\t[330] listitem \'\'\\n\\t\\t\\t\\t[331] link \'About\', clickable, url=\'https://www.wolframalpha.com/about\'\\n\\t\\t\\t\\t\\tStaticText \'About\'\\n\\t\\t\\t[333] listitem \'\'\\n\\t\\t\\t\\t[334] link \'Contact\', clickable, url=\'https://www.wolframalpha.com/contact-us\'\\n\\t\\t\\t\\t\\tStaticText \'Contact\'\\n\\t\\t\\t[336] listitem \'\'\\n\\t\\t\\t\\t[337] link \'Connect\', clickable, url=\'https://www.wolframalpha.com/social\'\\n\\t\\t\\t\\t\\tStaticText \'Connect\'\\n\\t\\t\\t\\t\\t[339] list \'\'\\n\\t\\t\\t\\t\\t\\t[340] listitem \'\'\\n\\t\\t\\t\\t\\t\\t\\t[341] image \'\'\\n\\t\\t\\t\\t\\t\\t[342] listitem \'\'\\n\\t\\t\\t\\t\\t\\t\\t[343] image \'\'\\n\\t\\t\\t\\t\\t\\t[344] listitem \'\'\\n\\t\\t\\t\\t\\t\\t\\t[345] image \'\'\\n\\t\\t\\t\\t\\t\\t[346] listitem \'\'\\n\\t\\t\\t\\t\\t\\t\\t[347] image \'\'\\n\\t\\t[348] list \'\'\\n\\t\\t\\t[349] listitem \'\'\\n\\t\\t\\t\\tStaticText \'©\'\\n\\t\\t\\t\\tStaticText \'2025\'\\n\\t\\t\\t\\tStaticText \'Wolfram Alpha LLC\'\\n\\t\\t\\t[352] listitem \'\'\\n\\t\\t\\t\\t[353] link \'Terms\', clickable, url=\'https://www.wolframalpha.com/termsofuse\'\\n\\t\\t\\t\\t\\tStaticText \'Terms\'\\n\\t\\t\\t[355] listitem \'\'\\n\\t\\t\\t\\t[356] link \'Privacy\', clickable, url=\'https://www.wolfram.com/legal/privacy/wolfram\'\\n\\t\\t\\t\\t\\tStaticText \'Privacy\'\\n\\t\\t[358] list \'\'\\n\\t\\t\\t[359] listitem \'\'\\n\\t\\t\\t\\t[360] link \'wolfram.com\', clickable, url=\'https://www.wolfram.com/\'\\n\\t\\t\\t\\t\\t[361] image \'\'\\n\\t\\t\\t[362] listitem \'\'\\n\\t\\t\\t\\t[363] link \'wolfram.com\', clickable, url=\'https://www.wolfram.com/\'\\n\\t\\t\\t\\t\\tStaticText \'wolfram.com\'\\n\\t\\t\\t[365] listitem \'\'\\n\\t\\t\\t\\t[366] link \'Wolfram Language\', clickable, url=\'https://www.wolfram.com/language\'\\n\\t\\t\\t\\t\\tStaticText \'Wolfram Language\'\\n\\t\\t\\t[368] listitem \'\'\\n\\t\\t\\t\\t[369] link \'Mathematica\', clickable, url=\'https://www.wolfram.com/mathematica\'\\n\\t\\t\\t\\t\\tStaticText \'Mathematica\'\\n\\t\\t\\t[371] listitem \'\'\\n\\t\\t\\t\\t[372] link \'Wolfram Demonstrations\', clickable, url=\'https://demonstrations.wolfram.com/\'\\n\\t\\t\\t\\t\\tStaticText \'Wolfram Demonstrations\'\\n\\t\\t\\t[374] listitem \'\'\\n\\t\\t\\t\\t[375] link \'Wolfram for Education\', clickable, url=\'https://www.wolfram.com/education\'\\n\\t\\t\\t\\t\\tStaticText \'Wolfram for Education\'\\n\\t\\t\\t[377] listitem \'\'\\n\\t\\t\\t\\t[378] link \'MathWorld\', clickable, url=\'https://mathworld.wolfram.com/\'\\n\\t\\t\\t\\t\\tStaticText \'MathWorld\'", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "Let\'s think step-by-step. The current webpage is the Wolfram Alpha homepage, and we want to evaluate the limit of the expression (sin x - x)/x^3 as x approaches 0. To do this, we can type the expression into the Wolfram Alpha input field and let the website compute the result for us.\\n\\nLooking at the webpage, we see that the Wolfram Alpha input field is represented by the element with id [89]. So, we can use this id to type our expression into the input field.\\n\\nThe limit expression in Wolfram Alpha can be represented in a mathematical form as \\"limit ((sin x - x)/x^3) as x->0\\". This expression can be directly input into the Wolfram Alpha input field.", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"index\\": 89, \\"text\\": \\"limit ((sin x - x)/x^3) as x->0\\", \\"security_risk\\": \\"MEDIUM\\", \\"summary\\": \\"Let\'s think step-by-step. The current webpage is the Wolfram Alpha homepage, and we want to evaluate the limit of the ex\\"}", "name": "browser_type"}, "id": "call_000001_browser_type", "type": "function"}]}'
TOOL_SPECS_JSON = '[{"function": {"description": "Type text into an input element.", "name": "browser_type", "parameters": {"properties": {"index": {"description": "Index from the browser state output.", "type": "integer"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "text": {"description": "Text to type.", "type": "string"}}, "required": ["index", "text"], "type": "object"}}, "type": "function"}, {"function": {"description": "View, create, and edit files in plain-text format.", "name": "file_editor", "parameters": {"properties": {"command": {"enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "file_text": {"description": "Content for create operations.", "type": "string"}, "insert_line": {"description": "Line before which to insert.", "type": "integer"}, "new_str": {"description": "Replacement text.", "type": "string"}, "old_str": {"description": "Text to replace.", "type": "string"}, "path": {"description": "Absolute file path.", "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "view_range": {"description": "Optional inclusive line range for view.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Create, update, and inspect the task list for the current conversation.", "name": "task_tracker", "parameters": {"properties": {"command": {"enum": ["plan", "update", "view"], "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task_list": {"description": "Markdown task list content.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}, {"function": {"description": "Log a thought without obtaining new information or changing the environment.", "name": "think", "parameters": {"properties": {"summary": {"description": "Concise summary of what this action does.", "type": "string"}, "thought": {"description": "The thought to log.", "type": "string"}}, "required": ["thought"], "type": "object"}}, "type": "function"}]'


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
