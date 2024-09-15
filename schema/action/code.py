from pydantic import Field
from typing import Literal

from schema.action.action import Action


class CodeAction(Action):
    language: Literal['git', 'go', 'glsl', 'assembly', 'racket', 'svelte', 'breadcrumbs', 'txt', 'velocity', 'python4', 'julia', 'yml', 'javasript', 'css', 'protobuf', 'rust', 'netlogo', 'hcl', 'scss', 'vb', 'http', 'sas', 'dart', 'vue', 'ejs', 'dockerfile', 'html', 'cypher', 'dockefile', 'vhdl', 'as3', 'c', 'ocaml', 'swift', 'apache', 'shell', 'groovy', 'matlab', 'gherkin', 'vtl', 'brainfuck', 'graphql', 'commandline', 'postgres', 'vbs', 'code', 'delphi', 'docker', 'toml', 'lua', 'r', 'pgsql', 'plsql', 'plaintext', 'cs', 'nginx', 'regex', 'pseudo', 'pseudo_rust', 'array', 'xslt', 'npm', 'proto', 'javascript', 'kotlin', 'react', 'verilog', 'cpp', 'bat', 'js', 'ts', 'asm', 'angular', 'terraform', 'mongodb', 'make', 'php', 'powershell', 'sh', 'sass', 'scala', 'xaml', 'lisp', 'asp', 'java', 'py', 'vba', 'vbnet', 'cmd', 'objc', 'pinescript', 'batch', 'solidity', 'mongo', 'actionscript', 'cmake', 'xml', 'svg', 'vbscript', 'python3', 'node', 'haskell', 'sqlite', 'less', 'jquery', 'yaml', 'typescript', 'prolog', 'pseudocode', 'python', 'ini', 'golang', 'perl', 'applescript', 'bash', 'cython', 's', 'mdx', 'json', 'javascipt', 'ruby', 'cql', 'tsx', 'mysql', 'csharp', 'jenkins', 'jsx', 'fsharp', 'gradle', 'pseudo_code', 'latex', 'openqasm', 'output', 'pascal', 'sql', 'text', 'xpath'] = Field(
        ..., description="The language of the code to execute" # code_feedback is multilingual
    )
    content: str = Field(..., description="The code to execute")
    description: str | None = Field(
        ..., description="The description/thought provided for the action"
    )

