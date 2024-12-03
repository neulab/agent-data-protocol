from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime
# from litellm import ModelResponse


class Inputs(BaseModel):
    task: str = ""
    class Config:
        extra = "allow"

class Outputs(BaseModel):
    content: str = ""
    class Config:
        extra = "allow"

class Args(BaseModel):
    action_suffix: Optional[str] = None
    agent: Optional[str] = None
    agent_state: Optional[str] = None
    background: Optional[bool] = None
    blocking: Optional[bool] = None
    browser_actions: Optional[str] = None
    browsergym_send_msg_to_user: Optional[str] = None
    code: Optional[str] = None
    command: Optional[str] = None
    confirmation_state: Optional[str] = None
    content: Optional[str] = None
    end: Optional[int] = None
    hidden: Optional[bool] = None
    images_urls: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None
    include_extra: Optional[bool] = None
    inputs: Optional[Inputs] = None
    is_confirmed: Optional[str] = None
    keep_prompt: Optional[bool] = None
    kernel_init_code: Optional[str] = None
    outputs: Optional[Outputs] = None
    path: Optional[str] = None
    plan: Optional[Union[str, list, dict]] = None
    start: Optional[int] = None
    thought: Optional[str] = None
    timestamp: Optional[datetime] = None # 2024-10-31T19:35:13.242Z
    wait_for_response: Optional[bool] = None
    # AGENT: Optional[str] = None
    # CUSTOM_LLM_MODEL: Optional[str] = None
    # CONFIRMATION_MODE: Optional[bool] = None
    # LANGUAGE: Optional[str] = None
    # LLM_API_KEY: Optional[str] = None
    # LLM_BASE_URL: Optional[str] = None
    # LLM_MODEL: Optional[str] = None
    # SECURITY_ANALYZER: Optional[str] = None
    # USING_CUSTOM_MODEL: Optional[bool] = None
    class Config:
        extra = "allow"


class Extras(BaseModel):
    active_page_index: Optional[int] = None
    agent_state: Optional[str] = None
    axtree_object: Optional[dict] = None
    code: Optional[str] = None
    command: Optional[str] = None
    command_id: Optional[int] = None
    dom_object: Optional[dict] = None
    error: Optional[bool] = None
    exit_code: Optional[int] = None
    extra_element_properties: Optional[dict] = None
    focused_element_bid: Optional[str] = None
    hidden: Optional[bool] = None
    interpreter_details: Optional[str] = None
    last_browser_action: Optional[str] = None
    last_browser_action_error: Optional[str] = None
    new_content: Optional[str] = None
    old_content: Optional[str] = None
    open_pages_urls: Optional[List[str]] = None
    outputs: Optional[Outputs] = None
    path: Optional[str] = None
    prev_exist: Optional[bool] = None
    screenshot: Optional[str] = None
    status_code: Optional[int] = None
    url: Optional[str] = None


# class ModelResponse(BaseModel):
#     id: Optional[str] = None
#     created: Optional[int] = None
#     model: Optional[str] = None
#     object: Optional[str] = None
#     system_fingerprint: Optional[str] = None
#     choices: Optional[List[dict]] = None
#     usage: Optional[dict] = None
#     service_tier: Optional[str] = None


class ToolCallMetadata(BaseModel):
    function_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    model_response: Optional[dict] = None
    total_calls_in_response: Optional[int] = None


class TrajectoryItem(BaseModel):
    action: Optional[str] = None
    args: Optional[Args] = None
    token: Optional[str] = None
    status: Optional[str] = None
    id: Optional[int] = None
    timestamp: Optional[datetime] = None # 2024-11-11T13:09:26.404714
    source: str
    message: str = ""
    observation: Optional[str] = None
    content: str = ""
    log: Optional[str] = None
    extras: Optional[Extras] = None
    cause: Optional[int] = None
    error: Optional[Union[str, bool]] = None
    error_code: Optional[int] = None
    timeout: Optional[int] = None
    tool_call_metadata: Optional[ToolCallMetadata] = None


class SchemaRaw(BaseModel):
    version: str
    email: Optional[str] = None
    polarity: str = ""
    feedback: str
    token: Optional[str] = None
    timestamp: str # 2024-11-11-13-27-12
    permissions: str
    trajectory: List[TrajectoryItem]
