from typing import List

from pydantic import BaseModel, RootModel


class ScreenAgentItem(BaseModel):
    task_prompt: str
    send_prompt: str
    current_task: str
    LLM_response: str
    LLM_response_editer: str
    video_height: int
    video_width: int
    saved_image_name: str
    session_id: str
    task_prompt_en: str
    task_prompt_zh: str
    send_prompt_en: str
    send_prompt_zh: str
    actions: str
    LLM_response_editer_en: str
    LLM_response_editer_zh: str
    screenshot: str


class SchemaRaw(RootModel):
    root: List[ScreenAgentItem]
