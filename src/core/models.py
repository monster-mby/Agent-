from __future__ import annotations
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    id: str = ""

class ChatResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
