"""会话相关请求/响应模型（优化版 v2）"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    name: str = Field(default="未命名会话", min_length=1, max_length=200)
    knowledge_base_ids: List[str] = Field(
        default_factory=list,
        description="关联的知识库 ID 列表",
    )

    @field_validator("knowledge_base_ids")
    @classmethod
    def check_ids(cls, v):
        if not all(isinstance(i, str) and i.strip() for i in v):
            raise ValueError("每个 ID 必须为非空字符串")
        return v


class SessionSimpleResponse(BaseModel):
    """会话列表项响应（轻量）"""
    session_id: str
    user_id: str
    name: str
    knowledge_base_ids: List[str]
    langgraph_thread_id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class SessionDetailResponse(SessionSimpleResponse):
    """会话详情响应（含规则 ID 列表）"""
    rules: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """发送消息请求"""
    query: str = Field(..., min_length=1)
    knowledge_base_ids: Optional[List[str]] = Field(None)
    stream: bool = Field(False, description="是否使用流式响应")


class MessageResponse(BaseModel):
    """消息响应"""
    message_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int