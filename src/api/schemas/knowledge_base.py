"""知识库 API 的请求/响应模型（适配扩展后的 ORM）"""
from datetime import datetime
from enum import StrEnum
from typing import Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

# 直接复用 ORM 中的枚举，避免重复定义
from src.infrastructure.kb_manager import IndexingStatus as KBStatus

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    limit: int
    offset: int

class CreateKBRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    embedding_model: str = Field(default="text-embedding-v3")

class UpdateKBRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)

class KBListResponse(BaseModel):
    kb_id: str
    name: str
    description: str = ""
    status: KBStatus
    document_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class KBDetailResponse(KBListResponse):
    embedding_model: str = "text-embedding-v3"
    created_by: str = "unknown"
    deleted_at: Optional[datetime] = None

# 文档相关
class DocumentUploadMeta(BaseModel):
    tags: Optional[List[str]] = None
    source_url: Optional[str] = None
    custom: Optional[dict] = None

class UpdateDocumentRequest(BaseModel):
    content: Optional[str] = Field(None, max_length=50_000)
    metadata: Optional[DocumentUploadMeta] = None

class DocumentResponse(BaseModel):
    doc_id: str
    kb_id: str
    file_name: str
    content_preview: str = ""
    metadata: dict | None = Field(
        default=None,
        validation_alias="doc_metadata",   # 从 ORM 的 doc_metadata 属性取值
        serialization_alias="metadata",    # 序列化时还是输出 "metadata"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

class ReindexTaskResponse(BaseModel):
    task_id: str
    status: str = "pending"
    message: str = "任务已提交"


class ReindexRequest(BaseModel):
    """重新索引请求"""
    force: bool = Field(default=False, description="强制重新索引（即使未变更）")