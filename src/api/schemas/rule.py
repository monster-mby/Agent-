"""规则相关的请求/响应模型"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict

# 从基础设施层导入唯一源头，避免重复定义
from src.infrastructure.rules_engine import RuleCategory


# ---------- 请求模型 ----------
class CreateRuleRequest(BaseModel):
    """创建规则请求"""
    content: str = Field(..., min_length=1, max_length=4000, description="规则内容")
    priority: int = Field(default=3, ge=1, le=5, description="优先级 1-5，5 最高")
    category: RuleCategory = Field(default=RuleCategory.GENERAL, description="规则分类")


class UpdateRuleRequest(BaseModel):
    """更新规则请求（仅需提供要修改的字段）"""
    content: Optional[str] = Field(None, min_length=1, max_length=4000)
    priority: Optional[int] = Field(None, ge=1, le=5)
    category: Optional[RuleCategory] = None
    enabled: Optional[bool] = None


class CreateRuleFromTemplateRequest(BaseModel):
    """从模板创建规则请求"""
    template_name: str = Field(..., min_length=1, description="模板名称")
    overrides: Optional[CreateRuleRequest] = Field(None, description="可覆盖模板中的字段")


# ---------- 响应模型 ----------
class RuleListResponse(BaseModel):
    """规则列表响应（精简字段，不含创建者）"""
    rule_id: str
    session_id: str
    content: str
    priority: int
    category: RuleCategory
    enabled: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RuleDetailResponse(RuleListResponse):
    """规则详情响应（含完整信息）"""
    created_by: str
    updated_at: Optional[datetime] = None


class RuleTemplateResponse(BaseModel):
    """规则模板响应"""
    name: str
    description: str
    category: RuleCategory
    default_priority: int
    content_template: str
    rules_count: int

    model_config = ConfigDict(from_attributes=True)