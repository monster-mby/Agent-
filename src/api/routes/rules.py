"""规则路由 – 嵌套在 /sessions/{session_id} 下"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_owned_session, get_rules_engine
from src.api.schemas.rule import (
    CreateRuleRequest,
    UpdateRuleRequest,
    RuleListResponse,
    RuleDetailResponse,
)
from src.infrastructure.session_manager import Session as SessionSchema
from src.infrastructure.rules_engine import RulesEngine

router = APIRouter(prefix="/sessions/{session_id}/rules", tags=["rules"])


@router.post("", status_code=201, response_model=RuleDetailResponse)
def create_rule(
    payload: CreateRuleRequest,
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """创建规则（持久化到数据库）"""
    rule = re.add_rule(
        session_id=session.session_id,
        content=payload.content,
        priority=payload.priority,
        category=payload.category.value if hasattr(payload.category, 'value') else payload.category,
    )
    return RuleDetailResponse.model_validate(rule)


@router.get("", response_model=list[RuleListResponse])
def list_rules(
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """列出当前会话的所有启用规则"""
    rules = re.get_enabled_rules(session.session_id)
    return [RuleListResponse.model_validate(r) for r in rules]


@router.get("/{rule_id}", response_model=RuleDetailResponse)
def get_rule(
    rule_id: str,
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """获取单个规则详情"""
    rule = re.get_rule(rule_id)
    if not rule or rule.session_id != session.session_id:
        raise HTTPException(404, "Rule not found")
    return RuleDetailResponse.model_validate(rule)


@router.patch("/{rule_id}", response_model=RuleDetailResponse)
def update_rule(
    rule_id: str,
    payload: UpdateRuleRequest,
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """更新规则（部分更新）"""
    rule = re.get_rule(rule_id)
    if not rule or rule.session_id != session.session_id:
        raise HTTPException(404, "Rule not found")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    updated_rule = re.update_rule(rule_id, **updates)
    if not updated_rule:
        raise HTTPException(404, "Rule not found after update")
    return RuleDetailResponse.model_validate(updated_rule)


@router.delete("/{rule_id}", status_code=204)
def delete_rule(
    rule_id: str,
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """删除规则"""
    rule = re.get_rule(rule_id)
    if not rule or rule.session_id != session.session_id:
        raise HTTPException(404, "Rule not found")

    success = re.delete_rule(rule_id)
    if not success:
        raise HTTPException(404, "Rule not found or already deleted")


@router.post("/{rule_id}/toggle", response_model=RuleDetailResponse)
def toggle_rule(
    rule_id: str,
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """切换规则启用/禁用状态"""
    rule = re.get_rule(rule_id)
    if not rule or rule.session_id != session.session_id:
        raise HTTPException(404, "Rule not found")

    updated_rule = re.toggle_rule(rule_id)
    return RuleDetailResponse.model_validate(updated_rule)
