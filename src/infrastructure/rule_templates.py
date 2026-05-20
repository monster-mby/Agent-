"""
src/infrastructure/rule_templates.py — 规则模板库

提供预置规则模板的定义、验证、按策略应用到规则引擎。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════
# 依赖的类型（避免循环引用）
# ═══════════════════════════════════════════════════════

class RuleCategory(str, Enum):
    """规则分类枚举 — 与 RulesEngine 中的定义一致"""
    GENERAL       = "general"
    CODE_REVIEW   = "code_review"
    TECH_WRITING  = "tech_writing"
    COMPLIANCE    = "compliance"
    TEACHING      = "teaching"


# ═══════════════════════════════════════════════════════
# 模板数据模型 (Pydantic 校验)
# ═══════════════════════════════════════════════════════

class TemplateRuleEntry(BaseModel):
    """单条模板规则 — 加载时立即校验合法性"""

    content: str = Field(..., min_length=1, max_length=4000, description="规则内容")
    priority: int = Field(..., ge=1, le=5, description="优先级 1-5，5 最高")
    category: RuleCategory = Field(..., description="规则分类")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content 不能为空或纯空白")
        return stripped


class TemplateManifest(BaseModel):
    """单个模板的完整元信息"""

    name: str = Field(..., description="模板唯一标识（枚举值 .value）")
    description: str = Field(default="", description="模板用途说明")
    rules: List[TemplateRuleEntry] = Field(..., min_length=1, description="规则列表")


# ═══════════════════════════════════════════════════════
# 应用策略枚举
# ═══════════════════════════════════════════════════════

class ApplyStrategy(str, Enum):
    APPEND             = "append"              # 直接追加（默认）
    OVERWRITE_CATEGORY = "overwrite_category"  # 覆盖同 category 的已有规则
    MERGE_DEDUP        = "merge_dedup"         # 合并去重（同 content 保留高 priority）


# ═══════════════════════════════════════════════════════
# 预置规则模板定义
# ═══════════════════════════════════════════════════════

_PRESET_TEMPLATES: Dict[str, TemplateManifest] = {}


def _register(m: TemplateManifest) -> TemplateManifest:
    """注册模板到内置字典，返回原对象方便链式赋值"""
    _PRESET_TEMPLATES[m.name] = m
    return m


_CODE_REVIEW = _register(TemplateManifest(
    name=RuleCategory.CODE_REVIEW.value,
    description="代码审查：检查逻辑错误、边界条件、安全漏洞，遵循 PEP 8 风格",
    rules=[
        TemplateRuleEntry(
            content="严格检查代码逻辑错误、边界条件和异常处理",
            priority=5,
            category=RuleCategory.CODE_REVIEW,
        ),
        TemplateRuleEntry(
            content="遵循 PEP 8 风格指南，保持代码一致性",
            priority=4,
            category=RuleCategory.CODE_REVIEW,
        ),
        TemplateRuleEntry(
            content="识别潜在安全漏洞（SQL注入、XSS、硬编码密钥等）",
            priority=5,
            category=RuleCategory.CODE_REVIEW,
        ),
    ],
))

_TECH_WRITING = _register(TemplateManifest(
    name=RuleCategory.TECH_WRITING.value,
    description="技术写作：清晰简洁、面向中级开发者、提供示例和场景",
    rules=[
        TemplateRuleEntry(
            content="使用清晰简洁的语言，避免冗长复杂的句式",
            priority=4,
            category=RuleCategory.TECH_WRITING,
        ),
        TemplateRuleEntry(
            content="面向中级开发者，平衡专业性与可读性",
            priority=3,
            category=RuleCategory.TECH_WRITING,
        ),
        TemplateRuleEntry(
            content="提供具体代码示例和实际应用场景",
            priority=4,
            category=RuleCategory.TECH_WRITING,
        ),
    ],
))

_COMPLIANCE = _register(TemplateManifest(
    name=RuleCategory.COMPLIANCE.value,
    description="合规审查：逐条对照法规、标注条款出处、识别风险点",
    rules=[
        TemplateRuleEntry(
            content="逐条对照相关法规条款，确保合规性",
            priority=5,
            category=RuleCategory.COMPLIANCE,
        ),
        TemplateRuleEntry(
            content="引用原文时标注具体条款编号和出处",
            priority=4,
            category=RuleCategory.COMPLIANCE,
        ),
        TemplateRuleEntry(
            content="识别潜在的合规风险点并给出改进建议",
            priority=5,
            category=RuleCategory.COMPLIANCE,
        ),
    ],
))

_TEACHING = _register(TemplateManifest(
    name=RuleCategory.TEACHING.value,
    description="教学指导：由浅入深、生活化类比、每知识点配实例",
    rules=[
        TemplateRuleEntry(
            content="由浅入深讲解，先概念后实践",
            priority=4,
            category=RuleCategory.TEACHING,
        ),
        TemplateRuleEntry(
            content="使用生活化类比帮助理解抽象概念",
            priority=3,
            category=RuleCategory.TEACHING,
        ),
        TemplateRuleEntry(
            content="每个知识点配至少一个具体示例",
            priority=4,
            category=RuleCategory.TEACHING,
        ),
    ],
))


# ═══════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════

RULE_TEMPLATES = _PRESET_TEMPLATES


def list_templates() -> List[Dict[str, Any]]:
    """
    列出所有可用模板的摘要信息（不包含完整规则内容）。

    Returns:
        [{"name": "code_review", "description": "...", "rule_count": 3}, ...]
    """
    return [
        {
            "name": tmpl.name,
            "description": tmpl.description,
            "rule_count": len(tmpl.rules),
        }
        for tmpl in _PRESET_TEMPLATES.values()
    ]


def get_template(template_name: str) -> Optional[TemplateManifest]:
    """
    获取指定模板的完整定义。

    Args:
        template_name: 模板名称（如 "code_review"）

    Returns:
        TemplateManifest 对象，不存在则返回 None
    """
    return _PRESET_TEMPLATES.get(template_name)


def apply_template(
    engine: "RulesEngine",
    session_id: str,
    template_name: str,
    *,
    created_by: str = "system",
    strategy: ApplyStrategy = ApplyStrategy.APPEND,
    default_priority: Optional[int] = None,
    variables: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    将规则模板应用到指定会话。

    Args:
        engine:       RulesEngine 实例
        session_id:   目标会话 ID
        template_name: 模板名称（"code_review" / "tech_writing" / "compliance" / "teaching"）
        created_by:   规则创建者标识
        strategy:     应用策略
                      - append:              直接追加（默认）
                      - overwrite_category:  删除同 category 已有规则后写入
                      - merge_dedup:         合并去重，同 content 保留高 priority
        default_priority: 若传入，覆盖模板中所有规则的 priority
        variables:    模板变量替换字典，如 {"language": "Python"}

    Returns:
        {
            "template": "code_review",
            "strategy": "append",
            "applied_count": 3,
            "session_id": "abc123",
            "rule_ids": [...]
        }

    Raises:
        ValueError: 模板名称不存在
    """
    manifest = _PRESET_TEMPLATES.get(template_name)
    if manifest is None:
        available = sorted(_PRESET_TEMPLATES.keys())
        raise ValueError(
            f"未知模板: '{template_name}'，可用模板: {available}"
        )

    # ── 1. 构建规则数据列表 ─────────────────
    rules_data: List[Dict[str, Any]] = []
    for entry in manifest.rules:
        content = entry.content
        # 模板变量替换
        if variables:
            for key, val in variables.items():
                content = content.replace(f"{{{key}}}", val)

        rules_data.append({
            "session_id": session_id,
            "content": content,
            "priority": default_priority if default_priority is not None else entry.priority,
            "category": entry.category,
            "created_by": created_by,
        })

    # ── 2. 按策略执行 ───────────────────────
    if strategy == ApplyStrategy.OVERWRITE_CATEGORY:
        # 收集所有涉及的 category，删除同 category 已有规则
        categories = {entry.category for entry in manifest.rules}
        for cat in categories:
            existing = engine.get_enabled_rules(session_id)
            for rule in existing:
                if rule.category == cat:
                    engine.delete_rule(rule.rule_id)

    elif strategy == ApplyStrategy.MERGE_DEDUP:
        existing_rules = engine.get_enabled_rules(session_id)
        existing_map: Dict[str, Any] = {}
        for r in existing_rules:
            existing_map[r.content] = r

        deduped: List[Dict[str, Any]] = []
        for new_rule in rules_data:
            old = existing_map.get(new_rule["content"])
            if old is not None:
                # 保留 priority 更高的
                if new_rule["priority"] > old.priority:
                    engine.update_rule(old.rule_id, priority=new_rule["priority"])
                # 无论是否更新，不重复添加
                continue
            deduped.append(new_rule)
        rules_data = deduped

    # ── 3. 批量写入 ─────────────────────────
    created = engine.add_rules_batch(rules_data)

    return {
        "template": template_name,
        "strategy": strategy.value,
        "applied_count": len(created),
        "session_id": session_id,
        "rule_ids": [r.rule_id for r in created],
    }
