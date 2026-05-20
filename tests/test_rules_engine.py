"""
tests/test_rules_engine.py - RulesEngine 单元测试 + RulesInjectorNode 集成测试

覆盖维度：
  - CRUD (add / get / update / toggle / delete / batch)
  - 优先级排序 & 启用过滤
  - System Prompt 构建（顺序、空规则）
  - RulesInjectorNode 集成（单条、多条、禁用、无规则）
  - 异常场景（空 content、超长、不存在 rule_id、非法 category）
  - 边界（不存在 session_id、close 幂等）
"""

import uuid
import pytest
from src.infrastructure.rules_engine import RulesEngine
from src.infrastructure.rule_templates import RuleCategory
from src.agent.langgraph.node_impls import RulesInjectorNode
from src.agent.langgraph.state import GraphState


# ═══════════════════════════════════════════════════════
# 模块级常量
# ═══════════════════════════════════════════════════════

TEST_SESSION_ID = "test-session-001"
TEST_CONTENT = "始终用中文回答"


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture
def db_path(tmp_path):
    """用 pytest 内置 tmp_path，自动清理"""
    return str(tmp_path / "test_rules.db")


@pytest.fixture
def engine(db_path):
    """RulesEngine 实例（function 级隔离）"""
    eng = RulesEngine(db_path=db_path)
    yield eng
    eng.close()


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def _add_rules(engine, session_id, *contents_and_priorities):
    """
    批量添加规则的快捷方法。

    用法: _add_rules(engine, sid, ("A", 5), ("B", 3))
    """
    rules = []
    for content, priority in contents_and_priorities:
        rules.append(engine.add_rule(session_id, content, priority=priority))
    return rules


# ═══════════════════════════════════════════════════════
# 测试 1：Rule CRUD
# ═══════════════════════════════════════════════════════

class TestRuleCRUD:

    # ── 正常路径 ────────────────────────────────

    def test_add_rule(self, engine):
        rule = engine.add_rule(
            session_id=TEST_SESSION_ID,
            content=TEST_CONTENT,
            priority=5,
            category=RuleCategory.GENERAL,
        )
        assert rule.session_id == TEST_SESSION_ID
        assert rule.content == TEST_CONTENT
        assert rule.priority == 5
        assert rule.category == RuleCategory.GENERAL
        assert rule.enabled is True
        # rule_id 应为合法 UUID
        uuid.UUID(rule.rule_id)

    def test_get_rule(self, engine):
        rule = engine.add_rule(TEST_SESSION_ID, "测试规则")
        retrieved = engine.get_rule(rule.rule_id)
        assert retrieved is not None
        assert retrieved.rule_id == rule.rule_id
        assert retrieved.content == "测试规则"

    def test_update_rule(self, engine):
        rule = engine.add_rule(TEST_SESSION_ID, "原始内容")
        updated = engine.update_rule(rule.rule_id, content="新内容", priority=4)
        assert updated is not None
        assert updated.content == "新内容"
        assert updated.priority == 4

    def test_toggle_rule(self, engine):
        rule = engine.add_rule(TEST_SESSION_ID, "测试")
        # 第一次切换：启用 → 禁用
        toggled = engine.toggle_rule(rule.rule_id)
        assert toggled is not None
        assert toggled.enabled is False
        # 第二次切换：禁用 → 启用
        toggled_again = engine.toggle_rule(rule.rule_id)
        assert toggled_again is not None
        assert toggled_again.enabled is True

    def test_delete_rule_soft(self, engine):
        """软删除：enabled 置为 False，记录仍存在"""
        rule = engine.add_rule(TEST_SESSION_ID, "测试")
        result = engine.delete_rule(rule.rule_id)
        assert result is True
        retrieved = engine.get_rule(rule.rule_id)
        assert retrieved is not None
        assert retrieved.enabled is False

    def test_add_rules_batch(self, engine):
        data = [
            {"session_id": TEST_SESSION_ID, "content": "规则A", "priority": 5},
            {"session_id": TEST_SESSION_ID, "content": "规则B", "priority": 3},
        ]
        rules = engine.add_rules_batch(data)
        assert len(rules) == 2
        assert rules[0].priority == 5
        assert rules[1].priority == 3
        assert all(r.enabled for r in rules)

    # ── 异常 & 边界 ─────────────────────────────

    @pytest.mark.parametrize("bad_content,expected_msg", [
        ("", "不能为空"),
        ("   ", "不能为空"),
        ("x" * 4001, "不能超过"),
    ])
    def test_add_rule_invalid_content_raises(self, engine, bad_content, expected_msg):
        with pytest.raises(ValueError, match=expected_msg):
            engine.add_rule(TEST_SESSION_ID, bad_content)

    def test_add_rule_invalid_category_raises(self, engine):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            engine.add_rule(TEST_SESSION_ID, "内容", category="invalid_cat")

    def test_get_nonexistent_rule(self, engine):
        assert engine.get_rule("nonexistent-id") is None

    def test_update_nonexistent_rule(self, engine):
        assert engine.update_rule("nonexistent-id", priority=1) is None

    def test_toggle_nonexistent_rule(self, engine):
        assert engine.toggle_rule("nonexistent-id") is None

    def test_delete_nonexistent_rule(self, engine):
        assert engine.delete_rule("nonexistent-id") is False


# ═══════════════════════════════════════════════════════
# 测试 2：get_enabled_rules — 优先级排序 & 启用过滤
# ═══════════════════════════════════════════════════════

class TestGetEnabledRules:

    def test_priority_order(self, engine):
        _add_rules(
            engine, TEST_SESSION_ID,
            ("低优先级", 1),
            ("高优先级", 5),
            ("中优先级", 3),
        )
        rules = engine.get_enabled_rules(TEST_SESSION_ID)
        assert len(rules) == 3
        assert rules[0].priority == 5
        assert rules[1].priority == 3
        assert rules[2].priority == 1

    def test_only_enabled_returned(self, engine):
        _add_rules(engine, TEST_SESSION_ID, ("启用规则", 3))
        disabled = engine.add_rule(TEST_SESSION_ID, "禁用规则", priority=5)
        engine.toggle_rule(disabled.rule_id)

        rules = engine.get_enabled_rules(TEST_SESSION_ID)
        assert len(rules) == 1
        assert rules[0].content == "启用规则"

    def test_unknown_session_returns_empty(self, engine):
        rules = engine.get_enabled_rules("nonexistent-session")
        assert rules == []


# ═══════════════════════════════════════════════════════
# 测试 3：build_system_prefix
# ═══════════════════════════════════════════════════════

class TestBuildSystemPrefix:

    def test_empty_rules(self, engine):
        prefix = engine.build_system_prefix(TEST_SESSION_ID)
        assert prefix == ""

    def test_unknown_session(self, engine):
        prefix = engine.build_system_prefix("nonexistent-session")
        assert prefix == ""

    def test_prefix_content_and_order(self, engine):
        """验证：内容包含 & 顺序严格按 priority 降序"""
        _add_rules(
            engine, TEST_SESSION_ID,
            ("P1", 1),
            ("P5", 5),
            ("P3", 3),
        )
        prefix = engine.build_system_prefix(TEST_SESSION_ID)
        lines = prefix.split("\n")
        assert len(lines) == 3

        # 第一条必须是 priority=5
        assert "P5" in lines[0]
        assert "优先级 5" in lines[0]
        # 第二条 priority=3
        assert "P3" in lines[1]
        assert "优先级 3" in lines[1]
        # 第三条 priority=1
        assert "P1" in lines[2]
        assert "优先级 1" in lines[2]

    def test_prefix_excludes_disabled(self, engine):
        _add_rules(engine, TEST_SESSION_ID, ("启用", 5))
        disabled = engine.add_rule(TEST_SESSION_ID, "禁用", priority=4)
        engine.toggle_rule(disabled.rule_id)

        prefix = engine.build_system_prefix(TEST_SESSION_ID)
        assert "启用" in prefix
        assert "禁用" not in prefix


# ═══════════════════════════════════════════════════════
# 测试 4：close 幂等性
# ═══════════════════════════════════════════════════════

class TestClose:

    def test_close_is_idempotent(self, engine):
        engine.close()
        # 第二次 close 不应抛异常
        engine.close()


# ═══════════════════════════════════════════════════════
# 测试 5：RulesInjectorNode 集成测试
# ═══════════════════════════════════════════════════════

class TestRulesInjectorNode:

    @pytest.fixture
    def node(self, engine):
        return RulesInjectorNode(rules_engine=engine)

    # ── 正常路径 ────────────────────────────────

    def test_inject_single_rule(self, engine, node):
        engine.add_rule(TEST_SESSION_ID, "始终用中文", priority=5)
        state: GraphState = {"session_id": TEST_SESSION_ID}
        result = node(state)
        assert result["system_prefix"] == "[general] (优先级 5) 始终用中文"

    def test_inject_multiple_rules_ordered(self, engine, node):
        """多条规则按优先级降序注入"""
        _add_rules(
            engine, TEST_SESSION_ID,
            ("先做代码审查", 3),
            ("始终用中文", 5),
            ("禁止输出敏感信息", 4),
        )
        state: GraphState = {"session_id": TEST_SESSION_ID}
        result = node(state)
        lines = result["system_prefix"].split("\n")
        assert len(lines) == 3
        assert "始终用中文" in lines[0]       # priority 5
        assert "禁止输出敏感信息" in lines[1]  # priority 4
        assert "先做代码审查" in lines[2]       # priority 3

    def test_inject_excludes_disabled(self, engine, node):
        """禁用规则不注入 system_prefix"""
        _add_rules(engine, TEST_SESSION_ID, ("启用规则", 3))
        disabled = engine.add_rule(TEST_SESSION_ID, "禁用规则", priority=5)
        engine.toggle_rule(disabled.rule_id)

        state: GraphState = {"session_id": TEST_SESSION_ID}
        result = node(state)
        assert "启用规则" in result["system_prefix"]
        assert "禁用规则" not in result["system_prefix"]

    # ── 边界 ────────────────────────────────────

    def test_no_session_id_returns_empty(self, node):
        state: GraphState = {}
        result = node(state)
        assert result == {}

    def test_no_enabled_rules_returns_empty_prefix(self, engine, node):
        state: GraphState = {"session_id": TEST_SESSION_ID}
        result = node(state)
        assert result["system_prefix"] == ""

if __name__ == "__main__":    pytest.main([__file__, "-v", "--tb=short"])