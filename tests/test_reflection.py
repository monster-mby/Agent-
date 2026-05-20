"""
tests/test_reflection.py — 反思模式测试套件（优化版）

测试范围：
1. ReflectionSkill 单元测试（含重试逻辑）
2. should_continue_reflection 条件边测试（含边界）
3. 完整反思流水线集成测试（Mock LLM，含异常场景）

基于：
- graphs.py v3.0（build_should_continue + ReflectionConfig）
- state.py（ReflectionContext / Critique / CritiqueRecord 等）
- ReflectionSkill（execute → dict / execute_from_context → Pydantic）
"""

import pytest
from unittest.mock import Mock
from langgraph.graph import END  # ← 新增：导入 LangGraph 的 END 常量
from src.agent.langgraph.state import (
    GraphState,
    create_initial_state,
    ReflectionContext,
    Critique,
    CritiquePoint,
    Feedback,
    FeedbackSource,
    CritiqueRecord,
)
from src.agent.langgraph.graphs import (
    build_should_continue,
    ReflectionConfig,
)
from src.skills.custom.reflection_skills.reflection.skill import (
    ReflectionSkill,
    ReflectionInput,
    ReflectionOutput,
)


# ═══════════════════════════════════════════════════════════════
# 模块级常量和 Fixtures
# ═══════════════════════════════════════════════════════════════

# 基础 Mock 技能类（供 mock_skill_manager 和异常测试复用）
from src.skills.base.base_skill import BaseSkill
from typing import Dict, Any


class _MockRagSkill(BaseSkill):
    """Mock RAG 答案技能 — 返回固定答案"""
    name = "rag_answer"
    description = "Mock RAG 答案技能"

    def execute(self, input_data=None, **kwargs) -> Dict[str, Any]:
        return {"answer": "原始生成的答案"}


class _FailingRagSkill(BaseSkill):
    """Mock RAG 技能 — 总是抛出异常"""
    name = "rag_answer"
    description = "会失败的 Mock 技能"

    def execute(self, input_data=None, **kwargs) -> Dict[str, Any]:
        raise RuntimeError("技能执行失败")


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端 — 默认返回成功响应"""
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "修订后的文本内容"
    mock_response.usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    mock_response.model = "gpt-4"
    mock_llm.chat.return_value = mock_response
    return mock_llm


@pytest.fixture
def mock_skill_manager():
    """Mock SkillManager — 注册 _MockRagSkill"""
    from src.skills.base.skill_manager import SkillManager

    skill_manager = SkillManager()
    skill_manager.register(_MockRagSkill)
    return skill_manager


# ═══════════════════════════════════════════════════════════════
# 1. ReflectionSkill 单元测试
# ═══════════════════════════════════════════════════════════════

class TestReflectionSkill:
    """ReflectionSkill 单元测试"""

    # ── 正常路径 ─────────────────────────────────────────────

    def test_execute_success(self, mock_llm_client):
        """execute() 返回 dict，验证所有字段"""
        skill = ReflectionSkill(
            llm_client=mock_llm_client,
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
        )

        input_data = ReflectionInput(
            generation="原始文本",
            critique=Critique(
                points=[
                    CritiquePoint(
                        target_field="content",
                        severity="major",
                        description="内容不够准确",
                        suggested_fix="补充更多细节",
                    )
                ],
                overall_score=0.6,
                summary="需要改进准确性",
            ),
        )

        result = skill.execute(input_data)

        assert result["success"] is True
        assert result["revised_generation"] == "修订后的文本内容"
        assert result["model"] == "gpt-4"
        assert result["usage"]["total_tokens"] == 150
        assert result["retry_count"] == 0

        mock_llm_client.chat.assert_called_once()
        call_args = mock_llm_client.chat.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["model"] == "gpt-4"

    def test_execute_from_context(self, mock_llm_client):
        """execute_from_context() 返回 ReflectionOutput Pydantic 模型"""
        skill = ReflectionSkill(llm_client=mock_llm_client)

        ctx = ReflectionContext(
            refined_output="原始输出",
            feedback=Feedback(
                content="外部反馈",
                source=FeedbackSource(source_type="auto", source_id="test"),
            ),
            critique=Critique(
                points=[
                    CritiquePoint(
                        target_field="content",
                        severity="minor",
                        description="需要改进",
                    )
                ],
                overall_score=0.7,
            ),
        )

        result = skill.execute_from_context(ctx)

        assert result.success is True
        assert result.revised_generation == "修订后的文本内容"

    # ── 异常路径 ─────────────────────────────────────────────

    def test_execute_client_not_configured(self):
        """LLM 客户端未注入 → 返回success=False"""
        skill = ReflectionSkill()

        input_data = ReflectionInput(
            generation="原始文本",
            critique=Critique(points=[]),
        )

        result = skill.execute(input_data)

        assert result["success"] is False
        assert "LLM 客户端未配置" in result["error"]

    def test_execute_retry_logic(self):
        """第一次调用失败，第二次成功 → retry_count=1, call_count=2"""
        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            ConnectionError("网络错误"),
            Mock(content="成功", usage={}, model="gpt-4"),
        ]

        skill = ReflectionSkill(llm_client=mock_llm, max_retries=2)

        input_data = ReflectionInput(
            generation="测试",
            critique=Critique(points=[]),
        )

        result = skill.execute(input_data)

        assert result["success"] is True
        assert result["retry_count"] == 1
        assert mock_llm.chat.call_count == 2


# ═══════════════════════════════════════════════════════════════
# 2. should_continue_reflection 条件边测试
#    （通过 build_should_continue 工厂闭包，覆盖所有停止条件）
# ═══════════════════════════════════════════════════════════════

class TestShouldContinueReflection:
    """条件边函数测试 — 覆盖三类停止条件 + 边界"""

    # ── 停止条件 1：迭代次数 ─────────────────────────────────

    def test_stop_at_max_iterations(self):
        """iteration >= max_iterations → END"""
        config = ReflectionConfig(target_skill="test", max_iterations=2)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=2,
        )

        assert fn(state) == END

    def test_continue_below_max_iterations(self):
        """iteration < max_iterations → reviser（边界：恰好小 1）"""
        config = ReflectionConfig(target_skill="test", max_iterations=2)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=1,
        )

        assert fn(state) == "reviser"

    # ── 停止条件 2：质量分数 ─────────────────────────────────

    def test_stop_at_high_quality(self):
        """overall_score >= threshold → END"""
        config = ReflectionConfig(target_skill="test", quality_threshold=0.8)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=0,
            critique=Critique(points=[], overall_score=0.85),
        )

        assert fn(state) == END

    def test_continue_below_quality_threshold(self):
        """overall_score 略低于 threshold → reviser（边界：0.79 < 0.8）"""
        config = ReflectionConfig(target_skill="test", quality_threshold=0.8)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=0,
            critique=Critique(points=[], overall_score=0.79),
        )

        assert fn(state) == "reviser"

    def test_continue_when_critique_is_none(self):
        """critique 为 None → 质量检查跳过，继续循环"""
        config = ReflectionConfig(target_skill="test", quality_threshold=0.8)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=0,
            # critique 未设置，默认为 None
        )

        assert fn(state) == "reviser"

    # ── 停止条件 3：收敛检测 ─────────────────────────────────

    def test_stop_at_convergence(self):
        """improvement_score < threshold → END"""
        config = ReflectionConfig(target_skill="test", convergence_threshold=0.05)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        ctx = ReflectionContext(
            refined_output="测试输出",
            iteration=1,
            improvement_score=0.03,
        )
        ctx.critique_history = [
            CritiqueRecord(critique=Critique(points=[]), iteration=0),
            CritiqueRecord(critique=Critique(points=[]), iteration=1),
        ]
        state["reflection_context"] = ctx

        assert fn(state) == END

    def test_continue_when_not_converged(self):
        """improvement_score 大 → 不收敛，继续"""
        config = ReflectionConfig(target_skill="test", convergence_threshold=0.05)
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        ctx = ReflectionContext(
            refined_output="测试输出",
            iteration=1,
            improvement_score=0.10,
        )
        ctx.critique_history = [
            CritiqueRecord(critique=Critique(points=[]), iteration=0),
            CritiqueRecord(critique=Critique(points=[]), iteration=1),
        ]
        state["reflection_context"] = ctx

        assert fn(state) == "reviser"

    # ── 综合场景 ─────────────────────────────────────────────

    def test_continue_reflection_all_conditions_fail(self):
        """三项条件均不满足 → 继续循环"""
        config = ReflectionConfig(target_skill="test")
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        state["reflection_context"] = ReflectionContext(
            refined_output="测试输出",
            iteration=0,
            critique=Critique(
                points=[
                    CritiquePoint(
                        target_field="content",
                        severity="major",
                        description="需要改进",
                    )
                ],
                overall_score=0.6,
            ),
        )

        assert fn(state) == "reviser"

    def test_no_reflection_context(self):
        """reflection_context 缺失 → 安全停止"""
        config = ReflectionConfig(target_skill="test")
        fn = build_should_continue(config)

        state = create_initial_state("test input")
        # 不设置 reflection_context

        assert fn(state) == END


# ═══════════════════════════════════════════════════════════════
# 3. 集成测试（完整反思流水线）
# ═══════════════════════════════════════════════════════════════

class TestReflectionPipelineIntegration:
    """完整反思流水线集成测试（Mock LLM + Mock SkillManager）"""

    def test_one_iteration_then_quality_passes(
        self, mock_llm_client, mock_skill_manager
    ):
        """正常流程：Critic→Reviser→Critic，质量达标后停止"""
        from src.agent.langgraph.graphs import run_reflection_pipeline

        mock_llm_client.chat.side_effect = [
            Mock(
                content='{"summary":"需改进","points":[{"target_field":"content","severity":"major","description":"准确性不足","suggested_fix":"补充细节"}],"overall_score":0.6}',
                usage={},
                model="gpt-4",
            ),  # Critic 1
            Mock(content="修订后的改进文本", usage={}, model="gpt-4"),  # Reviser
            Mock(
                content='{"summary":"已提升","points":[],"overall_score":0.85}',
                usage={},
                model="gpt-4",
            ),  # Critic 2 → 达标
        ]

        config = ReflectionConfig(
            target_skill="rag_answer",
            model="gpt-4",
            output_field="answer",
            max_iterations=2,
            quality_threshold=0.8,
        )

        result = run_reflection_pipeline(
            input_text="测试输入",
            skill_manager=mock_skill_manager,
            llm_client=mock_llm_client,
            config=config,
            session_id="test-int-001",
        )

        ctx = result.get("reflection_context")
        assert ctx is not None
        assert ctx.iteration == 1
        assert ctx.refined_output == "修订后的改进文本"
        assert ctx.critique is not None
        assert ctx.critique.overall_score == 0.85
        assert ctx.status == "critiquing"
        assert mock_llm_client.chat.call_count == 3

    def test_max_iterations_reached(
        self, mock_llm_client, mock_skill_manager
    ):
        """质量始终不达标 → 迭代到 max_iterations 后强制停止"""
        from src.agent.langgraph.graphs import run_reflection_pipeline

        # 始终低质量评分
        low_score_json = '{"summary":"差","points":[],"overall_score":0.5}'
        mock_llm_client.chat.side_effect = [
            Mock(content=low_score_json, usage={}, model="gpt-4"),  # Critic 1
            Mock(content="修订版1", usage={}, model="gpt-4"),        # Reviser 1
            Mock(content=low_score_json, usage={}, model="gpt-4"),  # Critic 2
            Mock(content="修订版2", usage={}, model="gpt-4"),        # Reviser 2
            Mock(content=low_score_json, usage={}, model="gpt-4"),  # Critic 3 → 达到 max
        ]

        config = ReflectionConfig(
            target_skill="rag_answer",
            model="gpt-4",
            max_iterations=2,
        )

        result = run_reflection_pipeline(
            input_text="测试输入",
            skill_manager=mock_skill_manager,
            llm_client=mock_llm_client,
            config=config,
            session_id="test-int-002",
        )

        ctx = result.get("reflection_context")
        assert ctx.iteration == 2
        assert ctx.status == "critiquing"
        assert mock_llm_client.chat.call_count == 5

    def test_json_parse_failure_graceful(
        self, mock_llm_client, mock_skill_manager
    ):
        """Critic 返回无效 JSON → 使用默认批评，不崩溃"""
        from src.agent.langgraph.graphs import run_reflection_pipeline

        # 确保 Mock 返回的是 Mock 对象（模拟 LLM 响应）
        mock_response = Mock()
        mock_response.content = "这不是有效的 JSON"
        mock_response.usage = {}
        mock_response.model = "gpt-4"
        mock_llm_client.chat.return_value = mock_response  # ← 修复：使用 return_value 而非 side_effect

        config = ReflectionConfig(
            target_skill="rag_answer",
            model="gpt-4",
            max_iterations=1,
        )

        result = run_reflection_pipeline(
            input_text="测试输入",
            skill_manager=mock_skill_manager,
            llm_client=mock_llm_client,
            config=config,
            session_id="test-int-003",
        )

        ctx = result.get("reflection_context")
        assert ctx is not None
        assert ctx.critique is not None
        assert "解析失败" in ctx.critique.summary

    def test_skill_execution_failure_raises(
        self, mock_llm_client
    ):
        """目标技能执行失败 → 抛出异常"""
        from src.agent.langgraph.graphs import run_reflection_pipeline
        from src.skills.base.skill_manager import SkillManager

        skill_manager = SkillManager()
        skill_manager.register(_FailingRagSkill)

        config = ReflectionConfig(target_skill="rag_answer", model="gpt-4")

        with pytest.raises(Exception):
            run_reflection_pipeline(
                input_text="测试输入",
                skill_manager=skill_manager,
                llm_client=mock_llm_client,
                config=config,
                session_id="test-int-004",
            )
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])