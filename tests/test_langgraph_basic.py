"""
tests/test_langgraph_basic.py — LangGraph 基础功能测试（v3.0 适配版）

测试维度：
1. 图构建与编译（通过 run_pipeline 验证）
2. 节点执行（translator 单独 + translator→summarizer 串联）
3. config 传递链路（修复前会静默丢弃！）
4. 异常处理（输入验证 + 技能失败 + 配置无效）
5. State 结构完整性
6. 边界情况测试
7. Config 传递链路测试（针对已知 P0 bug）
8. RAG 图构建与执行（standard / graphrag / hybrid）

Mock 策略：
- 使用 fake SkillManager + fake Skill 隔离真实 I/O
- 确保测试快速、确定、不依赖外部服务
"""

import pytest
from copy import deepcopy
from typing import Any, Dict

from src.agent.langgraph.state import GraphState, create_initial_state
from src.agent.langgraph.node_impls import BaseNode, TranslatorNode
from src.agent.langgraph.graphs import RAGType


# ═══════════════════════════════════════════════════════════════
# 测试常量
# ═══════════════════════════════════════════════════════════════



SAMPLE_TEXT = "你好世界"
SAMPLE_SESSION = "test-session"
TRANSLATED_TEXT = "Hello, world!"
SUMMARY_TEXT = "A summary."
MEETING_SUMMARY = "Meeting summary."
DRAFTED_EMAIL = "Drafted email."


# ═══════════════════════════════════════════════════════════════
# Fake 对象 — 隔离真实 I/O
# ═══════════════════════════════════════════════════════════════

class FakeSkillOutput:
    """模拟技能返回的 Pydantic 模型（兼容 extract_output 逻辑）"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        """模拟 Pydantic 的 model_dump 行为"""
        return {k: v for k, v in self.__dict__.items()}


class FakeSkill:
    """模拟技能类（可控制 execute 行为）"""

    def __init__(self, return_value=None, should_fail=False):
        self.return_value = return_value
        self.should_fail = should_fail
        self.execute_calls: list = []

    def execute(self, input_data):
        self.execute_calls.append(input_data)
        if self.should_fail:
            raise RuntimeError("FakeSkill 模拟失败")
        return self.return_value or FakeSkillOutput(
            translated_text=TRANSLATED_TEXT,
            summary=SUMMARY_TEXT,
        )


class FakeSkillManager:
    """模拟 SkillManager（可注册任意技能）"""

    def __init__(self):
        self._registry: Dict[str, type] = {}

    def register(self, skill_name: str, skill_factory):
        self._registry[skill_name] = skill_factory

    def get(self, skill_name: str):
        return self._registry.get(skill_name)


def make_fake_skill(return_value=None, should_fail=False):
    """FakeSkill 工厂函数，减少重复构造代码"""
    return FakeSkill(return_value=return_value, should_fail=should_fail)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def fake_skill_manager():
    """预注册 translator + summarizer 的 FakeSkillManager（session 级复用）"""
    sm = FakeSkillManager()
    sm.register("translator", lambda: make_fake_skill(
        return_value=FakeSkillOutput(translated_text=TRANSLATED_TEXT)
    ))
    sm.register("text_summarizer", lambda: make_fake_skill(
        return_value=FakeSkillOutput(summary=SUMMARY_TEXT)
    ))
    return sm


@pytest.fixture(scope="session")
def failing_skill_manager():
    """translator 正常但 summarizer 会失败（session 级复用）"""
    sm = FakeSkillManager()
    sm.register("translator", lambda: make_fake_skill(
        return_value=FakeSkillOutput(translated_text="Hello!")
    ))
    sm.register("text_summarizer", lambda: make_fake_skill(should_fail=True))
    return sm


@pytest.fixture(scope="session")
def rag_skill_manager():
    """预注册 RAG 相关技能的 FakeSkillManager"""
    sm = FakeSkillManager()
    sm.register("query_rewrite", lambda: make_fake_skill(
        return_value=FakeSkillOutput(rewritten_query="优化后的查询")
    ))
    sm.register("vector_search", lambda: make_fake_skill(
        return_value=FakeSkillOutput(documents=[{"id": 1, "text": "doc1"}])
    ))
    sm.register("graphrag_searcher", lambda: make_fake_skill(
        return_value=FakeSkillOutput(graph_results=[{"entity": "A", "relation": "B"}])
    ))
    sm.register("rerank", lambda: make_fake_skill(
        return_value=FakeSkillOutput(reranked=[{"id": 1, "score": 0.95}])
    ))
    sm.register("rag_answer", lambda: make_fake_skill(
        return_value=FakeSkillOutput(answer="基于检索结果的回答")
    ))
    return sm


@pytest.fixture
def base_state():
    """最小有效初始状态"""
    return create_initial_state(input_data=SAMPLE_TEXT, session_id=SAMPLE_SESSION)


# ═══════════════════════════════════════════════════════════════
# 1. 图构建测试（通过 run_pipeline 验证）
# ═══════════════════════════════════════════════════════════════

class TestGraphBuilding:
    """图构建与编译（已迁移至 run_pipeline 验证）"""

    def test_invoke_pipeline_success(self, fake_skill_manager):
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text=SAMPLE_TEXT,
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": "en"},
                "summarizer_config": {},
            },
            skill_manager=fake_skill_manager,
        )
        assert result is not None
        assert result.get("error") is None

    def test_invoke_pipeline_with_custom_config(self, fake_skill_manager):
        """带配置构建图不报错"""
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text="你好世界",
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": "en"},
                "summarizer_config": {"max_length": "short"}
            },
            skill_manager=fake_skill_manager,
        )
        assert result is not None
        # ✅ 修复：FakeSkillOutput 经过 extract_output 后返回 dict
        assert isinstance(result.get("current_output"), dict)
        assert result["current_output"].get("summary") == "A summary."

    def test_invalid_pipeline_name_raises(self):
        from src.agent.langgraph.graphs import run_pipeline

        with pytest.raises(KeyError):
            run_pipeline(
                input_text="test",
                pipeline_name="non_existent_pipeline",
                steps_config={},
            )


# ═══════════════════════════════════════════════════════════════
# 2. 节点单元测试（不依赖完整图）
# ═══════════════════════════════════════════════════════════════

class TestTranslatorNode:
    """TranslatorNode 独立测试"""

    def test_adapt_input_string(self, fake_skill_manager):
        node = TranslatorNode(
            skill_manager=fake_skill_manager, config={"target_lang": "en"}
        )
        skill = fake_skill_manager.get("translator")()
        adapted = node.adapt_input(SAMPLE_TEXT, skill, config=None)

        assert hasattr(adapted, "text"), "适配后应有 text 属性"
        assert adapted.text == SAMPLE_TEXT

    def test_adapt_input_dict(self, fake_skill_manager):
        node = TranslatorNode(
            skill_manager=fake_skill_manager, config={"target_lang": "en"}
        )
        skill = fake_skill_manager.get("translator")()
        adapted = node.adapt_input(
            {"text": SAMPLE_TEXT, "source_lang": "zh"},
            skill,
            config=None,
        )
        assert adapted.text == SAMPLE_TEXT

    def test_config_passed_to_node(self, fake_skill_manager):
        """
        🔴 关键测试：验证 config 确实被节点接收并存储
        """
        node = TranslatorNode(
            skill_manager=fake_skill_manager,
            config={"target_lang": "en", "source_lang": "zh"},
        )
        assert hasattr(node, "_config"), "节点必须有 _config 属性"
        assert node._config.get("target_lang") == "en"
        assert node._config.get("source_lang") == "zh"


class TestBaseNodeExecution:
    """BaseNode 执行流程测试"""

    def test_successful_execution(self, fake_skill_manager, base_state):
        node = TranslatorNode(skill_manager=fake_skill_manager, config={})
        result = node(base_state, config=None)

        assert isinstance(result, dict), "返回值必须是 dict"
        assert "current_output" in result
        assert "skill_results" in result
        assert result["current_output"] == TRANSLATED_TEXT
        assert len(result["skill_results"]) == 1
        assert result["skill_results"][0].skill_name == "translator"
        assert result.get("error") is None

    def test_execution_records_elapsed_time(self, fake_skill_manager, base_state):
        node = TranslatorNode(skill_manager=fake_skill_manager, config={})
        result = node(base_state, config=None)

        record = result["skill_results"][0]
        assert "elapsed_ms" in record or hasattr(record, "elapsed_ms")
        elapsed = (
            record.get("elapsed_ms") if isinstance(record, dict) else record.elapsed_ms
        )
        assert elapsed > 0, "执行耗时必须 > 0"

    def test_skill_failure_produces_error_state(self, failing_skill_manager, base_state):
        # failing_skill_manager 中 translator 正常，summarizer 失败
        # 这里直接测 translator 失败：用 make_fake_skill(should_fail=True)
        sm = FakeSkillManager()
        sm.register("translator", lambda: make_fake_skill(should_fail=True))

        node = TranslatorNode(skill_manager=sm, config={})
        result = node(base_state, config=None)

        assert result is not None
        assert result.get("error") is not None, "失败时必须设置 error 字段"
        assert result["current_output"] is None

    def test_cancelled_state_prevents_execution(self, fake_skill_manager):
        state = deepcopy(create_initial_state(input_data="test"))
        state["cancelled"] = True

        node = TranslatorNode(skill_manager=fake_skill_manager, config={})
        result = node(state, config=None)

        assert result == {} or result.get("current_output") is None

    def test_empty_input_guarded(self, fake_skill_manager):
        state = deepcopy(create_initial_state(input_data=""))
        state["current_input"] = ""

        node = TranslatorNode(skill_manager=fake_skill_manager, config={})
        result = node(state, config=None)

        assert result.get("error") is not None


# ═══════════════════════════════════════════════════════════════
# 3. 集成测试（使用 fake SkillManager 隔离真实技能）
# ═══════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """完整流水线集成测试"""

    def test_invoke_full_pipeline(self, fake_skill_manager):
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text="你好，世界！",
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": "en"},
                "summarizer_config": {"max_length": "short"},
            },
            session_id="test-001",
            skill_manager=fake_skill_manager,
        )

        assert result is not None
        # ✅ 修复：current_output 是 dict {'summary': 'A summary.'}
        assert isinstance(result.get("current_output"), dict)
        assert result["current_output"].get("summary") == SUMMARY_TEXT
        assert len(result["skill_results"]) == 2
        assert result["skill_results"][0].skill_name == "translator"
        assert result["skill_results"][1].skill_name == "text_summarizer"

    def test_invoke_summarize_then_email_pipeline(self, fake_skill_manager):
        """测试总结后起草邮件流水线"""
        sm = FakeSkillManager()
        sm.register("text_summarizer", lambda: FakeSkill(
            return_value=FakeSkillOutput(summary="Meeting summary.")
        ))
        sm.register("email_drafter", lambda: FakeSkill(
            return_value=FakeSkillOutput(email_draft="Drafted email.")
        ))

        from src.agent.langgraph.graphs import run_pipeline
        result = run_pipeline(
            input_text="Meeting notes...",
            pipeline_name="summarize_then_email",
            steps_config={"summarizer_config": {}, "email_config": {}},
            skill_manager=sm,
        )

        assert result is not None
        assert len(result["skill_results"]) == 2
        # ✅ 修复：current_output 现在是 dict {'email_draft': 'Drafted email.'}
        assert isinstance(result.get("current_output"), dict)


# ═══════════════════════════════════════════════════════════════
# 4. 输入验证测试
# ═══════════════════════════════════════════════════════════════

class TestInputValidation:
    """入口输入验证"""

    def test_empty_string_raises(self):
        from src.agent.langgraph.graphs import run_pipeline, PipelineExecutionError

        with pytest.raises(PipelineExecutionError, match="不能为空"):
            run_pipeline(
                input_text="",
                pipeline_name="translate_then_summarize",
                steps_config={},
            )

    def test_whitespace_only_raises(self):
        from src.agent.langgraph.graphs import run_pipeline, PipelineExecutionError

        with pytest.raises(PipelineExecutionError, match="不能为空"):
            run_pipeline(
                input_text="   \n\t  ",
                pipeline_name="translate_then_summarize",
                steps_config={},
            )

    def test_none_input_raises(self):
        from src.agent.langgraph.graphs import run_pipeline, PipelineExecutionError

        with pytest.raises(PipelineExecutionError):
            run_pipeline(
                input_text=None,
                pipeline_name="translate_then_summarize",
                steps_config={},
            )

    def test_non_string_input_raises(self):
        from src.agent.langgraph.graphs import run_pipeline, PipelineExecutionError

        with pytest.raises(PipelineExecutionError):
            run_pipeline(
                input_text=12345,
                pipeline_name="translate_then_summarize",
                steps_config={},
            )


# ═══════════════════════════════════════════════════════════════
# 5. State 结构测试
# ═══════════════════════════════════════════════════════════════

class TestStateStructure:
    """State 结构和初始化"""

    def test_create_initial_state_defaults(self):
        state = create_initial_state(input_data="测试输入", session_id="test-123")

        assert state["current_input"] == "测试输入"
        assert state["session_id"] == "test-123"
        assert state["current_output"] is None
        assert state["skill_results"] == []
        assert state["error"] is None
        assert state["cancelled"] is False
        assert state["messages"] == []

    @pytest.mark.skip(reason="RAG 字段待 state.py 补充 Annotated 定义后启用")
    def test_rag_fields_exist(self):
        """RAG 专用字段已定义（暂时 skip，等 state.py 补字段）"""
        assert "rewritten_query" in GraphState.__annotations__
        assert "search_results" in GraphState.__annotations__
        assert "reranked_results" in GraphState.__annotations__

    def test_schema_version_is_v2(self):
        state = create_initial_state(input_data="test")
        assert state["schema_version"] == "2.0"

    def test_missing_session_id_auto_generates(self):
        state = create_initial_state(input_data="test")
        assert state["session_id"] is not None
        assert len(state["session_id"]) > 0

    def test_state_is_mutable_dict(self):
        state = create_initial_state(input_data="original")
        state["current_output"] = "modified"
        assert state["current_output"] == "modified"

    def test_token_usage_initial_value(self):
        from src.agent.langgraph.state import TokenUsage

        state = create_initial_state(input_data="test")
        assert "cumulative_token_usage" in state
        assert isinstance(state["cumulative_token_usage"], TokenUsage)
        assert state["cumulative_token_usage"].total_tokens == 0
        assert state["cumulative_token_usage"].prompt_tokens == 0
        assert state["cumulative_token_usage"].completion_tokens == 0


# ═══════════════════════════════════════════════════════════════
# 6. 边界情况测试
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界与异常情况"""

    def test_very_long_input(self, fake_skill_manager):
        from src.agent.langgraph.graphs import run_pipeline

        long_text = "测试。" * 10_000
        result = run_pipeline(
            input_text=long_text,
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": "en"},
                "summarizer_config": {},
            },
            skill_manager=fake_skill_manager,
        )
        assert result is not None

    def test_special_characters(self, fake_skill_manager):
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text="你好 👋 世界 🌍 测试 <script>alert(1)</script>",
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": "en"},
                "summarizer_config": {},
            },
            skill_manager=fake_skill_manager,
        )
        assert result is not None

    @pytest.mark.parametrize("target_lang", ["en", "zh"])
    def test_various_target_languages(self, fake_skill_manager, target_lang):
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text=SAMPLE_TEXT,
            pipeline_name="translate_then_summarize",
            steps_config={
                "translator_config": {"target_lang": target_lang},
                "summarizer_config": {},
            },
            skill_manager=fake_skill_manager,
        )
        assert result is not None

    def test_missing_summarizer_config_uses_defaults(self, fake_skill_manager):
        from src.agent.langgraph.graphs import run_pipeline

        result = run_pipeline(
            input_text=SAMPLE_TEXT,
            pipeline_name="translate_then_summarize",
            steps_config={"translator_config": {"target_lang": "en"}},
            skill_manager=fake_skill_manager,
        )
        assert result is not None
        assert result.get("error") is None


# ═══════════════════════════════════════════════════════════════
# 7. Config 传递链路测试（针对已知 P0 bug）
# ═══════════════════════════════════════════════════════════════

class TestConfigPropagation:
    """
    🔴 针对前几轮审查发现的 P0 bug：
    - BaseNode.__init__ 不接受 config → 配置静默丢弃
    - BaseNode.__call__ 不把 config 传给 adapt_input

    这些测试确保修复后配置能正确传递：
    graphs.py → TranslatorNode.__init__ → adapt_input
    """

    def test_translator_node_stores_config(self):
        node = TranslatorNode(config={"target_lang": "en"})
        assert node._config == {"target_lang": "en"}

    def test_adapt_input_reads_from_stored_config(self, fake_skill_manager):
        node = TranslatorNode(
            skill_manager=fake_skill_manager,
            config={"target_lang": "en", "source_lang": "zh"},
        )
        skill = fake_skill_manager.get("translator")()
        adapted = node.adapt_input(SAMPLE_TEXT, skill, config=None)

        assert adapted.target_lang == "en"
        assert adapted.source_lang == "zh"

    def test_adapt_input_runnable_config_overrides_stored_config(
        self, fake_skill_manager
    ):
        """
        优先级：RunnableConfig > 构造函数 config > 默认值
        """
        node = TranslatorNode(
            skill_manager=fake_skill_manager,
            config={"target_lang": "en"},
        )
        skill = fake_skill_manager.get("translator")()
        adapted = node.adapt_input(
            "Hello",
            skill,
            config={"configurable": {"target_lang": "zh"}},
        )
        assert adapted.target_lang == "zh", "RunnableConfig 应覆盖构造函数配置"

    def test_config_partial_override(self, fake_skill_manager):
        """部分覆盖：RunnableConfig 只覆盖传入的字段"""
        node = TranslatorNode(
            skill_manager=fake_skill_manager,
            config={"target_lang": "en", "source_lang": "zh"},
        )
        skill = fake_skill_manager.get("translator")()
        adapted = node.adapt_input(
            SAMPLE_TEXT,
            skill,
            config={"configurable": {"target_lang": "en"}},
        )
        assert adapted.target_lang == "en"
        assert adapted.source_lang == "zh", "未覆盖的字段应保留构造函数值"


# ═══════════════════════════════════════════════════════════════
# 8. RAGType 枚举测试
# ═══════════════════════════════════════════════════════════════

class TestRAGType:
    """RAGType 枚举行为"""

    def test_valid_types(self):
        assert RAGType("standard") == RAGType.STANDARD
        assert RAGType("graphrag") == RAGType.GRAPHRAG
        assert RAGType("hybrid") == RAGType.HYBRID

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            RAGType("invalid_type")

    def test_string_value(self):
        assert RAGType.STANDARD.value == "standard"
        assert RAGType.GRAPHRAG.value == "graphrag"
        assert RAGType.HYBRID.value == "hybrid"


# ═══════════════════════════════════════════════════════════════
# 9. RAG 图构建与执行测试
# ═══════════════════════════════════════════════════════════════

class TestRAGGraph:
    """RAG 图构建、执行与配置传递"""

    # ── 标准 RAG ──

    def test_standard_rag_success(self, rag_skill_manager):
        from src.agent.langgraph.graphs import run_rag_pipeline

        result = run_rag_pipeline(
            query="什么是 LangGraph？",
            rag_type="standard",
            skill_manager=rag_skill_manager,
        )

        assert result is not None
        assert result.get("error") is None
        assert len(result["skill_results"]) == 4  # rewrite + vector + rerank + answer
        skill_names = [r.skill_name for r in result["skill_results"]]
        assert "query_rewrite" in skill_names
        assert "vector_search" in skill_names
        assert "rerank" in skill_names
        assert "rag_answer" in skill_names

    def test_standard_rag_with_config(self, rag_skill_manager):
        from src.agent.langgraph.graphs import run_rag_pipeline

        result = run_rag_pipeline(
            query="解释 RAG",
            rag_type="standard",
            steps_config={
                "rewrite_config": {"method": "hyde"},
                "vector_config": {"top_k": 5},
                "rerank_config": {"model": "bge-reranker"},
                "answer_config": {"max_tokens": 256},
            },
            skill_manager=rag_skill_manager,
        )

        assert result is not None
        assert result.get("error") is None

    # ── GraphRAG ──

    def test_graphrag_success(self, rag_skill_manager):
        from src.agent.langgraph.graphs import run_rag_pipeline

        result = run_rag_pipeline(
            query="实体关系查询",
            rag_type="graphrag",
            skill_manager=rag_skill_manager,
        )

        assert result is not None
        assert result.get("error") is None
        skill_names = [r.skill_name for r in result["skill_results"]]
        assert "graphrag_searcher" in skill_names
        assert "vector_search" not in skill_names  # GraphRAG 不用向量检索

    # ── Hybrid（并行 fan-in） ──

    def test_hybrid_parallel_fan_in(self, rag_skill_manager):
        """
        Hybrid 模式下 vector_search 和 graphrag_search 并行执行，
        LangGraph 在 rerank 处自动 fan-in 等待两者完成。
        """
        from src.agent.langgraph.graphs import run_rag_pipeline

        result = run_rag_pipeline(
            query="混合检索查询",
            rag_type="hybrid",
            skill_manager=rag_skill_manager,
        )

        assert result is not None
        assert result.get("error") is None
        skill_names = [r.skill_name for r in result["skill_results"]]
        assert "vector_search" in skill_names
        assert "graphrag_searcher" in skill_names
        # 两个检索都完成后才进入 rerank
        assert skill_names.index("rerank") > skill_names.index("vector_search")
        assert skill_names.index("rerank") > skill_names.index("graphrag_searcher")

    # ── 异常 ──

    def test_invalid_rag_type_raises(self, rag_skill_manager):
        from src.agent.langgraph.graphs import run_rag_pipeline

        with pytest.raises(ValueError, match="无效的 rag_type"):
            run_rag_pipeline(
                query="test",
                rag_type="nonexistent",
                skill_manager=rag_skill_manager,
            )

    def test_rag_skill_failure_produces_error(self, rag_skill_manager):
        sm = FakeSkillManager()
        sm.register("query_rewrite", lambda: make_fake_skill(should_fail=True))
        sm.register("vector_search", lambda: make_fake_skill())
        sm.register("rerank", lambda: make_fake_skill())
        sm.register("rag_answer", lambda: make_fake_skill())

        from src.agent.langgraph.graphs import run_rag_pipeline, PipelineExecutionError

        with pytest.raises(PipelineExecutionError, match="RAG 流水线执行失败"):
            run_rag_pipeline(
                query="test",
                rag_type="standard",
                skill_manager=sm,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])