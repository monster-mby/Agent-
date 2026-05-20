"""
tests/test_langgraph_checkpoint.py — Checkpoint 持久化与断点续跑测试

测试维度：
1. Checkpointer 初始化与配置
2. 同一 thread_id 的状态恢复
3. 断点续跑（中断后继续执行）
4. 多会话隔离验证
5. SQLite 数据持久化验证
6. Checkpoint → compile 集成链路
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.langgraph.checkpointer import get_checkpointer, reset_checkpointer
from src.agent.langgraph.graphs import run_pipeline
from src.agent.langgraph.state import GraphState, create_initial_state
from src.agent.langgraph.node_impls import build_skill_node
from langgraph.graph import StateGraph, END

# ═══════════════════════════════════════════════════════════════
# 复用基础测试的 Fake 对象和 fixtures
# ═══════════════════════════════════════════════════════════════

from tests.test_langgraph_basic import (
    FakeSkillManager,
    FakeSkill,
    FakeSkillOutput,
    fake_skill_manager,        # session 级 fixture，已注册 translator + summarizer
    SAMPLE_TEXT,
    TRANSLATED_TEXT,
    SUMMARY_TEXT,
)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════
# 🔧 Checkpoint 测试专用：current_output 的实际结构是 dict
EXPECTED_TRANSLATED_OUTPUT = {"translated_text": TRANSLATED_TEXT}
def _get_thread_ids(db_path: str) -> list[str]:
    """读取 SQLite checkpoint 数据库中所有 thread_id"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cursor.fetchall()]


def _count_checkpoints(db_path: str) -> int:
    """读取 SQLite checkpoint 数据库中的 checkpoint 行数"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        return cursor.fetchone()[0]


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_db_path(tmp_path):
    """临时数据库文件路径（每个测试独立）"""
    return str(tmp_path / "test_checkpoints.sqlite")


@pytest.fixture
def compiled_test_graph(temp_db_path, fake_skill_manager):
    """
    构建一个最小化的编译图：translate → END，带 checkpoint。
    所有需要 StateGraph 的测试统一复用此 fixture。
    """
    sm = fake_skill_manager
    graph = StateGraph(GraphState)

    NodeClass = build_skill_node("translator", sm)
    graph.add_node("translate", NodeClass(config={}))
    graph.set_entry_point("translate")
    graph.add_edge("translate", END)

    checkpointer = get_checkpointer(temp_db_path)
    return graph.compile(checkpointer=checkpointer)


@pytest.fixture(autouse=True)
def cleanup_checkpointer(temp_db_path):
    """
    每个测试后显式重置指定路径的 checkpointer 缓存，
    避免单例模式导致的跨测试状态污染。
    """
    yield
    reset_checkpointer(temp_db_path)


# ═══════════════════════════════════════════════════════════════
# 1. Checkpointer 初始化测试
# ═══════════════════════════════════════════════════════════════

class TestCheckpointerInitialization:
    """Checkpointer 初始化与配置"""

    def test_get_checkpointer_returns_instance(self, temp_db_path):
        cp = get_checkpointer(temp_db_path)
        assert cp is not None

    def test_get_checkpointer_caches_instance(self, temp_db_path):
        cp1 = get_checkpointer(temp_db_path)
        cp2 = get_checkpointer(temp_db_path)
        assert cp1 is cp2, "应返回同一实例"

    def test_get_checkpointer_different_paths(self, tmp_path):
        cp1 = get_checkpointer(str(tmp_path / "db1.sqlite"))
        cp2 = get_checkpointer(str(tmp_path / "db2.sqlite"))
        assert cp1 is not cp2

    def test_memory_database_works(self):
        cp = get_checkpointer(":memory:")
        assert cp is not None

    def test_sqlite_file_created(self, temp_db_path):
        get_checkpointer(temp_db_path)
        assert Path(temp_db_path).exists(), "数据库文件应被创建"

    def test_reset_checkpointer_clears_cache(self, temp_db_path):
        """reset 后再次 get 应返回新实例"""
        cp1 = get_checkpointer(temp_db_path)
        reset_checkpointer(temp_db_path)
        cp2 = get_checkpointer(temp_db_path)
        assert cp1 is not cp2, "reset 后应返回全新实例"


# ═══════════════════════════════════════════════════════════════
# 2. 状态恢复测试（同一 thread_id）
# ═══════════════════════════════════════════════════════════════

class TestStateRestoration:
    """同一 thread_id 第二次调用恢复状态"""

    def test_same_thread_id_restores_state(self, compiled_test_graph):
        """
        验证：同一 thread_id 第二次 invoke 时，
        LangGraph 从 checkpoint 恢复上次的最终状态（追加而非覆盖）。

        关键行为：
        - skill_results 使用 operator.add reducer → 第二次结果追加到第一次后面
        - current_output 从 checkpoint 恢复为上次的最终输出
        - current_input 被新 state 覆盖（非追加字段）
        """
        config = {"configurable": {"thread_id": "shared-thread"}}

        # 第一次调用
        state1 = create_initial_state("第一次输入", session_id="shared-thread")
        result1 = compiled_test_graph.invoke(state1, config=config)

        assert len(result1["skill_results"]) == 1
        # ✅ 修复：current_output 是 dict，不是纯字符串
        assert result1["current_output"] == {"translated_text": TRANSLATED_TEXT}

        # 第二次调用（相同 thread_id，新 state）
        state2 = create_initial_state("第二次输入", session_id="shared-thread")
        result2 = compiled_test_graph.invoke(state2, config=config)

        # ✅ 精确断言：skill_results 被追加（1 → 2）
        assert len(result2["skill_results"]) == 2, (
            "第二次 invoke 应追加 skill_results（operator.add reducer）"
        )
        assert result2["skill_results"][0].skill_name == "translator"
        assert result2["skill_results"][1].skill_name == "translator"

        # ✅ current_output 来自第二次执行（translator 用"第二次输入"翻译）
        assert result2["current_output"] == {"translated_text": TRANSLATED_TEXT}

        # ✅ current_input 被第二次 state 覆盖
        assert result2["current_input"] == "第二次输入"


# ═══════════════════════════════════════════════════════════════
# 3. 多会话隔离测试
# ═══════════════════════════════════════════════════════════════

class TestSessionIsolation:
    """不同 thread_id 之间状态隔离"""

    def test_different_thread_ids_isolated(self, compiled_test_graph):
        """不同 thread_id 的会话互不干扰"""
        config_a = {"configurable": {"thread_id": "session-A"}}
        config_b = {"configurable": {"thread_id": "session-B"}}

        state_a = create_initial_state("你好", session_id="session-A")
        result_a = compiled_test_graph.invoke(state_a, config=config_a)

        state_b = create_initial_state("世界", session_id="session-B")
        result_b = compiled_test_graph.invoke(state_b, config=config_b)

        # 每个会话各自只有 1 次执行
        assert len(result_a["skill_results"]) == 1
        assert len(result_b["skill_results"]) == 1
        assert result_a["skill_results"][0].skill_name == "translator"
        assert result_b["skill_results"][0].skill_name == "translator"


# ═══════════════════════════════════════════════════════════════
# 4. 断点续跑测试（先留桩，待 interrupt API 成熟后实现）
# ═══════════════════════════════════════════════════════════════

class TestBreakpointResume:
    """断点续跑：节点中断后从 checkpoint 恢复继续执行"""

    @pytest.mark.skip(
        reason="需要 LangGraph interrupt API / NodeInterrupt 支持，待基础设施就绪后实现"
    )
    def test_resume_after_interrupt(self, compiled_test_graph):
        """
        模拟场景：
        1. 第一次 invoke 在两个节点之间的中断点停止
        2. 第二次 invoke（相同 thread_id）从中断点恢复，继续执行剩余节点
        3. 验证最终状态包含全量节点输出
        """
        pass


# ═══════════════════════════════════════════════════════════════
# 5. SQLite 数据持久化验证
# ═══════════════════════════════════════════════════════════════

class TestSQLitePersistence:
    """SQLite 文件数据可读性验证"""

    def test_checkpoint_data_written_to_sqlite(
        self, temp_db_path, compiled_test_graph
    ):
        """验证 checkpoint 数据实际写入 SQLite 文件"""
        state = create_initial_state("测试", session_id="persist-test")
        config = {"configurable": {"thread_id": "persist-test"}}
        compiled_test_graph.invoke(state, config=config)

        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            table_exists = cursor.fetchone()
            assert table_exists is not None, "checkpoints 表应存在"

        count = _count_checkpoints(temp_db_path)
        assert count > 0, "checkpoints 表应有数据"

    def test_multiple_sessions_in_one_db(self, temp_db_path, compiled_test_graph):
        """验证同一个数据库文件可以服务多个 thread_id"""
        for i in range(3):
            state = create_initial_state(f"测试{i}", session_id=f"session-{i}")
            config = {"configurable": {"thread_id": f"session-{i}"}}
            compiled_test_graph.invoke(state, config=config)

        thread_ids = _get_thread_ids(temp_db_path)
        assert len(thread_ids) >= 3, "应包含至少 3 个不同的 thread_id"
        assert "session-0" in thread_ids
        assert "session-1" in thread_ids
        assert "session-2" in thread_ids


# ═══════════════════════════════════════════════════════════════
# 6. 集成测试（Checkpointer 正确传入 compile）
# ═══════════════════════════════════════════════════════════════

class TestPipelineWithCheckpoint:
    """run_pipeline 正确集成 Checkpointer"""

    def test_run_pipeline_passes_checkpointer_to_compile(self, temp_db_path, fake_skill_manager):
        """
        验证 run_pipeline 内部将 checkpointer 传给了 graph.compile()，
        而非仅仅调用了 get_checkpointer。
        """
        with patch(
            "src.agent.langgraph.graphs.get_checkpointer"
        ) as mock_get_cp, patch(
            "src.agent.langgraph.graphs.StateGraph.compile"
        ) as mock_compile:
            mock_cp = MagicMock()
            mock_get_cp.return_value = mock_cp

            run_pipeline(
                input_text=SAMPLE_TEXT,
                pipeline_name="translate_then_summarize",
                steps_config={
                    "translator_config": {},
                    "summarizer_config": {},
                },
                session_id="integration-test",
                skill_manager=fake_skill_manager,
            )

            # 验证 checkpointer 被传入 compile()
            mock_compile.assert_called_once()
            _, kwargs = mock_compile.call_args
            assert "checkpointer" in kwargs, (
                "graph.compile() 必须接收 checkpointer 参数"
            )
            assert kwargs["checkpointer"] is mock_cp, (
                "传入 compile 的 checkpointer 必须来自 get_checkpointer"
            )

    def test_run_pipeline_writes_to_sqlite(self, temp_db_path, fake_skill_manager):
        """端到端：run_pipeline 执行后 SQLite 中有 checkpoint 数据"""
        from src.agent.langgraph.graphs import get_checkpointer

        # 预初始化 checkpointer（run_pipeline 内部也会调，这里确保路径一致）
        get_checkpointer(temp_db_path)

        with patch(
            "src.agent.langgraph.graphs.get_checkpointer",
            wraps=lambda: get_checkpointer(temp_db_path),
        ):
            result = run_pipeline(
                input_text=SAMPLE_TEXT,
                pipeline_name="translate_then_summarize",
                steps_config={
                    "translator_config": {},
                    "summarizer_config": {},
                },
                session_id="e2e-test",
                skill_manager=fake_skill_manager,
            )

        assert result is not None
        assert result.get("error") is None

        count = _count_checkpoints(temp_db_path)
        assert count > 0, (
            "run_pipeline 执行后应写入 checkpoint 数据到 SQLite"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])