"""
tests/test_session_manager.py — SessionManager 单元测试 + 桥接验证（优化版）

测试覆盖：
1. Session CRUD 基本功能（创建/读取/更新/归档/删除）
2. 消息历史管理（追加/批量/边界/排序）
3. 与 LangGraph Checkpoint 的桥接
4. 多会话隔离（同用户 / 不同用户）
5. JSON 字段序列化
6. 异常路径（不存在 ID / 空输入 / 关闭后行为）
7. 并发安全性
"""

import os
import tempfile
import uuid
import pytest
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.infrastructure.session_manager import SessionManager, SessionConfig, Message
from src.agent.langgraph.checkpointer import get_checkpointer, reset_checkpointer

# ═══════════════════════════════════════════════════════
# 模块级常量
# ═══════════════════════════════════════════════════════

TEST_USER_A = "user-1"
TEST_USER_B = "user-2"
TEST_KB_IDS = ["kb-1", "kb-2"]
TEST_METADATA = {"theme": "dark", "language": "zh-CN"}

# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def _create_test_session(sm: SessionManager, user_id: str = TEST_USER_A, **kwargs) -> dict:
    """创建单个测试会话，返回 session 对象"""
    return sm.create_session(user_id=user_id, name=kwargs.pop("name", "测试会话"), **kwargs)


def _create_test_sessions(sm: SessionManager, user_id: str, count: int, **kwargs) -> list:
    """批量创建测试会话"""
    return [
        _create_test_session(sm, user_id=user_id, name=f"会话-{i}", **kwargs)
        for i in range(count)
    ]


def _add_test_messages(sm: SessionManager, session_id: str, count: int, role: str = "user") -> list:
    """批量添加测试消息，返回消息列表"""
    messages = [Message(role=role, content=f"消息-{i}") for i in range(count)]
    sm.add_messages_batch(session_id, messages)
    return messages


def _assert_valid_session(obj, expected_user_id: str = TEST_USER_A):
    """验证 Session 对象基本字段合法性"""
    assert obj.session_id is not None
    uuid.UUID(obj.session_id)  # 若非法 UUID 则抛异常
    assert obj.user_id == expected_user_id
    assert obj.langgraph_thread_id is not None
    uuid.UUID(obj.langgraph_thread_id)
    # 时间字段有效性
    assert obj.created_at is not None
    assert isinstance(obj.created_at, datetime)  # ✅ 修复：直接检查类型，不调用 fromisoformat
    assert obj.created_at.tzinfo is not None  # 确保带时区


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_checkpoints():
    """每个测试前重置 checkpointer，避免状态污染"""
    reset_checkpointer(":memory:")
    yield
    reset_checkpointer(":memory:")


@pytest.fixture
def db_path():
    """每个测试使用独立的临时文件，避免 :memory: 并发限制"""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_session_")
    os.close(fd)
    yield path
    # 测试后清理——Windows 下可能有 checkpointer 等持有文件句柄
    if os.path.exists(path):
        import time
        for _ in range(10):          # ✅ 最多等 1 秒（10次 × 0.1秒）
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(0.1)


@pytest.fixture
def sm(db_path):
    """基于文件数据库的 SessionManager（function 级隔离）"""
    # ✅ 关键：获取 checkpointer 引用，以便后续显式关闭
    checkpointer = get_checkpointer(db_path)

    manager = SessionManager(
        config=SessionConfig(db_path=db_path),
        checkpointer=checkpointer
    )
    yield manager

    # ✅ 修复顺序：先关 SessionManager，再关 checkpointer
    manager.close()

    # ✅ 关键：显式释放 checkpointer 的连接
    if hasattr(checkpointer, "close"):
        checkpointer.close()
    elif hasattr(checkpointer, "reset"):
        checkpointer.reset()

    # ✅ 帮助 GC 尽早回收
    del manager
    del checkpointer


@pytest.fixture
def session(sm):
    """创建一个已存在的测试会话"""
    return _create_test_session(sm)


@pytest.fixture
def session_with_messages(sm):
    """创建一个带 5 条消息的测试会话"""
    s = _create_test_session(sm)
    _add_test_messages(sm, s.session_id, count=5)
    return sm.get_session(s.session_id)  # 重新读取，含最新 message_history


# ═══════════════════════════════════════════════════════
# 测试 1：Session CRUD 基本功能
# ═══════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionCRUD:

    # ── 创建 ──────────────────────────────────────

    def test_create_session_basic(self, sm):
        session = _create_test_session(sm, name="测试会话", knowledge_base_ids=TEST_KB_IDS)
        _assert_valid_session(session)
        assert session.name == "测试会话"
        assert session.knowledge_base_ids == TEST_KB_IDS

    def test_create_session_defaults(self, sm):
        """默认 name、空 knowledge_base_ids"""
        session = sm.create_session(user_id=TEST_USER_A)
        _assert_valid_session(session)
        assert isinstance(session.name, str)
        assert session.knowledge_base_ids == []

    # ── 读取 ──────────────────────────────────────

    def test_get_session_by_id(self, sm, session):
        retrieved = sm.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_session_by_thread_id(self, sm, session):
        retrieved = sm.get_session_by_thread_id(session.langgraph_thread_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    @pytest.mark.parametrize("lookup_type,identifier", [
        ("id", "nonexistent-id"),
        ("thread_id", "00000000-0000-0000-0000-000000000000"),
        ("thread_id", ""),
    ])
    def test_get_session_not_found(self, sm, lookup_type, identifier):
        """查找不存在的会话应返回 None"""
        if lookup_type == "id":
            assert sm.get_session(identifier) is None
        else:
            assert sm.get_session_by_thread_id(identifier) is None

    # ── 更新 ──────────────────────────────────────

    def test_update_session_name(self, sm, session):
        updated = sm.update_session(session.session_id, name="新名称")
        assert updated is not None
        assert updated.name == "新名称"

    def test_update_session_knowledge_base_ids(self, sm, session):
        updated = sm.update_session(session.session_id, knowledge_base_ids=["kb-x"])
        assert updated.knowledge_base_ids == ["kb-x"]

    def test_update_session_metadata(self, sm, session):
        updated = sm.update_session(session.session_id, metadata=TEST_METADATA)
        assert updated.metadata == TEST_METADATA

    def test_update_nonexistent_session(self, sm):
        """更新不存在的会话应返回 None"""
        assert sm.update_session("nonexistent", name="x") is None

    # ── 列表过滤 ──────────────────────────────────

    @pytest.mark.parametrize("status_filter,expected_count", [
        ("active", 2),
        ("archived", 1),
        ("deleted", 1),
        ("all", 4),
    ])
    def test_list_sessions_by_status(self, sm, status_filter, expected_count):
        """参数化验证状态过滤"""
        sessions = _create_test_sessions(sm, TEST_USER_A, 2)       # 2 active
        archived = _create_test_session(sm, name="归档会话")
        deleted = _create_test_session(sm, name="已删会话")
        sm.archive_session(archived.session_id)
        sm.delete_session(deleted.session_id)

        result = sm.list_sessions(user_id=TEST_USER_A, status=status_filter)
        assert len(result) == expected_count

    def test_list_sessions_user_isolation(self, sm):
        """只返回指定用户的会话"""
        _create_test_sessions(sm, TEST_USER_A, 2)
        _create_test_sessions(sm, TEST_USER_B, 1)

        assert len(sm.list_sessions(TEST_USER_A)) == 2
        assert len(sm.list_sessions(TEST_USER_B)) == 1

    # ── 归档 / 删除 ──────────────────────────────

    def test_archive_session(self, sm, session):
        archived = sm.archive_session(session.session_id)
        assert archived.status == "archived"
        # active 列表不包含
        assert len(sm.list_sessions(TEST_USER_A, status="active")) == 0

    def test_archive_nonexistent_session(self, sm):
        assert sm.archive_session("nonexistent") is None

    def test_delete_session_soft(self, sm, session):
        assert sm.delete_session(session.session_id) is True
        deleted = sm.get_session(session.session_id)
        assert deleted is not None
        assert deleted.status == "deleted"

    def test_delete_nonexistent_session(self, sm):
        assert sm.delete_session("nonexistent") is False

    # ── 时间字段更新 ──────────────────────────────

    def test_created_at_populated(self, sm, session):
        assert session.created_at is not None
        assert isinstance(session.created_at, datetime)  # ✅ 修复：直接检查类型
        assert session.created_at.tzinfo is not None  # 带时区

    def test_updated_at_after_modification(self, sm, session):
        original = sm.get_session(session.session_id)
        sm.add_message(session.session_id, Message(role="user", content="触发更新"))
        after = sm.get_session(session.session_id)
        # updated_at 应晚于或等于 created_at
        assert after.updated_at >= original.created_at  # ✅ 修复：直接比较 datetime 对象


# ═══════════════════════════════════════════════════════
# 测试 2：消息历史管理
# ═══════════════════════════════════════════════════════

@pytest.mark.unit
class TestMessageHistory:

    def test_add_single_message(self, sm, session):
        updated = sm.add_message(session.session_id, Message(role="user", content="你好"))
        assert len(updated.message_history) == 1
        assert updated.message_history[0].content == "你好"

    def test_add_messages_batch(self, sm, session):
        msgs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ]
        updated = sm.add_messages_batch(session.session_id, msgs)
        assert len(updated.message_history) == 3

    def test_add_messages_batch_empty(self, sm, session):
        """空列表不崩溃，历史不变"""
        updated = sm.add_messages_batch(session.session_id, [])
        assert updated is not None
        assert len(updated.message_history) == 0

    @pytest.mark.parametrize("limit,expected_count", [
        (3, 3),
        (10, 5),   # 只有 5 条，不越界
        (0, 0),
    ])
    def test_get_history_with_limit(self, sm, session_with_messages, limit, expected_count):
        history = sm.get_message_history(session_with_messages.session_id, limit=limit)
        assert len(history) == expected_count

    def test_get_history_default_no_limit(self, sm, session_with_messages):
        """不传 limit 返回全部"""
        assert len(sm.get_message_history(session_with_messages.session_id)) == 5

    def test_get_history_nonexistent_session(self, sm):
        assert sm.get_message_history("nonexistent") is None

    def test_add_message_to_nonexistent_session(self, sm):
        assert sm.add_message("nonexistent", Message(role="user", content="x")) is None

    # ── 消息内容边界 ──────────────────────────────

    def test_message_empty_content(self, sm, session):
        updated = sm.add_message(session.session_id, Message(role="user", content=""))
        assert updated.message_history[-1].content == ""

    def test_message_unicode_emoji(self, sm, session):
        content = "你好 🌍🎉 — Unicode 测试"
        updated = sm.add_message(session.session_id, Message(role="user", content=content))
        assert updated.message_history[-1].content == content

    def test_message_long_content(self, sm, session):
        """100KB 内容不截断不崩溃"""
        long_content = "A" * 102400
        updated = sm.add_message(session.session_id, Message(role="user", content=long_content))
        assert len(updated.message_history[-1].content) == 102400

    # ── 消息排序 ──────────────────────────────────

    def test_message_ordering(self, sm, session):
        """消息按创建时间升序排列"""
        _add_test_messages(sm, session.session_id, count=3)
        history = sm.get_message_history(session.session_id)
        assert len(history) == 3
        # 验证排序：created_at 非递减
        for i in range(1, len(history)):
            assert history[i].created_at >= history[i - 1].created_at  # ✅ 修复：直接比较


# ═══════════════════════════════════════════════════════
# 测试 3：与 LangGraph Checkpoint 的桥接
# ═══════════════════════════════════════════════════════

@pytest.mark.integration
class TestCheckpointBridge:

    def test_sessions_have_unique_thread_ids(self, sm):
        s1 = _create_test_session(sm)
        s2 = _create_test_session(sm)
        assert s1.langgraph_thread_id != s2.langgraph_thread_id

    def test_thread_id_persists(self, sm):
        session = _create_test_session(sm)
        retrieved = sm.get_session(session.session_id)
        assert retrieved.langgraph_thread_id == session.langgraph_thread_id

    def test_bridge_config_format(self, sm, session):
        config = {"configurable": {"thread_id": session.langgraph_thread_id}}
        assert config["configurable"]["thread_id"] == session.langgraph_thread_id
        uuid.UUID(config["configurable"]["thread_id"])  # 格式有效


# ═══════════════════════════════════════════════════════
# 测试 4：多会话隔离
# ═══════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionIsolation:

    def test_concurrent_sessions_message_isolation(self, sm):
        """两个活跃会话的消息历史互不串扰"""
        sa = _create_test_session(sm, name="会话A")
        sb = _create_test_session(sm, name="会话B")

        sm.add_message(sa.session_id, Message(role="user", content="你好A"))
        sm.add_message(sb.session_id, Message(role="user", content="你好B"))

        history_a = sm.get_message_history(sa.session_id)
        history_b = sm.get_message_history(sb.session_id)

        # A 不含 B 的内容
        assert not any("你好B" in m.content for m in history_a)
        # B 不含 A 的内容
        assert not any("你好A" in m.content for m in history_b)

    def test_cross_user_isolation(self, sm):
        """不同用户的会话完全不可见"""
        _create_test_sessions(sm, TEST_USER_A, 2)
        _create_test_sessions(sm, TEST_USER_B, 1)

        assert len(sm.list_sessions(TEST_USER_A)) == 2
        assert len(sm.list_sessions(TEST_USER_B)) == 1
        # 用户的会话 ID 不重叠
        ids_a = {s.session_id for s in sm.list_sessions(TEST_USER_A)}
        ids_b = {s.session_id for s in sm.list_sessions(TEST_USER_B)}
        assert ids_a.isdisjoint(ids_b)


# ═══════════════════════════════════════════════════════
# 测试 5：JSON 字段序列化
# ═══════════════════════════════════════════════════════

@pytest.mark.unit
class TestJSONSerialization:

    def test_knowledge_base_ids_roundtrip(self, sm):
        session = _create_test_session(sm, knowledge_base_ids=["kb-a", "kb-b", "kb-c"])
        retrieved = sm.get_session(session.session_id)
        assert retrieved.knowledge_base_ids == ["kb-a", "kb-b", "kb-c"]

    def test_metadata_roundtrip(self, sm):
        session = _create_test_session(sm, metadata={"nested": {"key": "value"}, "list": [1, 2, 3]})
        retrieved = sm.get_session(session.session_id)
        assert retrieved.metadata == {"nested": {"key": "value"}, "list": [1, 2, 3]}

    def test_empty_knowledge_base_ids(self, sm):
        session = _create_test_session(sm, knowledge_base_ids=[])
        assert sm.get_session(session.session_id).knowledge_base_ids == []


# ═══════════════════════════════════════════════════════
# 测试 6：异常路径 / 边界条件
# ═══════════════════════════════════════════════════════

@pytest.mark.unit
class TestEdgeCases:

    def test_create_session_empty_user_id(self, sm):
        """空 user_id 应拒绝或创建失败"""
        with pytest.raises(ValueError):
            sm.create_session(user_id="")

    def test_close_then_operation(self, sm):
        """关闭后再操作应抛 RuntimeError"""
        sm.close()
        with pytest.raises(RuntimeError):
            sm.create_session(user_id=TEST_USER_A)
        with pytest.raises(RuntimeError):
            sm.get_session("any-id")
        with pytest.raises(RuntimeError):
            sm.list_sessions(TEST_USER_A)

    def test_sql_injection_in_content(self, sm, session):
        """消息内容含 SQL 注入字符不崩溃"""
        malicious = "'; DROP TABLE sessions; --"
        updated = sm.add_message(session.session_id, Message(role="user", content=malicious))
        assert updated.message_history[-1].content == malicious
        # 验证表还在（能正常查询）
        assert sm.get_session(session.session_id) is not None


# ═══════════════════════════════════════════════════════
# 测试 7：并发安全性
# ═══════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.unit
class TestConcurrency:

    def test_concurrent_session_creation(self, sm):
        """10 线程并发创建会话，数量和隔离性正确"""
        THREADS = 10
        sessions_per_thread = []

        def _create_and_message(thread_idx: int):
            s = sm.create_session(user_id=TEST_USER_A, name=f"并发会话-{thread_idx}")
            _add_test_messages(sm, s.session_id, count=5)
            sessions_per_thread.append(s.session_id)
            return s.session_id

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = [executor.submit(_create_and_message, i) for i in range(THREADS)]
            results = [f.result() for f in as_completed(futures)]

        # 无重复 session_id
        assert len(results) == len(set(results)) == THREADS
        # 每个会话都有 5 条消息
        for sid in results:
            assert len(sm.get_message_history(sid)) == 5
        # list 能查到全部
        assert len(sm.list_sessions(TEST_USER_A)) == THREADS


if __name__ == "__main__":    pytest.main([__file__, "-v", "--tb=short"])