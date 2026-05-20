"""
src/agent/langgraph/checkpointer.py — Checkpoint 持久化封装

提供：
- get_checkpointer(): 获取 SQLite Checkpointer 实例（同步）
- async_get_checkpointer(): 获取 AsyncSqliteSaver 实例（异步）
- reset_checkpointer(): 重置实例（测试用）
- 线程安全、按 (db_path, use_async) 多实例共存
- 支持 :memory: 内存数据库（测试友好）
"""

import logging
import threading
from pathlib import Path
from typing import Union

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger("langgraph.checkpointer")

# ── 全局状态 ──────────────────────────────────────────────

# 缓存 (context_manager, instance) 元组，防止 cm 被 gc 导致连接关闭
_instances: dict[tuple[str, bool], tuple] = {}
_lock = threading.Lock()

# ── 注册自定义 State 类型到 LangGraph msgpack 序列化器 ──
# 修复 P2 警告: "Deserializing unregistered type"
import os

_ALLOWED_MODULES = (
    "src.agent.langgraph.state:SkillExecutionRecord,"
    "src.agent.langgraph.state:StateError,"
    "src.agent.langgraph.state:StepMetadata"
)
_prev = os.environ.get("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "")
if _prev:
    os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = _prev + "," + _ALLOWED_MODULES
else:
    os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = _ALLOWED_MODULES

# ── 同步 API ──────────────────────────────────────────────

def get_checkpointer(
    db_path: str = "data/checkpoints.sqlite",
    use_async: bool = False,
) -> Union[SqliteSaver, AsyncSqliteSaver]:
    """
    获取 SQLite Checkpointer 实例（同步版本）

    按 (db_path, use_async) 缓存实例，避免重复创建数据库连接。
    线程安全。

    Args:
        db_path: SQLite 数据库文件路径。
                 传入 ":memory:" 使用内存数据库（测试友好）。
        use_async: 是否使用异步版本 AsyncSqliteSaver（已废弃，请用 async_get_checkpointer）。

    Returns:
        SqliteSaver 或 AsyncSqliteSaver 实例。

    Example:
        >>> checkpointer = get_checkpointer()
        >>> graph = workflow.compile(checkpointer=checkpointer)
        >>> graph.invoke(state, {"configurable": {"thread_id": "session-1"}})

        # 测试用内存库：
        >>> cp = get_checkpointer(":memory:")
    """
    key = (db_path, use_async)

    with _lock:
        if key in _instances:
            # 返回缓存的实例（第二个元素）
            return _instances[key][1]

        # 确保数据库目录存在（内存库跳过）
        db_file = Path(db_path)
        if db_path != ":memory:" and not db_file.parent.exists():
            db_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing SQLite Checkpointer: %s", db_file)

        try:
            if use_async:
                cm = AsyncSqliteSaver.from_conn_string(str(db_file))
                instance = cm.__enter__()
            else:
                cm = SqliteSaver.from_conn_string(str(db_file))
                instance = cm.__enter__()

            # ✅ 缓存 (context_manager, instance) 元组，防止 cm 被 gc
            _instances[key] = (cm, instance)
            logger.info("Checkpointer initialized successfully")
            return instance

        except Exception:
            logger.exception("Failed to initialize checkpointer")
            raise


# ── 异步 API ──────────────────────────────────────────────

async def async_get_checkpointer(
    db_path: str = "data/checkpoints.sqlite",
) -> AsyncSqliteSaver:
    """
    获取异步 SQLite Checkpointer 实例（异步版本）

    正确调用 AsyncSqliteSaver 的 __aenter__()，避免同步调用的错误。

    Args:
        db_path: SQLite 数据库文件路径。

    Returns:
        AsyncSqliteSaver 实例。

    Example:
        # >>> checkpointer = await async_get_checkpointer()
        # >>> graph = workflow.compile(checkpointer=checkpointer)
        # >>> async for event in graph.astream_events(state, config):
        # ...     print(event)
    """
    key = (db_path, True)

    with _lock:
        if key in _instances:
            return _instances[key][1]

    # 确保数据库目录存在（内存库跳过）
    db_file = Path(db_path)
    if db_path != ":memory:" and not db_file.parent.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Async SQLite Checkpointer: %s", db_file)

    try:
        cm = AsyncSqliteSaver.from_conn_string(str(db_file))
        # ✅ 正确调用异步上下文管理器的 __aenter__()
        instance = await cm.__aenter__()

        # ✅ 缓存 (context_manager, instance) 元组，防止 cm 被 gc
        with _lock:
            _instances[key] = (cm, instance)

        logger.info("Async Checkpointer initialized successfully")
        return instance

    except Exception:
        logger.exception("Failed to initialize async checkpointer")
        raise


def reset_checkpointer(
    db_path: str = "data/checkpoints.sqlite",
    use_async: bool = False,
) -> None:
    """
    重置指定 Checkpointer 实例（主要用于测试隔离）

    Args:
        db_path: 与 get_checkpointer 相同的路径参数。
        use_async: 与 get_checkpointer 相同的异步参数。

    Example:
        >>> reset_checkpointer(":memory:")
    """
    key = (db_path, use_async)
    with _lock:
        removed = _instances.pop(key, None)
        if removed is not None:
            cm, instance = removed
            # ✅ 正确关闭 context manager
            cm.__exit__(None, None, None)
            logger.info("Checkpointer instance reset: %s", db_path)
        else:
            logger.debug("No checkpointer instance to reset for: %s", db_path)
