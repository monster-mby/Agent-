"""指标埋点模块 - 将技能和LLM调用指标存储到SQLite数据库"""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── 配置 ────────────────────────────────────────────
DB_PATH = os.getenv("METRICS_DB_PATH", os.path.join(os.getcwd(), "data", "metrics.db"))
_POOL_LOCK = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_initialized = False


def _get_conn() -> sqlite3.Connection:
    """获取线程安全的连接（惰性初始化 + WAL 模式）"""
    global _conn, _initialized
    with _POOL_LOCK:
        if _conn is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
        if not _initialized:
            _init_tables(_conn)
            _initialized = True
    return _conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS skill_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_name  TEXT NOT NULL,
            trace_id    TEXT,
            status      TEXT NOT NULL,
            elapsed_ms  REAL,
            error_type  TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS llm_metrics (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id           TEXT,
            skill_name         TEXT,
            model              TEXT,
            provider           TEXT,
            prompt_tokens      INTEGER,
            completion_tokens  INTEGER,
            total_tokens       INTEGER,
            elapsed_ms         REAL,
            status             TEXT NOT NULL,
            created_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_trace  ON skill_metrics(trace_id);
        CREATE INDEX IF NOT EXISTS idx_skill_name   ON skill_metrics(skill_name);
        CREATE INDEX IF NOT EXISTS idx_llm_trace    ON llm_metrics(trace_id);
        CREATE INDEX IF NOT EXISTS idx_llm_name     ON llm_metrics(skill_name);
    """)


# ── 写入 API ────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_skill_metric(
    skill_name: str,
    trace_id: str = "",
    status: str = "",
    elapsed_ms: float = 0.0,
    error_type: Optional[str] = None,
) -> bool:
    """记录技能执行指标，返回 True 表示写入成功"""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO skill_metrics (skill_name, trace_id, status, elapsed_ms, error_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (skill_name, trace_id, status, elapsed_ms, error_type, _now()),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        _log("error", f"技能指标写入DB失败: {e}", exc_info=True)
    except Exception as e:
        _log("error", f"技能指标写入异常: {e}", exc_info=True)
    return False


def record_llm_metric(
    trace_id: str = "",
    skill_name: str = "",
    model: str = "",
    provider: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    elapsed_ms: float = 0.0,
    status: str = "",
) -> bool:
    """记录LLM调用指标，返回 True 表示写入成功"""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO llm_metrics "
            "(trace_id, skill_name, model, provider, prompt_tokens, completion_tokens, total_tokens, elapsed_ms, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, skill_name, model, provider, prompt_tokens, completion_tokens, total_tokens, elapsed_ms, status, _now()),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        _log("error", f"LLM指标写入DB失败: {e}", exc_info=True)
    except Exception as e:
        _log("error", f"LLM指标写入异常: {e}", exc_info=True)
    return False


def record_llm_metrics_batch(records: List[Tuple]) -> int:
    """批量写入LLM指标，返回成功条数"""
    try:
        conn = _get_conn()
        conn.executemany(
            "INSERT INTO llm_metrics "
            "(trace_id, skill_name, model, provider, prompt_tokens, completion_tokens, total_tokens, elapsed_ms, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*r, _now()) for r in records],
        )
        conn.commit()
        return len(records)
    except sqlite3.Error as e:
        _log("error", f"LLM批量写入DB失败: {e}", exc_info=True)
    except Exception as e:
        _log("error", f"LLM批量写入异常: {e}", exc_info=True)
    return 0


# ── 查询 API ────────────────────────────────────────
def get_skill_metrics(trace_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """按 trace_id 查询技能指标（不传则返回最近记录）"""
    try:
        conn = _get_conn()
        if trace_id:
            rows = conn.execute(
                "SELECT skill_name, trace_id, status, elapsed_ms, error_type, created_at "
                "FROM skill_metrics WHERE trace_id = ? ORDER BY id DESC LIMIT ?",
                (trace_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT skill_name, trace_id, status, elapsed_ms, error_type, created_at "
                "FROM skill_metrics ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(zip(("skill_name", "trace_id", "status", "elapsed_ms", "error_type", "created_at"), r)) for r in rows]
    except sqlite3.Error as e:
        _log("error", f"查询技能指标失败: {e}")
        return []


def get_llm_metrics(trace_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """按 trace_id 查询LLM指标"""
    try:
        conn = _get_conn()
        if trace_id:
            rows = conn.execute(
                "SELECT trace_id, skill_name, model, provider, prompt_tokens, completion_tokens, "
                "total_tokens, elapsed_ms, status, created_at "
                "FROM llm_metrics WHERE trace_id = ? ORDER BY id DESC LIMIT ?",
                (trace_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT trace_id, skill_name, model, provider, prompt_tokens, completion_tokens, "
                "total_tokens, elapsed_ms, status, created_at "
                "FROM llm_metrics ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("trace_id", "skill_name", "model", "provider", "prompt_tokens",
                "completion_tokens", "total_tokens", "elapsed_ms", "status", "created_at")
        return [dict(zip(keys, r)) for r in rows]
    except sqlite3.Error as e:
        _log("error", f"查询LLM指标失败: {e}")
        return []


# ── 日志 ────────────────────────────────────────────
import logging
_logger = logging.getLogger(__name__)

def _log(level: str, msg: str, exc_info: bool = False) -> None:
    getattr(_logger, level)(msg, exc_info=exc_info)