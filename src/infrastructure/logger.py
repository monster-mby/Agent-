"""结构化日志模块 - 输出单行JSON格式的日志到stdout"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── 上下文变量 ───────────────────────────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
skill_name_var: ContextVar[Optional[str]] = ContextVar("skill_name", default=None)

# ── JSON 格式化器 ────────────────────────────────────
class _SafeEncoder(json.JSONEncoder):
    """处理 datetime/UUID/bytes 等无法直接序列化的类型"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.hex()
        return str(obj)  # 最终兜底


class JsonFormatter(logging.Formatter):
    """将 LogRecord 格式化为单行 JSON"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", ""),
            "message": record.getMessage(),
            "context": {
                "trace_id": trace_id_var.get(),
                "skill_name": skill_name_var.get(),
            },
            "data": getattr(record, "data", {}),
            "elapsed_ms": getattr(record, "elapsed_ms", 0),
        }
        return json.dumps(log_data, cls=_SafeEncoder, ensure_ascii=False)


class DynamicStreamHandler(logging.StreamHandler):
    """动态获取当前 sys.stdout 的 StreamHandler（支持 pytest capsys 捕获）"""
    def __init__(self):
        super().__init__(stream=None)  # 不设置初始 stream

    @property
    def stream(self):
        """每次访问时动态获取当前的 sys.stdout"""
        return sys.stdout

    @stream.setter
    def stream(self, value):
        """忽略 setter（保持动态获取）"""
        pass


# ── 日志器配置 ───────────────────────────────────────
_logger = logging.getLogger("enterprise_learning_agent")
_logger.propagate = False
_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# 使用动态 StreamHandler，支持 pytest capsys 捕获
_handler = DynamicStreamHandler()
_handler.setFormatter(JsonFormatter())
_logger.addHandler(_handler)


class _ContextAdapter(logging.LoggerAdapter):
    """自动将 trace_id / skill_name 注入每一条日志"""
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("trace_id", trace_id_var.get())
        extra.setdefault("skill_name", skill_name_var.get())
        return msg, kwargs


logger = _ContextAdapter(_logger, {})

# ── 公开 API ─────────────────────────────────────────
def set_trace_id(trace_id: str = None) -> str:
    """设置 trace_id，不传则自动生成 UUID"""
    tid = trace_id or uuid.uuid4().hex[:16]
    trace_id_var.set(tid)
    return tid

def get_trace_id() -> str:
    return trace_id_var.get()

def set_skill_name(name: str) -> None:
    skill_name_var.set(name)

def clear_context() -> None:
    """清除上下文（请求结束时调用）"""
    trace_id_var.set("")
    skill_name_var.set(None)


def log_info(event: str, message: str = "", data: Dict[str, Any] = None, elapsed_ms: float = 0) -> None:
    _log(logging.INFO, event, message, data, elapsed_ms)

def log_warning(event: str, message: str = "", data: Dict[str, Any] = None, elapsed_ms: float = 0) -> None:
    _log(logging.WARNING, event, message, data, elapsed_ms)

def log_error(event: str, message: str = "", data: Dict[str, Any] = None, elapsed_ms: float = 0) -> None:
    _log(logging.ERROR, event, message, data, elapsed_ms)

def log_debug(event: str, message: str = "", data: Dict[str, Any] = None, elapsed_ms: float = 0) -> None:
    _log(logging.DEBUG, event, message, data, elapsed_ms)

def _log(level: int, event: str, message: str, data: Dict[str, Any], elapsed_ms: float) -> None:
    logger.log(level, message or event, extra={
        "event": event,
        "data": data or {},
        "elapsed_ms": elapsed_ms,
    })


@contextmanager
def timed(event: str):
    """计时上下文管理器，退出时自动记录耗时"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        log_info(event, f"{event} completed", elapsed_ms=elapsed)
