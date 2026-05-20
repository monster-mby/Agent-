"""基础设施模块 - 提供日志、指标等基础服务"""

from .logger import (
    logger,
    set_trace_id,
    get_trace_id,
    set_skill_name,
    clear_context,
    log_info,
    log_warning,
    log_error,
    log_debug,
    timed,
)

from .metrics import (
    record_skill_metric,
    record_llm_metric,
    record_llm_metrics_batch,
    get_skill_metrics,
    get_llm_metrics,
)

__all__ = [
    # Logger exports
    "logger",
    "set_trace_id",
    "get_trace_id",
    "set_skill_name",
    "clear_context",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "timed",

    # Metrics exports
    "record_skill_metric",
    "record_llm_metric",
    "record_llm_metrics_batch",
    "get_skill_metrics",
    "get_llm_metrics",
]
