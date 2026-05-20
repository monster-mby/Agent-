"""测试结构化日志和指标埋点功能"""
import json
import os
import sqlite3
from unittest.mock import patch

import pytest

from src.infrastructure.logger import (
    set_trace_id, get_trace_id, set_skill_name,
    log_info, log_error, clear_context,
)
from src.infrastructure import metrics as metrics_module


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_context():
    """每个测试前后自动清理上下文"""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def temp_metrics_db(tmp_path, monkeypatch):
    """为每个测试创建独立的临时数据库，测试后自动清理"""
    db_path = tmp_path / "metrics.db"

    # 重置 metrics 模块的全局状态
    monkeypatch.setattr(metrics_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(metrics_module, "_conn", None)
    monkeypatch.setattr(metrics_module, "_initialized", False)

    yield str(db_path)

    # 清理连接
    if metrics_module._conn is not None:
        metrics_module._conn.close()


@pytest.fixture
def capture_log(capsys):
    """捕获并解析 JSON 日志输出"""
    class LogCapture:
        def read(self):
            captured = capsys.readouterr()
            lines = [l.strip() for l in captured.out.strip().split("\n") if l.strip()]
            return [json.loads(line) for line in lines]
    return LogCapture()


# ═══════════════════════════════════════════════════════
# 结构化日志测试
# ═══════════════════════════════════════════════════════

class TestStructuredLogging:

    def test_trace_id_generation(self):
        trace_id = set_trace_id()
        assert trace_id, "trace_id 不应为空"
        assert get_trace_id() == trace_id

        custom_id = "test-trace-123"
        set_trace_id(custom_id)
        assert get_trace_id() == custom_id

    @pytest.mark.parametrize("level_fn,event,level", [
        (log_info, "skill.start", "INFO"),
        (log_error, "skill.error", "ERROR"),
    ])
    def test_structured_log_output(self, level_fn, event, level, capture_log):
        set_trace_id("test-trace-456")
        set_skill_name("test_skill")

        level_fn(event, "Test message", {"key": "value"}, 50.5)

        logs = capture_log.read()
        assert len(logs) == 1
        log = logs[0]

        assert log["level"] == level
        assert log["event"] == event
        assert log["message"] == "Test message"
        assert log["context"]["trace_id"] == "test-trace-456"
        assert log["context"]["skill_name"] == "test_skill"
        assert log["data"] == {"key": "value"}
        assert log["elapsed_ms"] == pytest.approx(50.5)

    def test_log_missing_trace_id_defaults(self, capture_log):
        """trace_id 未设置时日志仍可正常输出"""
        log_info("test.event", "No trace set")
        logs = capture_log.read()
        assert len(logs) == 1


# ═══════════════════════════════════════════════════════
# 指标埋点测试
# ═══════════════════════════════════════════════════════

class TestMetricsRecording:

    @pytest.mark.parametrize("status,error_type", [
        ("success", None),
        ("error", "RuntimeError"),
    ])
    def test_skill_metric_record_and_query(self, temp_metrics_db, status, error_type):
        assert metrics_module.record_skill_metric(
            skill_name="test_skill",
            trace_id="trace-1",
            status=status,
            elapsed_ms=150.5,
            error_type=error_type,
        )

        metrics = metrics_module.get_skill_metrics(trace_id="trace-1")
        assert len(metrics) == 1
        assert metrics[0]["skill_name"] == "test_skill"
        assert metrics[0]["status"] == status
        assert metrics[0]["elapsed_ms"] == pytest.approx(150.5)
        assert metrics[0]["error_type"] == error_type

    def test_llm_metric_record_and_query(self, temp_metrics_db):
        assert metrics_module.record_llm_metric(
            trace_id="trace-llm", skill_name="test_skill",
            model="gpt-4", provider="openai",
            prompt_tokens=100, completion_tokens=50,
            total_tokens=150, elapsed_ms=200.0, status="success",
        )

        metrics = metrics_module.get_llm_metrics(trace_id="trace-llm")
        assert len(metrics) == 1
        assert metrics[0]["model"] == "gpt-4"
        assert metrics[0]["provider"] == "openai"
        assert metrics[0]["prompt_tokens"] == 100
        assert metrics[0]["completion_tokens"] == 50
        assert metrics[0]["total_tokens"] == 150
        assert metrics[0]["elapsed_ms"] == pytest.approx(200.0)

    def test_llm_metrics_batch(self, temp_metrics_db):
        records = [
            ("trace-batch", "skill_a", "gpt-4", "openai", 10, 20, 30, 100.0, "success"),
            ("trace-batch", "skill_b", "claude-3", "anthropic", 15, 25, 40, 120.0, "success"),
        ]
        count = metrics_module.record_llm_metrics_batch(records)
        assert count == 2

        results = metrics_module.get_llm_metrics(trace_id="trace-batch")
        assert len(results) == 2

    def test_tables_exist(self, temp_metrics_db):
        conn = sqlite3.connect(temp_metrics_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "skill_metrics" in table_names
        assert "llm_metrics" in table_names
        conn.close()

    def test_record_failure_returns_false(self, temp_metrics_db, monkeypatch):
        """数据库异常时返回 False 而不是抛异常"""
        monkeypatch.setattr(metrics_module, "_get_conn", lambda: (_ for _ in ()).throw(sqlite3.Error("mock error")))
        result = metrics_module.record_skill_metric("s", "t", "success", 1.0)
        assert result is False

    def test_query_empty_returns_list(self, temp_metrics_db):
        assert metrics_module.get_skill_metrics(trace_id="nonexistent") == []
        assert metrics_module.get_llm_metrics(trace_id="nonexistent") == []


# ═══════════════════════════════════════════════════════
# BaseSkill 集成测试
# ═══════════════════════════════════════════════════════

class TestBaseSkillWithLogging:

    @pytest.fixture
    def skill_classes(self):
        from src.skills.base.base_skill import BaseSkill
        from typing import Any, Dict

        class SuccessSkill(BaseSkill):
            name = "success_skill"
            description = "Always succeeds"
            def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
                return {"result": "ok"}

        class FailingSkill(BaseSkill):
            name = "failing_skill"
            description = "Always fails"
            def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
                raise ValueError("test failure")

        return SuccessSkill, FailingSkill

    def test_execute_success(self, temp_metrics_db, capture_log, skill_classes):
        SuccessSkill, _ = skill_classes
        result = SuccessSkill()._execute_with_logging(input_data={"test": "data"})

        assert result["result"] == "ok"
        logs = capture_log.read()
        assert any(l["event"] == "skill.start" for l in logs)
        assert any(l["event"] == "skill.end" for l in logs)

        metrics = metrics_module.get_skill_metrics(trace_id=get_trace_id())
        assert len(metrics) == 1
        assert metrics[0]["status"] == "success"

    def test_execute_error(self, temp_metrics_db, capture_log, skill_classes):
        _, FailingSkill = skill_classes

        with pytest.raises(ValueError, match="test failure"):
            FailingSkill()._execute_with_logging(input_data={})

        logs = capture_log.read()
        assert any(l["event"] == "skill.error" for l in logs)

        metrics = metrics_module.get_skill_metrics(trace_id=get_trace_id())
        assert len(metrics) == 1
        assert metrics[0]["status"] == "error"
        assert metrics[0]["error_type"] == "ValueError"


if __name__ == "__main__":
    pytest.main()