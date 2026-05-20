"""测试结构化日志和指标埋点功能"""
import pytest
import json
import os
import sqlite3
import sys
from io import StringIO
from src.infrastructure.logger import set_trace_id, get_trace_id, set_skill_name, log_info, log_error, clear_context
from src.infrastructure.metrics import record_skill_metric, record_llm_metric, get_skill_metrics, get_llm_metrics, DB_PATH, _get_conn


class TestStructuredLogging:
    """测试结构化日志功能"""

    def test_trace_id_generation(self):
        """测试 trace_id 生成和获取"""
        clear_context()

        # 测试自动生成
        trace_id = set_trace_id()
        assert trace_id != ""
        assert len(trace_id) > 0

        # 测试获取
        retrieved_id = get_trace_id()
        assert retrieved_id == trace_id

        # 测试手动设置
        custom_id = "test-trace-123"
        set_trace_id(custom_id)
        assert get_trace_id() == custom_id

        clear_context()

    def test_structured_log_output(self, capsys):
        """测试结构化日志输出为 JSON 格式"""
        clear_context()
        set_trace_id("test-trace-456")
        set_skill_name("test_skill")

        # 记录日志
        log_info("skill.start", "Skill started", {"input": "test"}, 50.5)

        # 捕获 stdout 输出
        captured = capsys.readouterr()

        # 验证有输出
        assert captured.out != "", "日志应该输出到 stdout"
        assert captured.out.strip().startswith("{"), "日志应该是 JSON 格式"

        # 验证是有效的 JSON
        log_data = json.loads(captured.out.strip())

        # 验证必需字段
        assert "timestamp" in log_data
        assert "level" in log_data
        assert log_data["level"] == "INFO"
        assert "event" in log_data
        assert log_data["event"] == "skill.start"
        assert "message" in log_data
        assert log_data["message"] == "Skill started"
        assert "context" in log_data
        assert log_data["context"]["trace_id"] == "test-trace-456"
        assert log_data["context"]["skill_name"] == "test_skill"
        assert "data" in log_data
        assert log_data["data"] == {"input": "test"}
        assert "elapsed_ms" in log_data
        assert log_data["elapsed_ms"] == 50.5

        clear_context()

    def test_error_log(self, capsys):
        """测试错误日志记录"""
        clear_context()
        set_trace_id("test-error-trace")

        log_error("skill.error", "Skill failed", {"error_type": "ValueError"}, 100.0)

        captured = capsys.readouterr()
        assert captured.out != "", "错误日志应该输出到 stdout"

        log_data = json.loads(captured.out.strip())

        assert log_data["level"] == "ERROR"
        assert log_data["event"] == "skill.error"
        assert log_data["context"]["trace_id"] == "test-error-trace"

        clear_context()

    def test_log_missing_trace_id_defaults(self, capsys):
        """测试没有设置 trace_id 时的默认行为"""
        clear_context()

        log_info("test.event", "No trace id set")

        captured = capsys.readouterr()
        assert captured.out != "", "日志应该输出到 stdout"

        log_data = json.loads(captured.out.strip())
        # trace_id 应该是空字符串
        assert log_data["context"]["trace_id"] == ""

        clear_context()


class TestMetricsRecording:
    """测试指标埋点功能"""

    def test_skill_metric_recording(self):
        """测试技能指标记录"""
        # 记录一个技能指标
        success = record_skill_metric(
            skill_name="test_skill",
            trace_id="test-trace-skill",
            status="success",
            elapsed_ms=150.5,
            error_type=None
        )

        assert success is True

        # 查询刚记录的指标
        metrics = get_skill_metrics(trace_id="test-trace-skill", limit=1)
        assert len(metrics) == 1
        assert metrics[0]["skill_name"] == "test_skill"
        assert metrics[0]["status"] == "success"
        assert metrics[0]["elapsed_ms"] == 150.5

    def test_skill_metric_error_recording(self):
        """测试技能失败指标记录"""
        success = record_skill_metric(
            skill_name="failing_skill",
            trace_id="test-trace-error",
            status="error",
            elapsed_ms=50.0,
            error_type="RuntimeError"
        )

        assert success is True

        metrics = get_skill_metrics(trace_id="test-trace-error", limit=1)
        assert len(metrics) == 1
        assert metrics[0]["status"] == "error"
        assert metrics[0]["error_type"] == "RuntimeError"

    def test_llm_metric_recording(self):
        """测试 LLM 指标记录"""
        success = record_llm_metric(
            trace_id="test-trace-llm",
            skill_name="test_skill",
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            elapsed_ms=200.0,
            status="success"
        )

        assert success is True

        # 查询刚记录的指标
        metrics = get_llm_metrics(trace_id="test-trace-llm", limit=1)
        assert len(metrics) == 1
        assert metrics[0]["model"] == "gpt-4"
        assert metrics[0]["provider"] == "openai"
        assert metrics[0]["prompt_tokens"] == 100
        assert metrics[0]["completion_tokens"] == 50
        assert metrics[0]["total_tokens"] == 150
        assert metrics[0]["elapsed_ms"] == 200.0

    def test_tables_exist(self):
        """测试数据库表结构存在"""
        # 通过 _get_conn() 获取连接，确保表已初始化
        conn = _get_conn()

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row[0] for row in cursor.fetchall()}

        assert "skill_metrics" in table_names, f"skill_metrics 表不存在，现有表: {table_names}"
        assert "llm_metrics" in table_names, f"llm_metrics 表不存在，现有表: {table_names}"


class TestBaseSkillWithLogging:
    """测试 BaseSkill 的日志和指标集成"""

    def test_execute_success(self, capsys):
        """测试技能成功执行时的日志和指标"""
        from src.skills.base.base_skill import BaseSkill
        from typing import Any, Dict

        class TestSkill(BaseSkill):
            name = "test_logging_skill"
            description = "Test skill for logging"

            def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
                return {"result": "success"}

        skill = TestSkill()
        result = skill._execute_with_logging(input_data={"test": "data"})

        assert result["result"] == "success"

        # 验证有日志输出到 stdout
        captured = capsys.readouterr()
        assert captured.out != "", "应该有日志输出"

        # 解析日志行
        log_lines = [line.strip() for line in captured.out.split('\n') if line.strip()]
        assert len(log_lines) >= 2, f"应该至少有 start 和 end 两条日志，实际: {len(log_lines)}"

        # 验证第一条是 skill.start
        start_log = json.loads(log_lines[0])
        assert start_log["event"] == "skill.start"

        # 验证最后一条是 skill.end
        end_log = json.loads(log_lines[-1])
        assert end_log["event"] == "skill.end"
        assert end_log["level"] == "INFO"

        # 验证指标已记录
        trace_id = get_trace_id()
        metrics = get_skill_metrics(trace_id=trace_id, limit=1)
        assert len(metrics) > 0
        assert metrics[0]["status"] == "success"

    def test_execute_error(self, capsys):
        """测试技能失败执行时的日志和指标"""
        from src.skills.base.base_skill import BaseSkill
        from typing import Any, Dict

        class FailingSkill(BaseSkill):
            name = "failing_test_skill"
            description = "Test skill that fails"

            def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
                raise ValueError("Test error")

        skill = FailingSkill()

        with pytest.raises(ValueError, match="Test error"):
            skill._execute_with_logging(input_data={})

        # 验证有错误日志输出
        captured = capsys.readouterr()
        assert captured.out != "", "应该有错误日志输出"

        # 解析日志行
        log_lines = [line.strip() for line in captured.out.split('\n') if line.strip()]
        assert len(log_lines) >= 2, f"应该至少有 start 和 error 两条日志，实际: {len(log_lines)}"

        # 验证第一条是 skill.start
        start_log = json.loads(log_lines[0])
        assert start_log["event"] == "skill.start"

        # 验证最后一条是 skill.error
        error_log = json.loads(log_lines[-1])
        assert error_log["event"] == "skill.error"
        assert error_log["level"] == "ERROR"

        # 验证错误指标已记录
        trace_id = get_trace_id()
        metrics = get_skill_metrics(trace_id=trace_id, limit=1)
        assert len(metrics) > 0
        assert metrics[0]["status"] == "error"
        assert metrics[0]["error_type"] == "ValueError"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
