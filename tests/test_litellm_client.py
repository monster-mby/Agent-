import json
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from src.core.model_client import (
    LiteLLMClient,
    ChatMessage,
    ToolCall,
    ChatResponse,
    HAS_LITELLM,
    HAS_TENACITY,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
)

# 如果 conftest.py 已 Mock litellm，强制启用测试
import src.core.model_client as mc
if not mc.HAS_LITELLM:
    mc.HAS_LITELLM = True  # 强制启用，因为测试全是 Mock 的

# pytestmark 改为可选跳过（非必须）
pytestmark = pytest.mark.skipif(
    not HAS_LITELLM and "litellm" not in __import__("sys").modules,
    reason="litellm not installed and not mocked"
)




# ──────────────────────────────────────────────
# Fixtures 测试前置依赖
# ──────────────────────────────────────────────

@pytest.fixture
def mock_llm_config():
    """Mock LLMConfig 配置对象，匹配你的src/core/config.py结构"""
    from src.core.config import LLMConfig
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test-1234",
        base_url="https://api.openai.com/v1",
        temperature=0.5,
        max_tokens=1024,
        timeout=30.0,
    )


@pytest.fixture
def mock_get_llm_config(monkeypatch, mock_llm_config):
    """Mock get_llm_config 全局配置获取函数"""

    def mock():
        return mock_llm_config

    monkeypatch.setattr("src.core.model_client.get_llm_config", mock)
    return mock


@pytest.fixture
def mock_litellm_completion():
    """Mock litellm.completion 接口，完全隔离网络请求"""
    mock_response = Mock()
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    mock_choice = Mock()
    mock_choice.finish_reason = "stop"

    mock_message = Mock()
    mock_message.content = "这是一条模拟回复。"
    mock_message.tool_calls = None

    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    with patch("src.core.model_client._litellm_completion", return_value=mock_response) as mock:
        yield mock


@pytest.fixture
def mock_litellm_completion_with_tools():
    """Mock 带工具调用的litellm响应"""
    mock_response = Mock()
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 65

    mock_choice = Mock()
    mock_choice.finish_reason = "tool_calls"

    mock_message = Mock()
    mock_message.content = ""

    # 模拟工具调用结构
    mock_tool_call = Mock()
    mock_func = Mock()
    mock_func.name = "search_web"
    mock_func.arguments = json.dumps({"query": "Python 3.13 新特性"})
    mock_tool_call.function = mock_func

    mock_message.tool_calls = [mock_tool_call]
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    with patch("src.core.model_client._litellm_completion", return_value=mock_response) as mock:
        yield mock


# ──────────────────────────────────────────────
# 1. Pydantic 数据模型测试
# ──────────────────────────────────────────────

class TestPydanticModels:
    def test_chat_message_valid(self):
        """测试ChatMessage正常创建"""
        msg = ChatMessage(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"

    def test_chat_message_missing_required_field(self):
        """测试ChatMessage缺少必填字段时抛出校验错误"""
        with pytest.raises(ValidationError):
            ChatMessage(role="user")  # 缺少content字段

    def test_tool_call_valid(self):
        """测试ToolCall正常创建"""
        tc = ToolCall(name="test_tool", arguments={"param": 123})
        assert tc.name == "test_tool"
        assert tc.arguments == {"param": 123}

    def test_chat_response_default_values(self):
        """测试ChatResponse默认值正确"""
        resp = ChatResponse()
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.usage == {}


# ──────────────────────────────────────────────
# 2. 客户端初始化测试
# ──────────────────────────────────────────────

class TestLiteLLMClientInit:
    def test_init_with_default_config(self, mock_get_llm_config, mock_llm_config):
        """测试使用全局默认配置初始化"""
        client = LiteLLMClient()
        assert client.config.model == mock_llm_config.model
        assert client.config.temperature == 0.5

    def test_init_with_custom_config(self, mock_llm_config):
        """测试传入自定义配置初始化"""
        custom_config = mock_llm_config.model_copy(update={"model": "deepseek-chat"})
        client = LiteLLMClient(config=custom_config)
        assert client.config.model == "deepseek-chat"

    def test_init_with_kwargs_override_config(self, mock_get_llm_config):
        """测试通过kwargs动态覆盖配置字段"""
        client = LiteLLMClient(temperature=0.9, max_tokens=4096)
        assert client.config.temperature == 0.9
        assert client.config.max_tokens == 4096

    def test_init_with_extra_params(self, mock_get_llm_config):
        """测试非配置字段存入extra_defaults"""
        client = LiteLLMClient(tool_choice="required")
        assert "tool_choice" in client._extra_defaults
        assert client._extra_defaults["tool_choice"] == "required"

    def test_litellm_model_property(self, mock_get_llm_config):
        """测试litellm_model属性正常返回"""
        client = LiteLLMClient()
        assert client.litellm_model is not None


# ──────────────────────────────────────────────
# 3. Chat核心对话逻辑测试
# ──────────────────────────────────────────────

class TestLiteLLMClientChat:
    def test_basic_chat_flow(self, mock_get_llm_config, mock_litellm_completion):
        """测试基础对话流程"""
        client = LiteLLMClient()
        messages = [ChatMessage(role="user", content="测试对话")]

        response = client.chat(messages=messages)

        # 校验接口调用
        mock_litellm_completion.assert_called_once()
        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert call_kwargs["model"] == client.litellm_model
        assert call_kwargs["messages"][0]["content"] == "测试对话"

        # 校验返回结果
        assert isinstance(response, ChatResponse)
        assert response.content == "这是一条模拟回复。"
        assert response.usage["total_tokens"] == 30
        assert response.elapsed_ms > 0

    def test_message_normalization(self, mock_get_llm_config, mock_litellm_completion):
        """测试消息格式规范化（支持ChatMessage对象和dict混合）"""
        client = LiteLLMClient()
        mixed_messages = [
            ChatMessage(role="system", content="你是一个专业助手"),
            {"role": "user", "content": "你好"}
        ]

        client.chat(messages=mixed_messages)

        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert len(call_kwargs["messages"]) == 2
        assert all(isinstance(m, dict) for m in call_kwargs["messages"])

    def test_runtime_parameter_override(self, mock_get_llm_config, mock_litellm_completion):
        """测试调用时动态覆盖参数"""
        client = LiteLLMClient()  # 默认temperature=0.5

        client.chat(
            messages=[ChatMessage(role="user", content="test")],
            temperature=1.0,
            max_tokens=500,
            stop=["END"]
        )

        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["stop"] == ["END"]

    def test_tool_call_parsing(self, mock_get_llm_config, mock_litellm_completion_with_tools):
        """测试工具调用响应解析"""
        client = LiteLLMClient()

        response = client.chat(
            messages=[ChatMessage(role="user", content="帮我查一下资料")],
            tools=[{"type": "function", "function": {"name": "search_web"}}]
        )

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_web"
        assert response.tool_calls[0].arguments == {"query": "Python 3.13 新特性"}

    def test_invalid_tool_call_json_fallback(self, mock_get_llm_config):
        """测试工具调用参数JSON解析失败的兜底逻辑"""
        # 构造非法JSON的mock响应
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage = None
        mock_choice = Mock()
        mock_choice.finish_reason = "tool_calls"
        mock_message = Mock()
        mock_message.content = ""

        mock_tool_call = Mock()
        mock_func = Mock()
        mock_func.name = "test_tool"
        mock_func.arguments = "{invalid json}"  # 非法JSON
        mock_tool_call.function = mock_func
        mock_message.tool_calls = [mock_tool_call]

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        with patch("src.core.model_client._litellm_completion", return_value=mock_response):
            client = LiteLLMClient()
            response = client.chat(messages=[{"role": "user", "content": "test"}])

            assert len(response.tool_calls) == 1
            assert "_raw" in response.tool_calls[0].arguments


# ──────────────────────────────────────────────
# 4. 异常处理测试
# ──────────────────────────────────────────────

class TestLiteLLMClientExceptions:
    def test_authentication_error_convert(self, mock_get_llm_config):
        """测试认证错误转换为友好提示"""
        with patch("src.core.model_client._litellm_completion",
                   side_effect=AuthenticationError("Invalid API Key", llm_provider="openai", model="test")):
            client = LiteLLMClient()
            with pytest.raises(RuntimeError, match="认证失败"):
                client.chat(messages=[{"role": "user", "content": "test"}])

    def test_rate_limit_error_convert(self, mock_get_llm_config):
        """测试速率限制错误转换为友好提示"""
        # 临时禁用重试，直接测试异常转换逻辑
        with patch("src.core.model_client.HAS_TENACITY", False):
            with patch("src.core.model_client._litellm_completion",
                       side_effect=RateLimitError("Quota exceeded", llm_provider="openai", model="test")):
                client = LiteLLMClient()
                with pytest.raises(RuntimeError, match="速率限制"):
                    client.chat(messages=[{"role": "user", "content": "test"}])

    def test_service_unavailable_error_convert(self, mock_get_llm_config):
        """测试服务不可用错误转换"""
        with patch("src.core.model_client.HAS_TENACITY", False):
            with patch("src.core.model_client._litellm_completion",
                       side_effect=ServiceUnavailableError("Service down", llm_provider="openai", model="test")):
                client = LiteLLMClient()
                with pytest.raises(RuntimeError, match="LLM 服务调用失败"):
                    client.chat(messages=[{"role": "user", "content": "test"}])

    def test_api_connection_error_convert(self, mock_get_llm_config):
        """测试连接错误转换"""
        with patch("src.core.model_client.HAS_TENACITY", False):
            with patch("src.core.model_client._litellm_completion",
                       side_effect=APIConnectionError("Network error", llm_provider="openai", model="test")):
                client = LiteLLMClient()
                with pytest.raises(RuntimeError, match="LLM 服务调用失败"):
                    client.chat(messages=[{"role": "user", "content": "test"}])


# ──────────────────────────────────────────────
# 5. 重试逻辑测试（仅安装tenacity时运行）
# 【修复】解决异常子句过于宽泛的警告
# ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TENACITY, reason="tenacity not installed")
class TestLiteLLMClientRetry:
    def test_retry_trigger_on_retryable_exception(self, mock_get_llm_config):
        """测试可重试异常会触发重试机制"""
        # 模拟连续2次速率限制异常，第3次成功
        mock_call = Mock(side_effect=[
            RateLimitError("slow down", llm_provider="openai", model="test"),
            RateLimitError("slow down", llm_provider="openai", model="test"),
            Mock(
                choices=[Mock(
                    message=Mock(content="OK", tool_calls=None),
                    finish_reason="stop"
                )],
                model="test-model",
                usage=Mock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            )
        ])

        with patch("src.core.model_client._litellm_completion", mock_call):
            client = LiteLLMClient()
            # 明确捕获预期异常，解决宽泛警告
            try:
                client.chat(messages=[{"role": "user", "content": "test"}])
            except (RateLimitError, RuntimeError):
                pass
            # 验证重试逻辑触发，调用次数≥2
            assert mock_call.call_count >= 2
if __name__ == "__main__":
    pytest.main()