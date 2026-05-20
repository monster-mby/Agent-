"""
这是一个企业级基于 LiteLLM 的统一模型客户端，通过一个接口对接 100+ LLM 提供商，配置由 LLMConfig 统一管理，支持参数动态覆盖、Pydantic 类型安全输入输出、指数退避重试、结构化日志、Token 用量透传等能力，完美适配 OpenAI、DeepSeek、豆包等主流大模型。下面从功能概览、核心结构设计、类 / 方法深度详解、方法调用链路、代码质量评价五个维度深度解析。
一、功能概览
这个工具的核心能力是：
多提供商统一接口：基于 LiteLLM 实现，一个接口对接 OpenAI、DeepSeek、Anthropic、Ollama 等 100+ 提供商，无需为每个提供商写单独代码。
配置统一管理：完全复用 LLMConfig 管理配置，客户端不重复拆解字段，支持初始化时动态覆盖参数（自动生成临时配置，不污染原始配置）。
类型安全输入输出：用 Pydantic 模型定义 ChatMessage（对话消息）、ToolCall（工具调用）、ChatResponse（统一响应），从源头保证数据类型合法。
企业级重试机制：可选集成 tenacity，对 RateLimitError（速率限制）、ServiceUnavailableError（服务不可用）、APIConnectionError（连接失败）实现指数退避重试，最大重试 3 次，等待时间 2-30 秒。
全链路可观测性：内置结构化日志，记录请求参数、响应模型、Token 用量、耗时；ChatResponse 透传实际使用模型、Token 用量、结束原因、调用耗时，方便调试与成本统计。
灵活的参数透传：所有 LiteLLM 参数可通过 **extra_kwargs 透传，支持工具调用（Function Calling）、停止词、核采样等高级功能。


LiteLLMClient.__init__()（初始化）
  ├→ 检查 litellm 是否安装
  ├→ 获取基础配置（LLMConfig 或 get_llm_config()）
  ├→ 处理 kwargs 覆盖（生成临时 config / 存入 _extra_defaults）
  └→ 输出初始化日志

LiteLLMClient.chat()（核心对话）
  ├→ 规范化 messages（模型转 dict）
  ├→ 构建请求参数（基础 + 可选 + 工具 + 额外参数）
  ├→ 输出 debug 日志
  ├→ _call_with_retry()（重试调度）
  │   ├─ tenacity 可用 → _call_litellm_with_retry()
  │   │   └→ 带指数退避重试调用 litellm.completion
  │   └─ tenacity 不可用 → _call_litellm()
  │       └→ 直接调用 litellm.completion
  ├→ 解析响应（content / tool_calls / usage / finish_reason / 耗时）
  ├→ 封装为 ChatResponse
  ├→ 输出 info 日志
  └→ 返回 ChatResponse
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from src.core.models import ToolCall as _BaseToolCall, ChatResponse as _BaseChatResponse


from src.core.config import LLMConfig, get_llm_config

# 导入基础设施模块（带容错）
try:
    from src.infrastructure.logger import get_trace_id, log_info, log_error
    from src.infrastructure.metrics import record_llm_metric
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    # 提供降级函数
    def get_trace_id(): return ""
    def log_info(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass
    def record_llm_metric(*args, **kwargs): pass


# ── 可选依赖检查 ──
try:
    import litellm
    from litellm import completion as _litellm_completion
    from litellm.exceptions import (
        APIError,
        AuthenticationError,
        RateLimitError,
        ServiceUnavailableError,
        APIConnectionError,
    )

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    _litellm_completion = None  # type: ignore[assignment]
    # 定义占位异常，避免 except 块报错
    class APIError(Exception): pass                        # type: ignore[no-redef]
    class AuthenticationError(APIError): pass              # type: ignore[no-redef]
    class RateLimitError(APIError): pass                   # type: ignore[no-redef]
    class ServiceUnavailableError(APIError): pass          # type: ignore[no-redef]
    class APIConnectionError(APIError): pass               # type: ignore[no-redef]

logger = logging.getLogger(__name__)

__all__ = [
    "LiteLLMClient",
    "ChatMessage",
    "ToolCall",
    "ChatResponse",
    "HAS_LITELLM",
    "HAS_TENACITY",
    "AuthenticationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "APIError",
]

# ═══════════════════════════════════════════════════════════════
# 重试配置（tenacity 可选）
# ═══════════════════════════════════════════════════════════════

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )

    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    logger.info("tenacity 未安装，重试功能不可用。建议: pip install tenacity")


# ═══════════════════════════════════════════════════════════════
# Pydantic 输入/输出模型
# ═══════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: str = Field(..., description="user | assistant | system | tool")
    content: str = Field(..., description="消息内容")


class ToolCall(_BaseToolCall):
    """工具调用（扩展 id 字段）"""
    pass

class ChatResponse(_BaseChatResponse):
    """chat() 方法统一返回格式（扩展字段）"""
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = ""
    elapsed_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 整个客户端的核心，负责初始化、配置管理、对话调用、重试调度、结果解析、日志记录。
# ═══════════════════════════════════════════════════════════════
class LiteLLMClient:
    """
    基于 LiteLLM 的统一 LLM 客户端。

    使用方式：
      # 自动从环境变量 / .env 加载配置
      client = LiteLLMClient()

      # 显式指定
      client = LiteLLMClient(LLMConfig(provider="deepseek", model="deepseek-chat"))

      # 构造时覆盖部分参数（自动新建临时 config）
      client = LiteLLMClient(temperature=0.7, max_tokens=2048)
    """

    # ═══════════════════════════════════════════════════════════════
    # 作用：初始化客户端，获取基础配置，处理临时参数覆盖，不污染原始配置。。
    # ═══════════════════════════════════════════════════════════════
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        if not HAS_LITELLM:
            raise ImportError(
                "litellm 未安装，无法使用 LiteLLMClient。请执行: pip install litellm"
            )

        # 获取基础配置
        base_config = config or get_llm_config()

        # 如果传了 kwargs，生成新的临时 config（不污染原始 config）
        if kwargs:
            # 只覆盖同时存在于 LLMConfig 和 kwargs 的字段
            model_fields = type(base_config).model_fields

            update_fields = {
                k: v for k, v in kwargs.items()
                if k in model_fields
            }
            if update_fields:
                base_config = base_config.model_copy(update=update_fields)
            # 非 config 字段（如 tool_choice）存下来，在 chat() 里使用
            self._extra_defaults: dict[str, Any] = {
                k: v for k, v in kwargs.items()
                if k not in model_fields
            }

        else:
            self._extra_defaults = {}

        self.config: LLMConfig = base_config

        logger.info(
            "LiteLLMClient 初始化完成 | provider=%s model=%s base_url=%s api_key=%s timeout=%.0fs",
            self.config.provider,
            self.config.model,
            self.config.base_url or "<default>",
            self.config.api_key_masked,
            self.config.timeout,
        )

    # ═══════════════════════════════════════════════════════════
    # 属性
    # ═══════════════════════════════════════════════════════════

    @property
    def litellm_model(self) -> str:
        """LiteLLM 模型标识，直接复用 config"""
        return self.config.litellm_model

    # ═══════════════════════════════════════════════════════════
    # 作用：与 LLM 对话的统一入口，处理参数规范化、请求构建、重试调度、结果解析、日志记录。
    # ═══════════════════════════════════════════════════════════

    def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],#对话消息列表，支持模型或原生 dict
        tools: Optional[list[dict[str, Any]]] = None, #OpenAI Function Calling 格式的工具定义
        tool_choice: Optional[str] = None, # 工具选择策略："auto"/"none"/"required"/ 特定工具名
        temperature: Optional[float] = None, # 覆盖 config 中的温度
        max_tokens: Optional[int] = None, # 覆盖 config 中的最大 token 数
        top_p: Optional[float] = None, # 核采样参数
        stop: Optional[list[str]] = None, # 停止词列表
        **extra_kwargs, # 透传给 litellm.completion 的其他参数
    ) -> ChatResponse:
        """
        与 LLM 对话。

        Args:
            messages: 对话消息列表
            tools: OpenAI function-calling 格式的工具定义
            tool_choice: "auto" | "none" | "required" | 特定工具
            temperature: 覆盖 config 中的温度
            max_tokens: 覆盖 config 中的最大 token 数
            top_p: 核采样参数
            stop: 停止词列表
            **extra_kwargs: 透传给 litellm.completion 的其他参数

        Returns:
            ChatResponse: 统一响应格式（含内容、工具调用、用量、耗时）
        """
        # 记录LLM请求开始
        if INFRASTRUCTURE_AVAILABLE:
            try:
                trace_id = get_trace_id()
                log_info("llm.request", f"LLM request to model {self.litellm_model}",
                         {"model": self.litellm_model, "provider": self.config.provider})
            except Exception as e:
                logger.warning(f"记录LLM请求日志失败: {e}")
        else:
            trace_id = ""

        t_start = time.perf_counter()


        # 规范化 messages
        normalized_messages: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg.model_dump())
            else:
                normalized_messages.append(msg)

        # 构建请求参数
        request_kwargs: dict[str, Any] = {
            "model": self.litellm_model,
            "messages": normalized_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        if self.config.api_key:
            request_kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            request_kwargs["api_base"] = self.config.base_url
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop:
            request_kwargs["stop"] = stop
        if self.config.extra_headers:
            request_kwargs["extra_headers"] = self.config.extra_headers

        # tools
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = tool_choice or "auto"
            # 如果有 tool_choice 且不是 auto/none/required，传具体工具名
            if tool_choice and tool_choice not in ("auto", "none", "required"):
                request_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
            else:
                request_kwargs["tool_choice"] = tool_choice or "auto"

        # 合并额外参数
        request_kwargs.update(self._extra_defaults)
        request_kwargs.update(extra_kwargs)

        logger.debug("litellm 请求: model=%s temperature=%.2f max_tokens=%d tools=%d",
                      request_kwargs["model"], request_kwargs["temperature"],
                      request_kwargs["max_tokens"], len(tools or []))

        # ── 带重试的调用 ──
        response = self._call_with_retry(request_kwargs)

        # ── 解析响应 ──
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        message = response.choices[0].message
        content = message.content or ""

        # 解析 tool_calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            func = getattr(tc, "function", None)
            if not func:
                continue
            args = func.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("tool_call 参数 JSON 解析失败，保留原始字符串: %s", args[:200])
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=func.name, arguments=args))

        # 提取用量
        usage: dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        finish_reason = getattr(
            getattr(response.choices[0], "finish_reason", None),
            "__str__", lambda: ""
        )() or ""

        chat_response = ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=response.model if hasattr(response, "model") else self.litellm_model,
            usage=usage,
            finish_reason=finish_reason,
            elapsed_ms=round(elapsed_ms, 1),
        )

        logger.info(
            "litellm 响应 | model=%s finish=%s tokens_in=%d tokens_out=%d elapsed=%.0fms",
            chat_response.model,
            chat_response.finish_reason,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            elapsed_ms,
        )

        # 记录LLM指标
        if INFRASTRUCTURE_AVAILABLE:
            try:
                record_llm_metric(
                    trace_id=trace_id,
                    skill_name="",
                    model=chat_response.model,
                    provider=self.config.provider,
                    prompt_tokens=chat_response.usage.get("prompt_tokens", 0),
                    completion_tokens=chat_response.usage.get("completion_tokens", 0),
                    total_tokens=chat_response.usage.get("total_tokens", 0),
                    elapsed_ms=chat_response.elapsed_ms,
                    status="success"
                )
            except Exception as e:
                logger.warning(f"记录LLM指标失败: {e}")

        return chat_response

    # ═══════════════════════════════════════════════════════════
    # 重试逻辑
    # ═══════════════════════════════════════════════════════════

    # 作用：根据 tenacity 是否可用，选择带重试或直接调用的方式。
    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """
        调用 litellm.completion，可选重试。

        可重试异常：RateLimitError, ServiceUnavailableError, APIConnectionError
        不可重试：AuthenticationError（密钥错误重试也没用）
        """
        if HAS_TENACITY:
            return self._call_litellm_with_retry(kwargs)
        else:
            return self._call_litellm(kwargs)

    @staticmethod
    def _call_litellm(kwargs: dict[str, Any]) -> Any:
        """直接调用 litellm（无重试）"""
        try:
            return _litellm_completion(**kwargs)
        except AuthenticationError as e:
            raise RuntimeError(
                f"LLM 认证失败，请检查 api_key。原始错误: {e}"
            ) from e
        except RateLimitError as e:
            raise RuntimeError(
                f"LLM 速率限制，请稍后重试或降低请求频率。原始错误: {e}"
            ) from e
        except (APIError, ServiceUnavailableError, APIConnectionError) as e:
            raise RuntimeError(
                f"LLM 服务调用失败: {e}"
            ) from e

    if HAS_TENACITY:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((
                RateLimitError,
                ServiceUnavailableError,
                APIConnectionError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call_litellm_with_retry(self, kwargs: dict[str, Any]) -> Any:

            """带指数退避重试的 litellm 调用"""
            try:
                return _litellm_completion(**kwargs)
            except AuthenticationError as e:
                raise RuntimeError(
                    f"LLM 认证失败，请检查 api_key。原始错误: {e}"
                ) from e
            except (APIError, ServiceUnavailableError, APIConnectionError, RateLimitError) as e:
                logger.warning("litellm 调用失败（将重试）: %s", e)
                raise

    # ═══════════════════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"LiteLLMClient(provider={self.config.provider!r}, "
            f"model={self.config.model!r}, "
            f"base_url={self.config.base_url!r}, "
            f"api_key={self.config.api_key_masked!r})"
        )

__all__ = [
    "LiteLLMClient",
    "ChatMessage",
    "ToolCall",
    "ChatResponse",
    "HAS_LITELLM",
    "HAS_TENACITY",
    "AuthenticationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "APIError",
]
# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    client = LiteLLMClient()

    response = client.chat(
        messages=[ChatMessage(role="user", content="用一句话介绍 Python")],
    )
    print("\n回复:", response.content)
    print("用量:", response.usage)
    print("耗时:", response.elapsed_ms, "ms")
    print("模型:", response.model)