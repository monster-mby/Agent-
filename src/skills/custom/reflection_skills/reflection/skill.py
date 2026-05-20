"""
反思修订技能 — Reflection 模式核心组件

给定「生成结果 + 批评意见」，调用 LLM 生成修订后的版本。
这是反思闭环中的 Reviser 节点使用的技能。

温度配置：0.5（平衡创造性与稳定性）

职责：
- 仅封装 LLM 调用逻辑（不包含流程控制）
- 输入：generation（待修订文本）+ critique（批评报告，复用 state.Critique）
- 输出：revised_generation（修订后文本）

v2.0 优化项：
- 复用 state.py 的 CritiquePoint / Critique / TokenUsage，消除重复类型定义
- 配置（model / temperature / max_tokens / max_retries）通过 __init__ 注入，输入只含业务数据
- 提取 LLMResponseAdapter 统一多客户端响应格式
- 自定义异常 LLMCallError / LLMClientNotConfiguredError / PromptBuildingError
- llm_client 使用 Protocol 类型约束，IDE 自动补全 .chat()
- @timer 装饰器消除计时样板
- execute 拆分为 _build_prompt / _invoke_llm / _assemble_output，可独立测试
- 新增 execute_from_context(ReflectionContext) 简化节点调用
- 重试耗尽抛出 LLMCallError 而非空结果（防止静默数据损坏）
- 移除模拟修订逻辑，LLM 客户端缺失时抛出 LLMClientNotConfiguredError
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.agent.langgraph.state import Critique, CritiquePoint, TokenUsage
from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger("reflection_skill")


# ═══════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════

class LLMCallError(Exception):
    """LLM 调用失败（网络超时、服务不可用等）"""
    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.retry_count = retry_count
        self.original_error = original_error


class LLMClientNotConfiguredError(Exception):
    """LLM 客户端未注入"""
    def __init__(
        self,
        message: str = "LLM 客户端未配置，请通过 __init__ 或 ReflectionInput.llm_client 注入",
    ):
        super().__init__(message)


class PromptBuildingError(Exception):
    """Prompt 构建失败"""
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.original_error = original_error


# ═══════════════════════════════════════════════════════════════
# LLM 客户端协议
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class LLMClientProtocol(Protocol):
    """LLM 客户端协议 — 约束必须实现 .chat() 方法"""

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """发送聊天请求并返回响应"""
        ...


# ═══════════════════════════════════════════════════════════════
# LLM 响应适配器
# ═══════════════════════════════════════════════════════════════

class LLMResponseAdapter:
    """统一多客户端响应格式 → {"content": str, "usage": dict, "model": str}

    支持:
    - 带 .content / .usage / .model 属性的对象（OpenAI / Anthropic SDK）
    - dict 响应 {"content": ..., "usage": ..., "model": ...}
    - 纯字符串响应（无 usage 信息）
    """

    @staticmethod
    def extract(response: Any, default_model: str = "unknown") -> dict:
        """从各种 LLM 客户端响应中提取标准化字段"""
        # 对象格式（OpenAI / Anthropic SDK）
        if hasattr(response, 'content'):
            usage_raw = getattr(response, 'usage', None)
            if usage_raw is not None:
                usage = (
                    usage_raw.model_dump()
                    if hasattr(usage_raw, 'model_dump')
                    else usage_raw
                    if isinstance(usage_raw, dict)
                    else {
                        "prompt_tokens": getattr(usage_raw, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage_raw, 'completion_tokens', 0),
                        "total_tokens": getattr(usage_raw, 'total_tokens', 0),
                    }
                )
            else:
                usage = {}
            return {
                "content": response.content,
                "usage": usage,
                "model": getattr(response, 'model', default_model) or default_model,
            }

        # dict 格式（LangChain / 自建客户端）
        if isinstance(response, dict):
            return {
                "content": response.get("content", "") or "",
                "usage": response.get("usage", {}),
                "model": response.get("model", default_model) or default_model,
            }

        # 纯字符串兜底
        return {"content": str(response), "usage": {}, "model": default_model}


# ═══════════════════════════════════════════════════════════════
# 装饰器
# ═══════════════════════════════════════════════════════════════

def timer(func):
    """计时装饰器：将 elapsed_ms 注入返回的 dict 或带 elapsed_ms 属性的对象"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        if isinstance(result, dict):
            result["elapsed_ms"] = elapsed_ms
        elif hasattr(result, 'elapsed_ms'):
            object.__setattr__(result, 'elapsed_ms', elapsed_ms)
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class ReflectionInput(BaseModel):
    """反思修订输入 — 仅业务数据，静态配置通过 Skill.__init__ 注入"""

    model_config = {"arbitrary_types_allowed": True}  # ← 新增：允许任意类型（Protocol）

    generation: str = Field(
        ..., min_length=1, max_length=50_000,
        description="待修订的生成结果",
    )
    critique: Critique = Field(
        ..., description="批评报告（复用 state.Critique，severity 受 Literal 约束）",
    )
    critique_summary_override: Optional[str] = Field(
        None,
        description="覆盖批评总结（Critique 模型不含 summary 字段时的替代文案）",
    )
    llm_client: Optional[LLMClientProtocol] = Field(
        None,
        description="LLM 客户端（可选；未传则使用 Skill 级默认客户端）",
    )


class ReflectionOutput(BaseModel):
    """反思修订输出"""

    success: bool
    revised_generation: str = Field(default="", description="修订后的生成结果")
    model: str = Field(default="", description="使用的模型")
    usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token 用量（与 state.cumulative_token_usage 同构，方便合并）",
    )
    elapsed_ms: float = Field(default=0.0, description="执行耗时（毫秒）")
    retry_count: int = Field(default=0, description="重试次数")
    error: Optional[str] = Field(None, description="错误信息")


# ═══════════════════════════════════════════════════════════════
# 默认 Prompt 模板
# ═══════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = """你是一个专业的内容修订助手。你的任务是根据批评意见，对给定的文本进行修订和改进。

要求：
1. 严格遵循批评意见中指出的问题和改进建议
2. 保持原文的核心信息和意图不变
3. 修订后的文本应该更加准确、清晰、完整
4. 如果批评意见指出事实性错误，必须修正
5. 如果批评意见指出逻辑问题，必须重构相关段落
6. 保持语言风格一致
7. 不要引入批评意见中未提及的新问题

请直接输出修订后的文本，不要添加任何解释或说明。"""

USER_PROMPT_TEMPLATE = """## 待修订文本

{generation}

## 批评意见

{critique_summary}

{critique_points}

## 请输出修订后的文本："""


# ═══════════════════════════════════════════════════════════════
# 技能类
# ═══════════════════════════════════════════════════════════════

class ReflectionSkill(BaseSkill):
    """反思修订技能 — Reflection 模式核心组件

    职责:
    - 接收生成结果和批评意见
    - 调用 LLM 生成修订版本
    - 返回修订后的文本

    温度配置: 0.5（Reviser 节点专用）

    使用方式::

        # 方式 1：通过 __init__ 注入配置和默认客户端（推荐）
        skill = ReflectionSkill(
            llm_client=my_openai_client,
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            max_retries=3,
        )
        result = skill.execute(ReflectionInput(
            generation="...",
            critique=state["reflection_context"].critique,
        ))

        # 方式 2：从 ReflectionContext 直接调用
        result = skill.execute_from_context(state["reflection_context"])

        # 方式 3：每次调用覆盖 LLM 客户端
        result = skill.execute(ReflectionInput(
            generation="...",
            critique=...,
            llm_client=another_client,
        ))
    """

    name = "reflection"
    description = (
        "根据批评意见修订生成结果。"
        "用于反思模式的 Reviser 节点，将 generation + critique 转化为修订版。"
    )
    version = "2.0.0"
    author = "EnterpriseLearningAgent"
    triggers = [
        "反思修订", "内容修订", "根据批评修改", "reflection",
        "reviser", "修订模式", "改进文本",
    ]

    input_schema = ReflectionInput
    output_schema = ReflectionOutput

    def __init__(
        self,
        llm_client: Optional[LLMClientProtocol] = None,
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_on_failure: bool = True,
    ):
        """初始化反思修订技能

        Args:
            llm_client: 默认 LLM 客户端（需实现 .chat() 方法）
            model: 默认模型名称
            temperature: 生成温度（默认 0.5）
            max_tokens: 最大生成 token 数
            max_retries: 最大重试次数
            retry_on_failure: 失败时是否重试
        """
        super().__init__()
        self._default_llm_client = llm_client
        self._default_model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_on_failure = retry_on_failure

    # ── 公开接口 ────────────────────────────────────────

    @timer
    def execute(self, input_data: ReflectionInput) -> dict:
        """执行反思修订

        Args:
            input_data: 反思修订输入

        Returns:
            ReflectionOutput.model_dump()
        """
        try:
            logger.info(
                "ReflectionSkill 开始 | generation_len=%d | critique_points=%d | "
                "temperature=%.1f | model=%s",
                len(input_data.generation),
                len(input_data.critique.points),
                self._temperature,
                self._resolve_model(),
            )

            # 步骤 1：构建 Prompt
            system_prompt, user_prompt = self._build_prompt(input_data)

            # 步骤 2：调用 LLM（含重试）
            llm_response, retry_count = self._invoke_llm(
                input_data, system_prompt, user_prompt,
            )

            # 步骤 3：组装输出
            return self._assemble_output(input_data, llm_response, retry_count)

        except (LLMCallError, LLMClientNotConfiguredError, PromptBuildingError) as exc:
            logger.exception("ReflectionSkill 执行失败")
            return ReflectionOutput(
                success=False,
                revised_generation="",
                retry_count=getattr(exc, 'retry_count', 0),
                error=str(exc),
            ).model_dump()

        except Exception as exc:
            logger.exception("ReflectionSkill 未预期错误")
            return ReflectionOutput(
                success=False,
                revised_generation="",
                retry_count=0,
                error=f"未预期错误: {exc}",
            ).model_dump()

    def execute_from_context(self, reflection_context) -> ReflectionOutput:
        """从 ReflectionContext 直接调用，自动取出 refined_output + critique

        Args:
            reflection_context: state.py 的 ReflectionContext 实例

        Returns:
            ReflectionOutput

        Raises:
            TypeError: 传入的不是 ReflectionContext
            ValueError: refined_output 或 critique 为空
        """
        from src.agent.langgraph.state import ReflectionContext

        if not isinstance(reflection_context, ReflectionContext):
            raise TypeError(
                f"期望 ReflectionContext，实际 {type(reflection_context).__name__}"
            )

        if reflection_context.refined_output is None:
            raise ValueError("ReflectionContext.refined_output 为空，无法修订")
        if reflection_context.critique is None:
            raise ValueError("ReflectionContext.critique 为空，缺少批评意见")

        result_dict = self.execute(ReflectionInput(
            generation=reflection_context.refined_output,
            critique=reflection_context.critique,
        ))
        return ReflectionOutput(**result_dict)

    # ── 步骤 1: 构建 Prompt ─────────────────────────────

    def _build_prompt(self, inp: ReflectionInput) -> tuple[str, str]:
        """构建 system + user prompt

        Returns:
            (system_prompt, user_prompt)
        """
        try:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            user_prompt = self._render_user_prompt(
                generation=inp.generation,
                critique=inp.critique,
                summary_override=inp.critique_summary_override,
            )
            return system_prompt, user_prompt
        except Exception as exc:
            raise PromptBuildingError(
                f"Prompt 构建失败: {exc}",
                original_error=exc,
            ) from exc

    def _render_user_prompt(
        self,
        generation: str,
        critique: Critique,
        summary_override: Optional[str] = None,
    ) -> str:
        """渲染 user prompt 模板

        Args:
            generation: 待修订文本
            critique: 批评报告（复用 state.Critique）
            summary_override: 覆盖批评总结
        """
        # 批评总结：优先覆盖值 → Critique.summary → 默认文案
        critique_summary = (
            summary_override
            or getattr(critique, 'summary', None)
            or "以下是详细的批评意见："
        )

        # 渲染批评点列表
        critique_points = []
        for i, point in enumerate(critique.points, start=1):
            severity_label = point.severity.upper()
            point_str = f"{i}. [{severity_label}] {point.description}"
            if point.suggested_fix:
                point_str += f"\n   建议修复：{point.suggested_fix}"
            critique_points.append(point_str)

        critique_points_str = (
            "\n\n".join(critique_points)
            if critique_points
            else "无具体批评点"
        )

        return USER_PROMPT_TEMPLATE.format(
            generation=generation,
            critique_summary=critique_summary,
            critique_points=critique_points_str,
        )

    # ── 步骤 2: 调用 LLM ───────────────────────────────

    def _invoke_llm(
        self,
        inp: ReflectionInput,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict, int]:
        """调用 LLM（含重试逻辑）

        Returns:
            ({"content": str, "usage": dict, "model": str}, retry_count)

        Raises:
            LLMClientNotConfiguredError: 客户端未配置
            LLMCallError: 全部重试耗尽仍失败
        """
        llm_client = self._resolve_llm_client(inp)

        if llm_client is None:
            raise LLMClientNotConfiguredError()

        if not self._retry_on_failure or self._max_retries <= 0:
            return self._do_llm_call(llm_client, system_prompt, user_prompt), 0

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                result = self._do_llm_call(llm_client, system_prompt, user_prompt)
                if attempt > 0:
                    logger.info("LLM 重试第 %d 次成功", attempt)
                return result, attempt
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = min(2 ** attempt, 10)  # 指数退避，上限 10s
                    logger.warning(
                        "LLM 调用失败（第 %d/%d 次），%ds 后重试: %s",
                        attempt + 1, self._max_retries, wait, exc,
                    )
                    time.sleep(wait)

        # 全部重试耗尽 → 抛异常（不返回空结果，防止静默数据损坏）
        logger.error("LLM 重试 %d 次后仍失败", self._max_retries)
        raise LLMCallError(
            f"LLM 重试 {self._max_retries} 次后仍失败: {last_error}",
            retry_count=self._max_retries,
            original_error=last_error,
        )

    def _do_llm_call(
        self,
        client: LLMClientProtocol,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """单次 LLM 调用

        Returns:
            {"content": str, "usage": dict, "model": str}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat(
                messages=messages,
                model=self._default_model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return LLMResponseAdapter.extract(
                response,
                default_model=self._default_model or "unknown",
            )
        except Exception as exc:
            logger.error("LLM 调用失败: %s", exc)
            raise

    # ── 步骤 3: 组装输出 ───────────────────────────────

    def _assemble_output(
        self,
        inp: ReflectionInput,
        llm_response: dict,
        retry_count: int,
    ) -> dict:
        """组装最终 ReflectionOutput

        （elapsed_ms 由 execute 上的 @timer 装饰器自动注入）
        """
        revised_text = llm_response.get("content", "") or ""
        usage_raw = llm_response.get("usage", {})
        model_used = llm_response.get("model", self._resolve_model())

        # 将原始 usage dict 转为 TokenUsage 模型（与 state.cumulative_token_usage 同构）
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            model=model_used,
        )

        logger.info(
            "ReflectionSkill 完成 | revised_len=%d | retries=%d | tokens=%d",
            len(revised_text), retry_count, usage.total_tokens,
        )

        return ReflectionOutput(
            success=True,
            revised_generation=revised_text,
            model=model_used,
            usage=usage,
            retry_count=retry_count,
        ).model_dump()

    # ── 内部工具方法 ────────────────────────────────────

    def _resolve_llm_client(self, inp: ReflectionInput) -> Optional[LLMClientProtocol]:
        """解析 LLM 客户端：输入级覆盖 > Skill 级默认"""
        client = inp.llm_client or self._default_llm_client
        if client is not None and not isinstance(client, LLMClientProtocol):
            logger.warning(
                "LLM 客户端 %s 未实现 LLMClientProtocol 协议，可能导致调用失败",
                type(client).__name__,
            )
        return client

    def _resolve_model(self) -> Optional[str]:
        """解析模型名称（配置已从输入分离，统一使用 Skill 级默认）"""
        return self._default_model