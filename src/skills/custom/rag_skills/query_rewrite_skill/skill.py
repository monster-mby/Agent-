"""
QueryRewriteSkill — 基于 qwen-turbo 的查询改写技能

功能：指代消解 / 歧义消除 / 多意图拆分
依赖：pip install openai tenacity json5
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pydantic Schema
# ──────────────────────────────────────────────

class QueryHistoryItem(BaseModel):
    """对话历史项"""
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="对话内容")


class QueryRewriteInput(BaseModel):
    """QueryRewriteSkill 输入模型"""
    original_query: str = Field(..., description="用户原始问题")
    history: List[QueryHistoryItem] = Field(
        default_factory=list, description="对话历史（自动取最近 N 轮）"
    )
    max_sub_queries: int = Field(default=3, description="最大子问题拆分数量")
    history_window: Optional[int] = Field(
        default=None, description="历史窗口轮数，不传则使用类默认值"
    )
    temperature: Optional[float] = Field(
        default=None, description="LLM 温度，不传则使用类默认值"
    )
    api_key: Optional[str] = Field(None, description="DashScope API Key")


class QueryRewriteOutput(BaseModel):
    """QueryRewriteSkill 输出模型"""
    rewritten_queries: List[str] = Field(
        ..., description="改写后的查询列表（单意图或多意图拆分）"
    )
    is_rewritten: bool = Field(default=False, description="是否经过了模型改写")


# ──────────────────────────────────────────────
#  可重试异常判断（tenacity 用）
# ──────────────────────────────────────────────

def _is_retryable(exception: BaseException) -> bool:
    """仅对网络超时、服务端 5xx 做重试；4xx 不重试"""
    from openai import (
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )

    # 明确的超时 / 连接异常 → 重试
    if isinstance(exception, (APITimeoutError, APIConnectionError)):
        return True

    # 5xx → 重试
    if isinstance(exception, InternalServerError):
        return True

    # 其他（4xx / RateLimitError / BadRequestError 等）→ 不重试
    return False


# ──────────────────────────────────────────────
#  Skill 实现
# ──────────────────────────────────────────────

class QueryRewriteSkill(BaseSkill):
    """基于 qwen-turbo 对用户问题进行指代消解、歧义消除及多意图拆分"""

    name = "query_rewrite_skill"
    description = "基于 qwen-turbo 对用户问题进行指代消解、歧义消除及多意图拆分"
    version = "0.2.0"
    author = "monster"
    triggers = ["改写", "优化查询", "补充上下文"]

    input_schema = QueryRewriteInput
    output_schema = QueryRewriteOutput

    # ── 可配置类常量 ──
    DEFAULT_MODEL = "qwen-turbo"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TIMEOUT = 10                    # 秒
    DEFAULT_HISTORY_WINDOW = 3              # 最近 N 轮对话
    RETRY_MAX_ATTEMPTS = 3                  # 含首次，共 3 次
    RETRY_MIN_WAIT = 1                      # 指数退避起始秒
    RETRY_MAX_WAIT = 10                     # 指数退避最长秒

    # 前置校验：指代词列表
    PRONOUNS = ["这个", "那个", "它", "他", "她", "上次", "刚才", "前者", "后者"]
    # 前置校验：并列连词列表
    CONJUNCTIONS = ["并且", "还有", "同时", "以及", "另外", "和", "与"]
    # 最小查询长度（字符），低于此值不触发改写
    MIN_QUERY_LENGTH = 5

    # ── 客户端懒加载单例 ──
    _client: Optional[OpenAI] = None

    @classmethod
    def _get_client(cls, api_key: str) -> OpenAI:
        """获取或创建 OpenAI 客户端（懒加载单例）"""
        if cls._client is None:
            cls._client = OpenAI(
                api_key=api_key,
                base_url=cls.DEFAULT_BASE_URL,
            )
            logger.debug("OpenAI 客户端已初始化 | base_url=%s", cls.DEFAULT_BASE_URL)
        return cls._client

    # ── 前置校验 ──

    def _check_need_rewrite(
            self, query: str, history: List[QueryHistoryItem]
    ) -> bool:
        """判断是否需要调用模型进行改写"""
        query_stripped = query.strip()

        # 包含指代词或并列连词，直接触发
        if any(p in query for p in self.PRONOUNS):
            return True
        if any(c in query for c in self.CONJUNCTIONS):
            return True

        # 有历史 + 查询不是纯确认/寒暄（长度 >= 阈值）
        if len(history) > 0 and len(query_stripped) >= self.MIN_QUERY_LENGTH:
            return True

        # 过短查询（< 5字符）且无明确意图信号，通常不需要改写（除非有历史）
        if len(query_stripped) < self.MIN_QUERY_LENGTH:
            return False

        return False

    # ── Prompt 构建 ──

    def _build_prompt(
        self, query: str, history: List[QueryHistoryItem], max_splits: int, history_window: int
    ) -> List[Dict[str, str]]:
        """构建系统提示词与消息列表"""
        recent_history = history[-history_window:] if history else []
        history_str = "\n".join(
            [f"{h.role}: {h.content}" for h in recent_history]
        )

        system_prompt = (
            "你是一个专业的搜索查询优化助手。请根据对话历史和用户当前问题，执行以下操作：\n"
            "1. **指代消解**：将“这个”、“它”等代词替换为对话历史中的具体实体。\n"
            "2. **歧义消除**：将模糊的口语转化为精准的检索陈述句。\n"
            f"3. **多意图拆分**：如果问题包含多个独立诉求，将其拆分为最多 {max_splits} 个子问题。\n"
            "\n"
            "**输出要求**：\n"
            "- 仅输出一个 JSON 数组，例如：[\"子问题1\", \"子问题2\"]\n"
            "- 如果没有多意图，也请输出数组：[\"改写后的完整问题\"]\n"
            "- 不要输出任何解释性文字，不要用 Markdown 代码块包裹。\n"
            "\n"
            f"对话历史：\n{history_str if history_str else '无'}"
        )

        logger.debug("Prompt 构建完成 | history_rounds=%d | max_splits=%d",
                     len(recent_history), max_splits)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    # ── 输出解析 ──

    @staticmethod
    def _parse_llm_output(content: str, max_queries: int, fallback: str) -> List[str]:
        """鲁棒解析 LLM 输出为字符串列表

        处理策略（按优先级）：
        1. 正则提取 Markdown 代码块（```json ... ``` 或 ``` ... ```）
        2. json5 宽松解析（容错 trailing comma / 注释）
        3. json 标准解析
        4. 行分割兜底
        """
        if not content:
            logger.warning("LLM 返回空内容，使用兜底")
            return [fallback]

        original = content
        content = content.strip()

        # ── Step 1: 提取 Markdown 代码块 ──
        md_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if md_match:
            content = md_match.group(1).strip()
            logger.debug("从 Markdown 代码块中提取 JSON")

        # ── Step 2: 尝试 json5 宽松解析 ──
        try:
            import json5
            parsed = json5.loads(content)
            if isinstance(parsed, list):
                queries = [str(q) for q in parsed if q]
                if queries:
                    return queries[:max_queries]
        except ImportError:
            logger.debug("json5 未安装，回退到标准 json")
        except Exception as exc:
            logger.debug("json5 解析失败: %s", exc)

        # ── Step 3: 标准 JSON 解析 ──
        # 先尝试找到 JSON 数组的起止位置
        array_match = re.search(r'\[[\s\S]*\]', content)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    queries = [str(q) for q in parsed if q]
                    if queries:
                        return queries[:max_queries]
            except json.JSONDecodeError as exc:
                logger.debug("标准 JSON 解析失败: %s", exc)

        # ── Step 4: 行分割兜底 ──
        logger.warning("无法解析为 JSON，使用行分割兜底 | raw=%s", original[:200])
        queries = [
            q.strip()
            for q in content.replace("|", "\n").split("\n")
            if q.strip()
        ]
        queries = queries[:max_queries]

        if not queries:
            return [fallback]
        return queries

    # ── 带重试的 API 调用 ──

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_llm(
        self,
        client: OpenAI,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        """调用 LLM（带重试 + 超时），返回原始文本"""
        logger.debug("LLM 调用开始 | model=%s | temp=%.2f", self.DEFAULT_MODEL, temperature)
        response = client.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            timeout=self.DEFAULT_TIMEOUT,
        )
        return response.choices[0].message.content

    # ── 主流程 ──

    def execute(self, input_data: QueryRewriteInput) -> Dict[str, Any]:
        """执行查询改写"""
        api_key = input_data.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY：请在入参传入或设置环境变量")

        history_window = input_data.history_window or self.DEFAULT_HISTORY_WINDOW
        temperature = (
            input_data.temperature
            if input_data.temperature is not None
            else self.DEFAULT_TEMPERATURE
        )

        # ── 1. 必要性判断 ──
        if not self._check_need_rewrite(input_data.original_query, input_data.history):
            logger.info(
                "跳过改写 | query='%s' | history_rounds=%d",
                input_data.original_query[:50],
                len(input_data.history),
            )
            return {
                "rewritten_queries": [input_data.original_query],
                "is_rewritten": False,
            }

        # ── 2. 准备调用 ──
        client = self._get_client(api_key)
        messages = self._build_prompt(
            query=input_data.original_query,
            history=input_data.history,
            max_splits=input_data.max_sub_queries,
            history_window=history_window,
        )

        # ── 3. LLM 调用（带重试）──
        try:
            content = self._call_llm(
                client=client,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:
            # 4xx 等不可重试异常 → 不应静默吞掉
            logger.error(
                "QueryRewrite LLM 调用失败（不可重试） | error=%s | query='%s'",
                exc,
                input_data.original_query[:50],
            )
            # 仍然走兜底，但用 error 级别记录
            return {
                "rewritten_queries": [input_data.original_query],
                "is_rewritten": False,
            }

        # ── 4. 解析输出 ──
        queries = self._parse_llm_output(
            content=content,
            max_queries=input_data.max_sub_queries,
            fallback=input_data.original_query,
        )

        logger.info(
            "改写完成 | original='%s' → %d queries=%s",
            input_data.original_query[:50],
            len(queries),
            queries,
        )

        result = {
            "rewritten_queries": queries,
            "is_rewritten": (queries != [input_data.original_query]),
        }
        return self.validate_output(result)