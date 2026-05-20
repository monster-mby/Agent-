"""
src/agent/langgraph/node_impls.py — v3.0 重构版

核心改动：
- BaseNode 基类封装所有横切关注点（计时/记录/错误/日志）
- 具体节点只需声明 skill_name + 可选覆盖输入适配/输出提取
- SkillManager 通过依赖注入传入
- 结构化日志 + 取消检查 + Token 追踪
- ✅ 支持 RunnableConfig 传递配置
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

from src.agent.langgraph.state import (
    GraphState,
    SkillExecutionRecord,
    StepMetadata,
    StateError,
    TokenUsage,
    Critique,
    CritiquePoint,
    Feedback,
    FeedbackSource,
    ReflectionContext,
)
from src.skills.base.skill_manager import SkillManager


from src.skills.custom.rag_skills.query_rewrite_skill.skill import (
    QueryRewriteInput, QueryHistoryItem
)
logger = logging.getLogger("langgraph.node_impls")



# ═══════════════════════════════════════════════════════════════
# 通用节点基类
# ═══════════════════════════════════════════════════════════════

class BaseNode:
    """
    所有 LangGraph 节点的基类。

    封装横切关注点：
      - 计时（time.monotonic）
      - run_id 生成（uuid 防碰撞）
      - 取消检查
      - 空输入守卫
      - SkillExecutionRecord / StepMetadata 构建
      - 结构化错误返回
      - Token 用量提取
      - ✅ RunnableConfig 配置传递

    子类只需：
      1. 设置 skill_name 类属性
      2. 可选覆盖 adapt_input() / extract_output()
      3. 实现 _execute_impl()（如果需要非标准逻辑）
    """

    skill_name: str = "__base__"          # 子类必须覆盖
    input_schema: type[BaseModel] = None  # 可选，用于自动输入适配
    output_schema: type[BaseModel] = None # 可选，用于自动输出提取
    skill_manager: SkillManager = None    # 依赖注入

    def __init__(self, skill_manager: Optional[SkillManager] = None, config: Optional[Dict[str, Any]] = None):
        if skill_manager:
            self.skill_manager = skill_manager
        self._config = config or {}  # ← 构造函数注入的配置

    # ── 核心流程（子类不应覆盖）──────────────────────────

    def __call__(self, state: GraphState, config: Optional[RunnableConfig] = None) -> dict:
        """
        LangGraph 节点入口。

        Args:
            state: 当前 GraphState
            config: LangGraph RunnableConfig（包含 configurable 不可变配置）

        Returns:
            状态更新 dict（LangGraph 自动合并到 State）
        """
        run_id = self._generate_run_id()
        start_time = time.monotonic()

        # ✅ 新增：设置 skill_name 到 ContextVar（供日志系统使用）
        from src.infrastructure.logger import set_skill_name
        set_skill_name(self.skill_name)

        # ── 取消检查 ──
        if state.get("cancelled"):
            logger.info(
                "node cancelled",
                extra={"run_id": run_id, "skill": self.skill_name, "reason": state.get("cancel_reason")},
            )
            return {}

        # ── 空输入守卫 ──
        raw_input = state.get("current_input")
        if raw_input is None:
            logger.warning(
                "node skipped: no input",
                extra={"run_id": run_id, "skill": self.skill_name},
            )
            return {}

        try:
            logger.info(
                "node started",
                extra={"run_id": run_id, "skill": self.skill_name},
            )

            # 获取技能
            skill = self._get_skill()

            # ✅ 输入适配（传递 config）
            adapted_input = self.adapt_input(raw_input, skill, config)

            # 执行技能
            result = skill.execute(adapted_input)

            # 输出提取
            output = self.extract_output(result, skill)

            # Token 提取
            token_usage = self._extract_token_usage(result)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            # ✅ 确保最小耗时为 1ms（避免 Mock 或极速执行时为 0）
            elapsed_ms = max(1, elapsed_ms)

            logger.info(
                "node completed",
                extra={
                    "run_id": run_id,
                    "skill": self.skill_name,
                    "elapsed_ms": elapsed_ms,
                    "tokens": token_usage.model_dump() if token_usage else None,
                },
            )

            # logger.info(
            #     "node completed",
            #     extra={
            #         "run_id": run_id,
            #         "skill": self.skill_name,
            #         "elapsed_ms": elapsed_ms,
            #         "tokens": token_usage.model_dump() if token_usage else None,
            #     },
            # )

            return self._build_success_return(
                run_id=run_id,
                output=output,
                raw_input=raw_input,
                elapsed_ms=elapsed_ms,
                start_time=start_time,
                token_usage=token_usage,
            )


        except self._CATCHABLE_EXCEPTIONS as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            elapsed_ms = max(1, elapsed_ms)  # ✅ 确保最小 1ms
            # logger.error(
            #     "node failed",
            #     extra={
            #         "run_id": run_id,
            #         "skill": self.skill_name,
            #         "elapsed_ms": elapsed_ms,
            #         "error_type": type(e).__name__,
            #         "error": str(e),
            #     },
            #     exc_info=True,  # ← 修复：添加异常堆栈信息
            # )

            logger.error(
                "node failed | skill=%s | error=%s: %s | elapsed_ms=%d",
                self.skill_name, type(e).__name__, str(e), elapsed_ms,
                exc_info=True,
            )

            return self._build_error_return(
                run_id=run_id,
                exception=e,
                raw_input=raw_input,
                elapsed_ms=elapsed_ms,
                start_time=start_time,

            )

    # ── 子类可覆盖的方法 ──────────────────────────────────

    def adapt_input(self, raw_input: Any, skill, config: Optional[RunnableConfig] = None) -> Any:
        """
        输入适配：将 state["current_input"] 转换为 skill.input_schema 所需的格式。

        子类可以覆盖以处理特殊转换逻辑。

        ✅ 新增 config 参数，支持从 RunnableConfig 读取运行时配置

        默认行为：
          - 如果 raw_input 已经是 input_schema 的实例 → 直接返回
          - 如果是 str → 自动检测 schema 的第一个字段名并传入
          - 如果是 dict → 用 input_schema(**raw_input) 构建
        """
        schema = getattr(skill, "input_schema", None)
        if schema is None:
            return raw_input

        # ✅ 新增：如果 raw_input 是只含一个 'text' 键的 dict，解包为字符串
        if isinstance(raw_input, dict) and set(raw_input.keys()) == {"text"}:
            raw_input = raw_input["text"]

        if isinstance(raw_input, schema):
            return raw_input

        # ✅ 修复：如果是字符串，需要根据 schema 的第一个字段名来构造
        if isinstance(raw_input, str):
            # 获取 schema 的第一个字段名
            first_field_name = self._get_first_field_name(schema)
            if first_field_name:
                return schema(**{first_field_name: raw_input})
            else:
                # 兜底：尝试 text 字段
                try:
                    return schema(text=raw_input)
                except Exception:
                    logger.warning(
                        "cannot adapt string input to schema %s",
                        schema.__name__,
                    )
                    return raw_input

        if isinstance(raw_input, dict):
            return schema(**raw_input)

        # 无法自动适配，信任 skill.execute 能处理
        logger.warning(
            "cannot auto-adapt input",
            extra={"raw_type": str(type(raw_input)), "target_schema": schema.__name__},
        )
        return raw_input

    @staticmethod
    def _get_first_field_name(schema: type[BaseModel]) -> Optional[str]:
        """
        获取 Pydantic schema 的第一个字段名

        Args:
            schema: Pydantic BaseModel 子类

        Returns:
            第一个字段的名称，如果没有字段则返回 None
        """
        try:
            # Pydantic v2 使用 model_fields
            if hasattr(schema, 'model_fields'):
                fields = schema.model_fields
                if fields:
                    return list(fields.keys())[0]
            # Pydantic v1 兼容
            elif hasattr(schema, '__fields__'):
                fields = schema.__fields__
                if fields:
                    return list(fields.keys())[0]
        except Exception as e:
            logger.debug("Failed to get first field name from schema: %s", e)

        return None



    def extract_output(self, result: Any, skill) -> Any:
        """
        输出提取：从 skill.execute() 的结果中提取核心数据。

        默认行为：
          - 如果是 Pydantic BaseModel → model_dump()
          - 如果有 model_dump() 方法 → 调用它（兼容 FakeSkillOutput）
          - 否则直接返回
        """
        if isinstance(result, BaseModel):
            return result.model_dump()
        # ✅ 修复：兼容非 Pydantic 但实现了 model_dump() 的对象
        if hasattr(result, "model_dump") and callable(getattr(result, "model_dump")):
            return result.model_dump()
        return result

    # ── 内部方法（不应覆盖）───────────────────────────────

    # 可恢复异常：超时、网络错误等
    _RECOVERABLE_EXCEPTIONS = (ConnectionError, TimeoutError)

    # 所有捕获的异常（包括不可恢复的）
    _CATCHABLE_EXCEPTIONS = (
        ValueError, TypeError, KeyError,
        ConnectionError, TimeoutError,
        AttributeError, RuntimeError,  # ← 新增
    )

    def _generate_run_id(self) -> str:
        return f"{self.skill_name}_{uuid.uuid4().hex[:8]}"

    def _get_skill(self):
        """获取技能实例（优先依赖注入，回退到全局单例）"""
        sm = self.skill_manager or _get_default_skill_manager()
        skill_cls = sm.get(self.skill_name)
        if not skill_cls:
            raise ValueError(f"技能 '{self.skill_name}' 未注册")
        return skill_cls()

    def _extract_token_usage(self, result: Any) -> Optional[TokenUsage]:
        """
        从技能结果中提取 token 用量

        Note: 当前项目中的 Skill 尚未实现 token_usage 字段，
        此方法为未来 LLM 类技能预留扩展点。
        """
        if hasattr(result, "token_usage") and result.token_usage:
            tu = result.token_usage
            return TokenUsage(
                prompt_tokens=getattr(tu, "prompt_tokens", 0),
                completion_tokens=getattr(tu, "completion_tokens", 0),
                total_tokens=getattr(tu, "total_tokens", 0),
                model=getattr(tu, "model", "unknown"),
            )
        return None

    def _build_success_return(
            self,
            run_id: str,
            output: Any,
            raw_input: Any,
            elapsed_ms: int,
            start_time: float,
            token_usage: Optional[TokenUsage],
    ) -> dict:
        """构建成功时的状态更新 dict"""
        record = SkillExecutionRecord(
            run_id=run_id,
            skill_name=self.skill_name,
            input_summary=self._summarize(raw_input),
            output=output,
            elapsed_ms=elapsed_ms,
            success=True,
            timestamp=datetime.now(timezone.utc),
        )
        meta = StepMetadata(
            step_name=self.skill_name,
            elapsed_ms=elapsed_ms,
            retries=0,
            token_usage=token_usage,
            started_at=datetime.fromtimestamp(start_time, tz=timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

        # ✅ 修复：如果 output 是字典，直接合并到 state；否则保持原来逻辑
        if isinstance(output, dict):
            logger.warning(
                "_build_success_return: output is dict with keys=%s, merging to state",
                list(output.keys())
            )
            result = {
                **output,  # ← 合并字典中的所有键值对
                "skill_results": [record],
                "metadata": {self.skill_name: meta},
            }
            logger.warning(
                "_build_success_return: merged result keys=%s",
                list(result.keys())
            )

            # ✅ 新增：如果是 TextEmbedderNode，强制确保 query_vector 在 result 顶层
            if self.skill_name == "text_embedder" and "query_vector" in output:
                result["query_vector"] = output["query_vector"]
                result["embedded_query"] = output["query_vector"]
                logger.warning(
                    "_build_success_return: TextEmbedderNode forced query_vector into result, dim=%d",
                    len(output["query_vector"])
                )
        else:
            result = {
                "current_output": output,
                "skill_results": [record],
                "metadata": {self.skill_name: meta},
            }

        if token_usage:
            result["cumulative_token_usage"] = token_usage
        # 注意：不返回 error，避免覆盖前序错误
        return result

    def _build_error_return(
        self,
        run_id: str,
        exception: Exception,
        raw_input: Any,
        elapsed_ms: int,
        start_time: float,
    ) -> dict:
        """构建失败时的状态更新 dict"""
        # ✅ 动态判断是否可恢复
        recoverable = isinstance(exception, self._RECOVERABLE_EXCEPTIONS)

        state_error = StateError(
            code="SKILL_EXECUTION_ERROR",
            message=str(exception),
            step_name=self.skill_name,
            traceback=str(getattr(exception, "__traceback__", "")),
            recoverable=recoverable,
        )
        record = SkillExecutionRecord(
            run_id=run_id,
            skill_name=self.skill_name,
            input_summary=self._summarize(raw_input),
            output=None,
            elapsed_ms=elapsed_ms,
            success=False,
            error=state_error,
            timestamp=datetime.now(timezone.utc),
        )
        meta = StepMetadata(
            step_name=self.skill_name,
            elapsed_ms=elapsed_ms,
            retries=0,
            started_at=datetime.fromtimestamp(start_time, tz=timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

        return {
            "current_output": None,
            "skill_results": [record],
            "metadata": {self.skill_name: meta},
            "error": state_error,
        }

    @staticmethod
    def _summarize(data: Any, max_len: int = 120) -> str:
        """输入摘要（避免 state 膨胀）"""
        if data is None:
            return "None"
        s = str(data)
        return s[:max_len] + "..." if len(s) > max_len else s


# ═══════════════════════════════════════════════════════════════
# 模块级 SkillManager 单例
# ═══════════════════════════════════════════════════════════════

_default_skill_manager: Optional[SkillManager] = None


def _get_default_skill_manager() -> SkillManager:
    """获取默认的 SkillManager 单例"""
    global _default_skill_manager
    if _default_skill_manager is None:
        _default_skill_manager = SkillManager()
        logger.info("Default SkillManager initialized")
    return _default_skill_manager


class SummarizerNode(BaseNode):
    """摘要节点"""
    skill_name = "text_summarizer"

    def adapt_input(self, raw_input, skill, config: Optional[RunnableConfig] = None):
        """
        摘要特殊逻辑：将翻译结果作为输入

        ✅ 支持三种配置来源（优先级从高到低）：
        1. RunnableConfig.configurable（运行时传入）
        2. 构造函数注入 self._config
        3. 硬编码默认值
        """
        from src.skills.preset.content_creation.text_summarizer.skill import (
            TextSummarizerInput,
        )

        # 从 RunnableConfig 提取配置
        run_config = {}
        if config and hasattr(config, "get"):
            run_config = config.get("configurable", {})
        elif isinstance(config, dict):
            run_config = config.get("configurable", {})

        # 优先级：RunnableConfig > 构造函数注入 > 默认值
        max_length = (
                run_config.get("max_length") or
                self._config.get("max_length") or
                "medium"
        )
        style = (
                run_config.get("style") or
                self._config.get("style") or
                "paragraph"
        )

        if isinstance(raw_input, TextSummarizerInput):
            return raw_input
        if isinstance(raw_input, str):
            return TextSummarizerInput(
                text=raw_input,
                max_length=max_length,
                style=style,
            )
        if isinstance(raw_input, dict):
            return TextSummarizerInput(
                **{**raw_input, "max_length": max_length, "style": style}
            )
        return raw_input

    def extract_output(self, result, skill):
        """摘要输出：提取 summary 字段"""
        if hasattr(result, "summary"):
            return result.summary
        return super().extract_output(result, skill)


class TextEmbedderNode(BaseNode):
    """文本嵌入节点 - 特殊处理 List[EmbeddingCandidate] 输入"""
    skill_name = "text_embedder"

    def __call__(self, state: GraphState, config: Optional[RunnableConfig] = None) -> dict:
        """
        覆盖父类 __call__，确保 query_vector 被写入 state 顶层

        这是最可靠的做法，不受任何中间层影响
        """
        # 1. 调用父类，获取它原本要返回的结果（父类已经执行了 skill 和 extract_output）
        result = super().__call__(state, config)

        # 2. 父类返回的结果里，技能输出可能在 current_output 或直接就是字典
        inner = result.get("current_output", result)

        # 3. 如果 inner 是字典，且包含 query_vector，就把它提升到 state 顶层
        if isinstance(inner, dict) and "query_vector" in inner:
            result["query_vector"] = inner["query_vector"]
            # 同时保留一个备选键（其他节点可能读取 embedded_query）
            result["embedded_query"] = inner["query_vector"]
            logger.info("TextEmbedderNode: 已将 query_vector 提升到 state 顶层, dim=%d", len(inner["query_vector"]))

        return result

    def adapt_input(self, raw_input, skill, config: Optional[RunnableConfig] = None):
        """
        文本嵌入特殊逻辑：将字符串转换为 EmbeddingCandidate 列表

        注意：TextEmbedderInput.candidates 需要 List[EmbeddingCandidate]，不能直接传字符串
        """
        # ✅ 动态导入避免循环依赖
        from src.skills.custom.rag_skills.text_embedder.skill import (
            TextEmbedderInput,
            EmbeddingCandidate,
        )

        if isinstance(raw_input, TextEmbedderInput):
            return raw_input

        if isinstance(raw_input, str):
            try:
                # 将单个字符串包装成 EmbeddingCandidate 列表
                candidate = EmbeddingCandidate(text=raw_input, chunk_id=f"query_{hash(raw_input) % 10000}")
                return TextEmbedderInput(candidates=[candidate])
            except Exception as e:
                logger.error(
                    f"TextEmbedderNode adapt_input failed: {e}",
                    exc_info=True,
                )
                raise

        if isinstance(raw_input, dict):
            # 如果已经是 dict 格式，直接构造
            return TextEmbedderInput(**raw_input)

        # 无法适配
        logger.warning(
            "TextEmbedderNode: cannot adapt input type %s",
            type(raw_input).__name__,
        )
        return raw_input

    def extract_output(self, result, skill):
        """
        提取向量并写入 state["query_vector"]，供下游 VectorStoreRetrievalNode 使用

        Args:
            result: TextEmbedderSkill 返回的 dict，包含 embedded_chunks 列表

        Returns:
            dict: 同时更新 query_vector 和 current_output
        """
        # ✅ 新增：调试日志，确认方法被调用
        logger.warning("TextEmbedderNode.extract_output called, result type=%s", type(result).__name__)

        if not isinstance(result, dict):
            logger.warning("TextEmbedderNode: result is not dict, returning as-is")
            return {"current_output": result}

        # 从 embedded_chunks 中提取第一个向量的 embedding
        embedded_chunks = result.get("embedded_chunks", [])
        logger.warning("TextEmbedderNode: embedded_chunks count=%d", len(embedded_chunks))

        if embedded_chunks and len(embedded_chunks) > 0:
            first_chunk = embedded_chunks[0]
            if isinstance(first_chunk, dict):
                query_vector = first_chunk.get("embedding")
                logger.warning("TextEmbedderNode: first_chunk is dict, has embedding=%s", "embedding" in first_chunk)
            else:
                # EmbeddedChunk 对象
                query_vector = getattr(first_chunk, "embedding", None)
                logger.warning("TextEmbedderNode: first_chunk is object, has embedding=%s",
                               hasattr(first_chunk, "embedding"))

            if query_vector:
                logger.info(
                    "TextEmbedderNode: extracted query_vector, dim=%d",
                    len(query_vector)
                )
                output_dict = {
                    "query_vector": query_vector,  # ← 确保字段名正确
                    "current_output": result,  # 保留完整结果供调试
                }
                logger.warning("TextEmbedderNode.extract_output returning dict with keys: %s", list(output_dict.keys()))
                return output_dict

        logger.warning("TextEmbedderNode: no embedding found in result")
        logger.warning("TextEmbedderNode.extract_output 即将返回: %s", str(result)[:200])
        return {"current_output": result}

class RerankSkillNode(BaseNode):
    """重排序节点 - 特殊处理 query + documents 输入"""
    skill_name = "rerank_skill"

    def __call__(self, state: GraphState, config=None) -> dict:
        """覆盖父类入口：缓存完整 state 供 adapt_input 读取检索结果"""
        self._current_state = state
        try:
            return super().__call__(state, config)
        finally:
            self._current_state = None

    def adapt_input(self, raw_input, skill, config: Optional[RunnableConfig] = None):
        """
        重排序特殊逻辑：需要从 state 中获取 query 和 documents

        注意：RerankInput 需要 query (str) + candidates (List[RerankCandidate])
        """
        # ✅ 动态导入避免循环依赖
        from src.skills.custom.rag_skills.rerank_skill.skill import (
            RerankInput,
            RerankCandidate,
        )

        if isinstance(raw_input, RerankInput):
            return raw_input

        # ✅ 修复：从缓存的 state 中读取上游检索结果 (candidates)
        if isinstance(raw_input, str):
            query = raw_input
            candidates = []
            if hasattr(self, '_current_state') and self._current_state:
                raw_candidates = self._current_state.get("candidates", [])
                for c in raw_candidates:
                    try:
                        candidates.append(RerankCandidate(
                            content=c.get("text", ""),
                            doc_id=c.get("chunk_id", ""),
                            source=c.get("metadata", {}).get("source") if isinstance(c.get("metadata"), dict) else None,
                            original_score=c.get("score"),
                        ))
                    except Exception:
                        pass
            if candidates:
                logger.info("RerankSkillNode: 从 state.candidates 读取到 %d 条检索结果", len(candidates))
                return RerankInput(query=query, candidates=candidates)
            else:
                logger.warning("RerankSkillNode: state.candidates 为空，上游检索可能未返回结果")
                return RerankInput(query=query, candidates=[])

        if isinstance(raw_input, dict):
            try:
                return RerankInput(**raw_input)
            except Exception as e:
                logger.error(
                    f"RerankSkillNode adapt_input failed: {e}",
                    exc_info=True,
                )
                raise

        logger.warning(
            "RerankSkillNode: cannot adapt input type %s",
            type(raw_input).__name__,
        )
        return raw_input


class RagAnswerNode(BaseNode):
    """RAG 回答节点 - 特殊处理 query + search_results 输入"""
    skill_name = "rag_answer"

    def __init__(self, skill_manager=None, config=None, llm_client=None):
        super().__init__(skill_manager=skill_manager, config=config or {})
        self.llm_client = llm_client

    def __call__(self, state: GraphState, config=None) -> dict:
        """覆盖父类入口：缓存完整 state 供 adapt_input 读取检索结果"""
        # ✅ 从多个来源读取最终的检索结果
        search_results = (
            state.get("reranked_docs")
            or state.get("reranked_results")
            or state.get("candidates")
            or state.get("search_results")
        )
        if not search_results:
            logger.warning(
                "RagAnswerNode: 无检索结果，将交由 Skill 基于 LLM 自身知识回答"
            )
        self._current_state = state
        try:
            return super().__call__(state, config)
        finally:
            self._current_state = None

    def adapt_input(self, raw_input, skill, config=None):
        """RAG 回答特殊逻辑：需要从 state 中获取 query 和 search_results"""
        from src.skills.custom.rag_skills.rag_answer.skill import (
            RagAnswerInput,
            SearchResultRef,
        )

        if isinstance(raw_input, RagAnswerInput):
            return raw_input

        # ✅ 修复：从缓存的 state 中读取实际检索结果，统一字段名
        if isinstance(raw_input, str):
            query = raw_input
            raw_results = []
            if hasattr(self, '_current_state') and self._current_state:
                raw_results = (
                    self._current_state.get("reranked_docs")
                    or self._current_state.get("reranked_results")
                    or self._current_state.get("candidates")
                    or self._current_state.get("search_results")
                    or []
                )
            # ✅ 字段名标准化：兼容检索格式 {chunk_id, text} 和重排格式 {content, doc_id}
            results = []
            for item in (raw_results or []):
                if not isinstance(item, dict):
                    continue
                results.append(SearchResultRef(
                    chunk_id=item.get("chunk_id") or item.get("doc_id", ""),
                    text=item.get("text") or item.get("content", ""),
                    score=item.get("score") or item.get("original_score") or 0.0,
                    metadata=item.get("metadata", {}),
                ))
            if not results:
                logger.warning(
                    "RagAnswerNode: 检索结果为空，将基于 LLM 自身知识回答"
                )
            return RagAnswerInput(query=query, search_results=results, llm_client=self.llm_client)

        if isinstance(raw_input, dict):
            try:
                query = raw_input.get("current_input", "")
                search_results = raw_input.get("search_results", [])
                return RagAnswerInput(query=query, search_results=search_results)
            except Exception as e:
                logger.error(f"RagAnswerNode adapt_input failed: {e}", exc_info=True)
                raise

        logger.warning(
            "RagAnswerNode: cannot adapt input type %s", type(raw_input).__name__
        )
        return raw_input


def build_skill_node(skill_name: str, skill_manager: SkillManager) -> type[BaseNode]:
    """
    通用技能节点工厂函数

    返回一个配置好的 BaseNode 子类，调用方需自行实例化：

        NodeClass = build_skill_node("translator", skill_manager)
        node = NodeClass(config={"target_lang": "en"})

    Args:
        skill_name: 技能的唯一标识符（如 'translator', 'text_summarizer'）
        skill_manager: 技能管理器实例

    Returns:
        配置好的 BaseNode 子类（尚未实例化）

    Raises:
        ValueError: 技能在 skill_manager 中不存在
    """
    # 预检查：技能是否已注册
    if not skill_manager.get(skill_name):
        raise ValueError(f"Skill '{skill_name}' not found in manager")

    class DynamicSkillNode(BaseNode):
        """
        由 build_skill_node 动态生成的技能节点。
        每次调用 get_skill() 时从 skill_manager 重新获取技能，
        避免闭包绑定过期的实例。
        """

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            super().__init__(skill_manager=skill_manager, config=config or {})

        @property
        def skill_name(self) -> str:
            return skill_name

        def get_skill(self, config: Optional[RunnableConfig] = None):
            """
            延迟加载技能实例，每次调用都从 skill_manager 获取。
            如果技能实例化需要配置，从 config 中提取。
            """
            skill = skill_manager.get(skill_name)
            if not skill:
                raise ValueError(f"Skill '{skill_name}' not found in manager")

            # skill_manager.get() 返回的是技能类（需实例化）还是技能实例？
            # 两种情况都兼容：
            if callable(skill):
                # 返回的是类/工厂 → 实例化
                try:
                    return skill()
                except TypeError:
                    # 构造签名可能是 skill(config=...)
                    run_config = config.get("configurable", {}) if config else {}
                    return skill(config=run_config)
            else:
                # 返回的直接就是实例
                return skill

    # 设置稳定的类名，便于调试和日志
    DynamicSkillNode.__name__ = f"SkillNode_{skill_name}"
    DynamicSkillNode.__qualname__ = f"build_skill_node.<locals>.SkillNode_{skill_name}"

    return DynamicSkillNode
# ═══════════════════════════════════════════════════════════════
# 具体节点（只需声明 skill_name + 少量覆盖）
# ═══════════════════════════════════════════════════════════════

class TranslatorNode(BaseNode):
    """翻译节点"""
    skill_name = "translator"

    def adapt_input(self, raw_input, skill, config: Optional[RunnableConfig] = None):
        """
        翻译特殊逻辑：自动检测源语言，默认目标英文

        ✅ 支持三种配置来源（优先级从高到低）：
        1. RunnableConfig.configurable（运行时传入）
        2. 构造函数注入 self._config
        3. 硬编码默认值
        """
        from src.skills.preset.office_efficiency.translator.skill import TranslatorInput

        # 从 RunnableConfig 提取配置
        run_config = {}
        if config and hasattr(config, "get"):
            run_config = config.get("configurable", {})
        elif isinstance(config, dict):
            run_config = config.get("configurable", {})

        # 优先级：RunnableConfig > 构造函数注入 > 默认值
        target_lang = (
            run_config.get("target_lang") or
            self._config.get("target_lang") or
            "en"
        )
        source_lang = (
            run_config.get("source_lang") or
            self._config.get("source_lang") or
            "auto"
        )

        if isinstance(raw_input, TranslatorInput):
            return raw_input
        if isinstance(raw_input, str):
            return TranslatorInput(text=raw_input, source_lang=source_lang, target_lang=target_lang)
        if isinstance(raw_input, dict):
            return TranslatorInput(**{**raw_input, "source_lang": source_lang, "target_lang": target_lang})
        return raw_input

    def extract_output(self, result, skill):
        """翻译输出：提取 translated_text 字段"""
        if hasattr(result, "translated_text"):
            return result.translated_text
        return super().extract_output(result, skill)

"""
src/agent/langgraph/nodes/query_rewrite.py

QueryRewriteNode - 查询改写节点
适配 LangChain messages → QueryRewriteInput（QueryHistoryItem 列表）
"""




class QueryRewriteNode(BaseNode):
    """
    查询改写节点 —— 将 LangChain 消息历史转换为 QueryHistoryItem 列表，
    适配 QueryRewriteSkill 的输入契约。

    配置项（通过 config dict 传入）：
        max_sub_queries       : int  最大子查询数，默认 3（0 表示不拆分）
        history_message_count : int  纳入历史的最大消息条数，默认 6（≈3 轮）
        history_window        : int  历史窗口轮数，默认 3
    """

    skill_name = "query_rewrite"

    def __init__(self, skill_manager=None, config=None):
        super().__init__(skill_manager, config)
        self._current_state: Optional[GraphState] = None  # 显式初始化
    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState, config=None) -> dict:
        """缓存 state 供 adapt_input 使用，调用结束后自动清理"""
        self._current_state = state
        try:
            return super().__call__(state, config)
        finally:
            self._current_state = None  # 确保清理，防止状态残留和并发问题

    def adapt_input(
        self,
        raw_input: Union[str, Any],
        skill: Any,
        config: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> QueryRewriteInput:
        """
        将原始 state 数据转换为 QueryRewriteInput。

        Args:
            raw_input: 原始输入（字符串或其他可 str() 的类型）
            skill:     已解析的 QueryRewriteSkill 实例
            config:    节点配置，支持 dict 或 RunnableConfig

        Returns:
            QueryRewriteInput，可直接传入 skill.execute()
        """
        original_query = self._normalize_query(raw_input)

        # 配置
        max_sub_queries = self._get_config_value(config, "max_sub_queries", 3)
        history_window = self._get_config_value(config, "history_window", 3)
        history_message_count = self._get_config_value(
            config, "history_message_count", 6
        )

        # 消息 → 历史
        messages: List[Any] = self._resolve_messages()
        history = self._convert_messages_to_history(messages, history_message_count)

        return QueryRewriteInput(
            original_query=original_query,
            history=history,
            max_sub_queries=max_sub_queries,
            history_window=history_window,
        )

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_query(raw_input: Union[str, Any]) -> str:
        """规范化查询字符串，做非空校验"""
        if isinstance(raw_input, str):
            query = raw_input.strip()
        else:
            query = str(raw_input).strip()

        if not query:
            logger.warning("QueryRewriteNode: original_query is empty after normalization")
            query = ""  # 显式空字符串，不做静默替换
        return query

    def _resolve_messages(self) -> List[Any]:
        """
        从缓存的 state 中获取 messages 列表。
        如果 _current_state 为 None（例如节点被直接调用而非通过 LangGraph），返回空列表。
        """
        if self._current_state is None:
            logger.warning("QueryRewriteNode: _current_state is None, returning empty messages")
            return []

        messages = self._current_state.get("messages", [])
        if not messages:
            logger.debug("QueryRewriteNode: no messages found in state")
        return messages

    @staticmethod
    def _convert_messages_to_history(
        messages: List[Any],
        max_count: int,
    ) -> List[QueryHistoryItem]:
        """
        将 LangChain BaseMessage 列表转换为 QueryHistoryItem 列表。

        Args:
            messages:  LangChain 消息列表
            max_count: 截取最近 N 条消息

        Returns:
            QueryHistoryItem 列表（role + content）
        """
        if not messages:
            return []

        recent = messages[-max_count:] if len(messages) > max_count else messages
        history: List[QueryHistoryItem] = []

        for msg in recent:
            role = QueryRewriteNode._infer_role(msg)
            content = QueryRewriteNode._extract_content(msg)
            if content:  # 跳过空内容消息
                history.append(QueryHistoryItem(role=role, content=content))

        return history

    @staticmethod
    def _infer_role(msg: Any) -> str:
        """从 LangChain 消息类型推导角色标签"""
        msg_type = getattr(msg, "type", "")
        if msg_type == "human":
            return "user"
        if msg_type == "ai":
            return "assistant"
        if msg_type == "system":
            return "system"
        # 兜底：通过类名推断
        class_name = type(msg).__name__
        if "Human" in class_name:
            return "user"
        if "AI" in class_name:
            return "assistant"
        if "System" in class_name:
            return "system"
        logger.debug("QueryRewriteNode: unknown message type=%r, class=%s", msg_type, class_name)
        return "assistant"  # 安全兜底

    @staticmethod
    def _extract_content(msg: Any) -> str:
        """
        从 LangChain 消息中提取文本内容。
        兼容 content 为 str 或 list[dict]（多模态块）的情况。
        """
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        parts.append(str(text))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content).strip() if content else ""

    def _get_config_value(
        self,
        config: Optional[Union[Dict[str, Any], Any]],
        key: str,
        default: Any,
    ) -> Any:
        """
        按优先级获取配置值：run_config > self._config > default。

        使用 is not None 判断（而非 or），避免 0 / False 等假值被跳过。
        """
        # 1) 从运行时配置取
        run_config: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, dict):
                run_config = config.get("configurable", {}) or {}
            elif hasattr(config, "get"):
                run_config = getattr(config, "configurable", {}) or {}

        v = run_config.get(key)
        if v is not None:
            return v

        # 2) 从节点自身配置取
        node_config: Dict[str, Any] = getattr(self, "_config", {}) or {}
        v = node_config.get(key)
        if v is not None:
            return v

        # 3) 回退默认值
        return default





# ═══════════════════════════════════════════════════════════════
# 反思模式节点（Day 4 — 修复版）
# 追加到 node_impls.py 末尾。import 已全部在文件顶部。
# ═══════════════════════════════════════════════════════════════

# 确保文件顶部已有这些 import（若无则添加）：
# import json
# import re
# from src.agent.langgraph.state import (
#     GraphState, ReflectionContext,
#     Critique, CritiquePoint, Feedback, FeedbackSource,
# )

logger_reflection = logging.getLogger("reflection.node_impls")


# ═══════════════════════════════════════════════════════
# flake8 辅助函数
# ═══════════════════════════════════════════════════════

def _run_flake8(code: str) -> dict:
    """对代码字符串运行 flake8，返回结构化结果"""
    import subprocess
    import tempfile
    import os

    try:
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["flake8", "--format=%(row)d:%(col)d:%(code)s:%(text)s", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(tmp_path)

        issues = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(":", 3)
            if len(parts) >= 4:
                issues.append({
                    "line": int(parts[0]),
                    "col": int(parts[1]),
                    "code": parts[2].strip(),
                    "message": parts[3].strip(),
                })

        return {"issues": issues, "exit_code": result.returncode}

    except FileNotFoundError:
        return {"issues": [], "error": "flake8 未安装，跳过 lint 检查"}
    except Exception as e:
        return {"issues": [], "error": f"flake8 执行失败: {e}"}


# ═══════════════════════════════════════════════════════
# 反馈策略
# ═══════════════════════════════════════════════════════

def _rag_feedback(generation: str, state: GraphState, skill_manager) -> dict:
    """RAG 反馈：检查引用是否可追溯到检索结果"""
    search_results = state.get("search_results", [])

    if not search_results:
        return {
            "content": "RAG 反馈：无可用的检索结果，无法验证引用准确性",
            "checks": [{"name": "citation_verification", "passed": False, "detail": "无检索结果"}],
            "score": 0.5,
        }

    # 检查 generation 中是否引用了检索到的文档片段
    references_found = 0
    missing_references = []

    for result in search_results:
        snippet = str(result.get("text", str(result)))[:200]
        # 取前 50 字符作为特征在答案中查找
        if snippet[:50] in generation:
            references_found += 1
        else:
            missing_references.append(snippet[:80])

    score = references_found / len(search_results) if search_results else 0.5

    return {
        "content": (
                f"RAG 反馈：{references_found}/{len(search_results)} 条检索结果在答案中被引用。"
                + (f" 以下片段未被引用：{'；'.join(missing_references[:3])}" if missing_references
                   else " 所有检索结果均有引用。")
        ),
        "checks": [
            {"name": "citation_coverage", "passed": score >= 0.6,
             "detail": f"{references_found}/{len(search_results)}"},
        ],
        "score": score,
    }


def _code_feedback(generation: str, state: GraphState, skill_manager) -> dict:
    """
    代码反馈：对原始代码运行 flake8，检查审查结果是否覆盖了工具发现的问题
    """
    original_code = state.get("current_input", "")

    if not original_code or not isinstance(original_code, str):
        return {
            "content": "代码反馈：无原始代码可检测",
            "checks": [],
            "score": None,
        }

    lint_results = _run_flake8(original_code)

    if lint_results.get("error"):
        return {
            "content": f"代码反馈：{lint_results['error']}",
            "checks": [{"name": "flake8_lint", "passed": False, "detail": lint_results["error"]}],
            "score": None,
        }

    lint_issues = lint_results.get("issues", [])

    if not lint_issues:
        return {
            "content": "代码反馈：flake8 未发现任何问题 ",
            "checks": [{"name": "flake8_lint", "passed": True, "detail": "0 issues"}],
            "score": 1.0,
        }

    # 检查审查结果中是否覆盖了 lint 问题
    missed = []
    for issue in lint_issues:
        line_info = f"L{issue.get('line', '?')}"
        code_info = issue.get("code", "")
        if line_info not in generation and code_info not in generation:
            missed.append(
                f"{line_info} {code_info}: {issue.get('message', '')[:60]}"
            )

    coverage = 1.0 - (len(missed) / len(lint_issues)) if lint_issues else 1.0

    return {
        "content": (
                f"代码反馈：flake8 发现 {len(lint_issues)} 个问题。"
                f"审查覆盖了 {len(lint_issues) - len(missed)}/{len(lint_issues)} 个。"
                + (f"\n遗漏问题：\n" + "\n".join(missed[:5]) if missed
                   else "\n审查已覆盖所有 lint 问题。")
        ),
        "checks": [
            {"name": "flake8_issue_count", "passed": True, "detail": str(len(lint_issues))},
            {"name": "review_coverage", "passed": coverage >= 0.7, "detail": f"{coverage:.0%}"},
        ],
        "score": coverage,
    }


def _generic_feedback(generation: str, state: GraphState, skill_manager) -> dict:
    """基于实际内容质量的通用反馈（不再返回硬编码假数据）"""
    checks = []
    issues = []

    # 检查 1：是否为空
    if not generation or not generation.strip():
        checks.append({"name": "empty_check", "passed": False, "detail": "输出为空"})
        issues.append("生成结果为空")
    else:
        checks.append({"name": "empty_check", "passed": True})

    # 检查 2：长度是否过短（< 20 字符）
    if generation and len(generation.strip()) < 20:
        checks.append({"name": "length_check", "passed": False, "detail": f"仅 {len(generation.strip())} 字符"})
        issues.append("生成结果过短，可能未完整回答")
    else:
        checks.append({"name": "length_check", "passed": True})

    # 计算真实分数：全部通过 = 1.0，部分失败 = 按比例
    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    score = passed / total if total > 0 else 1.0

    content = "通用反馈：" + ("；".join(issues) if issues else "基础质量检查通过")

    return {"content": content, "checks": checks, "score": score}


FEEDBACK_STRATEGIES = {
    "rag_answer": _rag_feedback,
    "code_explainer": _code_feedback,  # ← 修复：改为正确的技能名称
    "default": _generic_feedback,
}



# ═══════════════════════════════════════════════════════
# Critic Prompt
# ═══════════════════════════════════════════════════════

CRITIC_SYSTEM = (
    "你是专业内容评审专家。分析给定文本和外部反馈，生成结构化批评报告。\n"
    "严格输出 JSON：{\"summary\":\"...\",\"points\":[{\"target_field\":\"...\","
    "\"severity\":\"critical|major|minor|suggestion\",\"description\":\"...\","
    "\"suggested_fix\":\"...\"}],\"overall_score\":0.85}"
)

CRITIC_USER = (
    "## 待评估文本\n{generation}\n\n"
    "## 外部反馈\n{feedback_content}\n\n"
    "请生成批评报告（纯 JSON）。"
)


# ═══════════════════════════════════════════════════════
# 四个反思节点
# ═══════════════════════════════════════════════════════

class GeneratorNode(BaseNode):
    """生成初始结果 → reflection_context.refined_output

    继承 BaseNode 复用计时/记录/错误处理。
    通过覆盖 __call__ 扩展：将输出写入 reflection_context。
    """

    def __init__(self, skill_manager: SkillManager, target_skill: str,
                 output_field: str = "answer", config: Optional[Dict[str, Any]] = None):
        super().__init__(skill_manager=skill_manager, config=config or {})
        self.target_skill = target_skill
        self.output_field = output_field

    @property
    def skill_name(self) -> str:
        return self.target_skill

    def __call__(self, state: GraphState, config=None) -> dict:
        import traceback
        try:
            # 调用父类标准流程（含计时/记录/错误封装）
            result = super().__call__(state, config)

            generation = result.get("current_output")
            if generation is None:
                return result

            # 提取文本
            if isinstance(generation, str):
                text = generation
            elif isinstance(generation, dict) and self.output_field in generation:
                text = str(generation[self.output_field])
            else:
                text = str(generation)

            # 更新 reflection_context
            ctx = state.get("reflection_context")
            if ctx is None:
                ctx = ReflectionContext()

            ctx.refined_output = text
            ctx.iteration = 0

            return {**result, "reflection_context": ctx}
        except Exception as e:
            logger.error("GeneratorNode 失败: %s", e)
            traceback.print_exc()
            raise



class ExternalFeedbackNode:
    """采集外部反馈 → reflection_context.feedback

    纯程序节点，不调用 LLM，不继承 BaseNode。
    """

    def __init__(self, target_skill: str = "default"):
        self.target_skill = target_skill

    def __call__(self, state: GraphState, config=None) -> dict:
        import traceback
        try:
            ctx = state.get("reflection_context")
            if ctx is None or ctx.refined_output is None:
                raise ValueError("reflection_context.refined_output 不可用")

            strategy = FEEDBACK_STRATEGIES.get(self.target_skill, _generic_feedback)
            fb_data = strategy(ctx.refined_output, state, None)

            feedback = Feedback(
                content=fb_data["content"],
                source=FeedbackSource(source_type="auto", source_id=f"strategy_{self.target_skill}"),
                score=fb_data.get("score"),
            )
            ctx.feedback = feedback
            return {"reflection_context": ctx}
        except Exception as e:
            logger.error("ExternalFeedbackNode 失败 (target_skill=%s): %s", self.target_skill, e)
            traceback.print_exc()
            raise


class CriticNode:
    """LLM 分析 → reflection_context.critique"""

    def __init__(self, llm_client, model: str = "gpt-4", temperature: float = 0.1):
        self.llm = llm_client
        self.model = model
        self.temperature = temperature

    def __call__(self, state: GraphState, config=None) -> dict:
        import traceback
        try:
            ctx = state.get("reflection_context")
            if ctx is None or ctx.refined_output is None or ctx.feedback is None:
                raise ValueError("reflection_context 缺少 refined_output 或 feedback")

            user_prompt = CRITIC_USER.format(
                generation=ctx.refined_output,
                feedback_content=ctx.feedback.content,
            )
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
            )

            raw = resp.content if hasattr(resp, "content") else str(resp)
            critique = self._parse(raw)
            critique.source_feedback_id = ctx.feedback.source.source_id
            ctx.add_critique(critique)
            ctx.status = "critiquing"
            return {"reflection_context": ctx}
        except Exception as e:
            logger.error("CriticNode 失败 (model=%s): %s", self.model, e)
            traceback.print_exc()
            raise

    def _parse(self, raw: str) -> Critique:
        try:
            return Critique.model_validate_json(raw)
        except Exception:
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                try:
                    return Critique.model_validate_json(m.group(0))
                except Exception:
                    pass
            return Critique(points=[], overall_score=0.5, summary=f"解析失败: {raw[:100]}")


class ReviserNode:
    """修订结果 → reflection_context.refined_output

    通过 skill_manager 懒加载 ReflectionSkill。
    """

    def __init__(self, skill_manager: SkillManager, config: Optional[Dict[str, Any]] = None):
        self.skill_manager = skill_manager
        self.config = config or {}
        self._reflection_skill = None

    def _get_skill(self):
        if self._reflection_skill is None:
            from src.skills.custom.reflection_skills.reflection.skill import ReflectionSkill
            self._reflection_skill = ReflectionSkill(
                llm_client=self.config.get("llm_client"),
                model=self.config.get("model", "gpt-4"),
                temperature=self.config.get("temperature", 0.5),
            )
        return self._reflection_skill

    def __call__(self, state: GraphState, config=None) -> dict:
        import traceback
        try:
            ctx = state.get("reflection_context")
            if ctx is None:
                raise ValueError("reflection_context 不可用")

            skill = self._get_skill()
            result = skill.execute_from_context(ctx)
            if not result.success:
                raise RuntimeError(f"修订失败: {result.error}")

            ctx.refined_output = result.revised_generation
            ctx.iteration += 1
            ctx.status = "revising"
            return {"reflection_context": ctx}
        except Exception as e:
            logger.error("ReviserNode 失败: %s", e)
            traceback.print_exc()
            raise


# ═══════════════════════════════════════════════════════
# RulesInjectorNode - 规则注入节点（第 8 阶段新增）
# ═══════════════════════════════════════════════════════

class RulesInjectorNode:
    """
    规则注入节点：从 RulesEngine 获取会话规则，注入到 state["system_prefix"]

    作为 LangGraph 图的第一个节点执行。
    """

    def __init__(self, rules_engine):
        """
        Args:
            rules_engine: RulesEngine 实例
        """
        self.rules_engine = rules_engine

    def __call__(self, state: GraphState, config=None) -> dict:
        session_id = state.get("session_id")
        if not session_id:
            logger.debug("No session_id in state, skipping rule injection")
            return {}

        system_prefix = self.rules_engine.build_system_prefix(session_id)

        if system_prefix:
            logger.info(
                "Injected rules for session %s: %d chars",
                session_id, len(system_prefix)
            )
        else:
            logger.debug("No enabled rules for session %s", session_id)

        return {"system_prefix": system_prefix}
