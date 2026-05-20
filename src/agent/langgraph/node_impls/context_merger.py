"""
src/agent/langgraph/node_impls/context_merger.py

ContextMergerNode - 纯程序节点，合并规则前缀、引用消息、检索结果和用户消息
"""

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

from src.agent.langgraph.state import GraphState
from src.infrastructure.session_manager import SessionManager

logger = logging.getLogger("langgraph.node_impls.context_merger")


# ============================================================================
# 类型定义
# ============================================================================

class ReferencedMessage(TypedDict):
    """被 @msg-{id} 引用的消息"""
    msg_id: str
    content: str


class RetrievalSource(TypedDict):
    """检索来源信息"""
    rank: int
    doc_name: str
    chunk_id: str
    score: float


# ============================================================================
# 上下文合并节点
# ============================================================================

class ContextMergerNode:
    """
    上下文合并节点（不调用 LLM）

    职责：
    1. 解析用户消息中的 @msg-{id} 引用标记
    2. 从 session.message_history 中提取被引用消息
    3. 按顺序拼接：[System] 规则前缀 + [引用消息] + [文档] 检索结果 + [用户] 用户消息
    """

    # 引用消息正则：@msg-{id}，其中 id 由字母、数字、连字符组成
    REFERENCE_PATTERN = re.compile(r'@msg-([a-zA-Z0-9-]+)')

    # 上下文段落标记（可配置）
    SECTION_SYSTEM = "[System]"
    SECTION_SUMMARY = "[对话摘要]"          # ← 新增
    SECTION_REFERENCES = "[引用消息]"
    SECTION_DOCUMENTS = "[文档]"
    SECTION_USER = "[用户]"

    # 截断配置
    DEFAULT_DOC_TRUNCATE_CHARS = 200
    DEFAULT_REFERENCE_TRUNCATE_CHARS = 500
    TRUNCATE_SUFFIX = "..."

    # 默认回退值
    DEFAULT_DOC_SOURCE = "unknown"
    DEFAULT_CHUNK_ID = "unknown"
    DEFAULT_SCORE = 0.0

    def __init__(
        self,
        session_manager: Optional["SessionManager"] = None,
        doc_truncate_chars: int = DEFAULT_DOC_TRUNCATE_CHARS,
        reference_truncate_chars: int = DEFAULT_REFERENCE_TRUNCATE_CHARS,
    ):
        """
        Args:
            session_manager: 会话管理器实例，用于获取消息历史。
                             为 None 时引用解析功能自动降级（不报错）。
            doc_truncate_chars: 检索文档片段截断长度（字符数）
            reference_truncate_chars: 引用消息截断长度（字符数）
        """
        self.session_manager = session_manager
        self.doc_truncate_chars = doc_truncate_chars
        self.reference_truncate_chars = reference_truncate_chars

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: GraphState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """
        合并上下文

        Args:
            state: 当前图状态
            config: LangGraph 运行时配置

        Returns:
            {"merged_context": str, "retrieval_sources": List[RetrievalSource]}
        """
        try:
            user_message = state.get("current_input", "")
            system_prefix = state.get("system_prefix", "")

            # ✅ 修复：多来源读取重排序结果（reranked_results / reranked_docs / current_output）
            reranked_results = state.get("reranked_results")
            if reranked_results is None:
                reranked_results = state.get("reranked_docs")  # RerankSkill 实际输出字段名
            if reranked_results is None:
                current_output = state.get("current_output")
                if isinstance(current_output, dict):
                    reranked_results = current_output.get("reranked_results") or current_output.get("reranked_docs")

            reranked_results = self._normalize_results(reranked_results)

            session_id = state.get("session_id")

            if not user_message:
                logger.warning("user_message 为空，跳过上下文合并")
                return {"merged_context": "", "retrieval_sources": []}

            # 解析引用消息（失败时降级为空列表，不阻断流程）
            referenced_messages = self._resolve_references(user_message, session_id)

            # 构建检索来源
            retrieval_sources = self._build_retrieval_sources(reranked_results)

            # 拼接上下文（含对话摘要）
            merged_context = self._merge_context(
                system_prefix=system_prefix,
                referenced_messages=referenced_messages,
                reranked_results=reranked_results,
                user_message=user_message,
                session_id=session_id,
                state=state,
            )

            logger.info(
                "ContextMergerNode completed | session_id=%s | refs=%d | docs=%d | context_len=%d",
                session_id, len(referenced_messages), len(reranked_results),
                len(merged_context),
            )

            return {
                "merged_context": merged_context,
                "retrieval_sources": retrieval_sources,
            }

        except Exception as e:
            logger.error(
                "ContextMergerNode 未预期异常 | session_id=%s | error=%s",
                state.get("session_id"), e, exc_info=True,
            )
            return {
                "merged_context": state.get("current_input", ""),
                "retrieval_sources": [],
            }

    # ------------------------------------------------------------------
    # 引用解析
    # ------------------------------------------------------------------

    def _resolve_references(
        self,
        message: str,
        session_id: Optional[str],
    ) -> List[ReferencedMessage]:
        """
        解析消息中的 @msg-{id} 引用，从会话历史中查找对应消息

        所有异常均在内部降级处理，返回空列表，不向上抛出。
        遵循项目"容错优先"原则。

        Args:
            message: 用户消息文本
            session_id: 会话 ID

        Returns:
            [{"msg_id": "...", "content": "..."}, ...]
        """
        if not session_id:
            return []

        msg_ids = self.REFERENCE_PATTERN.findall(message)
        if not msg_ids:
            return []

        # session_manager 未注入但用户使用了 @msg 引用 → 告警
        if not self.session_manager:
            logger.warning(
                "消息中包含 @msg 引用但 session_manager 未注入，"
                "引用解析将跳过 | session_id=%s | refs=%s",
                session_id, msg_ids,
            )
            return []

        try:
            history = self.session_manager.get_message_history(session_id)
        except Exception as e:
            logger.error(
                "获取会话 %s 的消息历史失败: %s",
                session_id, e, exc_info=True,
            )
            return []

        if not history:
            logger.debug("会话 %s 无消息历史，跳过引用解析", session_id)
            return []

        # 构建 msg_id → message 映射
        history_map: Dict[str, Any] = {}
        for msg in history:
            msg_id = self._get_attr(msg, "message_id")
            if msg_id:
                history_map[msg_id] = msg

        # 查找引用的消息
        references: List[ReferencedMessage] = []
        for msg_id in msg_ids:
            if msg_id in history_map:
                msg = history_map[msg_id]
                content = self._get_attr(msg, "content", str(msg))
                references.append({
                    "msg_id": msg_id,
                    "content": content,
                })
                logger.debug("成功解析引用消息: @msg-%s", msg_id)
            else:
                logger.warning(
                    "未找到引用消息: @msg-%s | session_id=%s",
                    msg_id, session_id,
                )

        return references

    # ------------------------------------------------------------------
    # 检索来源构建
    # ------------------------------------------------------------------

    def _build_retrieval_sources(
        self,
        reranked_results: List[Any],
    ) -> List[RetrievalSource]:
        """
        从重排序结果中提取检索来源信息

        Args:
            reranked_results: 重排序后的文档列表

        Returns:
            [{"rank": int, "doc_name": str, "chunk_id": str, "score": float}, ...]
        """
        sources: List[RetrievalSource] = []

        for idx, doc in enumerate(reranked_results):
            doc_name = self._get_attr(doc, "source", self.DEFAULT_DOC_SOURCE)
            chunk_id = self._get_attr(doc, "doc_id", self.DEFAULT_CHUNK_ID)
            score = self._get_attr(doc, "rerank_score",
                     self._get_attr(doc, "score", self.DEFAULT_SCORE))

            # 当 fallback 触发时记录告警，方便追踪上游数据质量问题
            if doc_name == self.DEFAULT_DOC_SOURCE and chunk_id == self.DEFAULT_CHUNK_ID:
                logger.warning(
                    "检索文档缺少 source 和 doc_id，使用默认值 | rank=%d",
                    idx + 1,
                )

            sources.append({
                "rank": idx + 1,
                "doc_name": doc_name,
                "chunk_id": chunk_id,
                "score": float(score),
            })

        return sources

    # ------------------------------------------------------------------
    # 上下文拼接
    # ------------------------------------------------------------------

    def _merge_context(
        self,
        system_prefix: str,
        referenced_messages: List[ReferencedMessage],
        reranked_results: List[Any],
        user_message: str,
        session_id: Optional[str] = None,
        state: Optional[GraphState] = None,
    ) -> str:
        """
        按顺序拼接完整上下文

        格式：
        [System]
        规则前缀

        [对话摘要]
        近期对话要点...

        [引用消息]
        @msg-xxx: 内容

        [文档]
        1. 文档名 (score: 0.95): 片段内容

        [用户]
        用户消息（已去除 @msg 引用标记）
        """
        parts: List[str] = []

        # 1. System 规则前缀
        system_part = self._format_system_prefix(system_prefix)
        if system_part:
            parts.append(system_part)

        # ✅ 新增：2. 对话历史摘要
        summary_part = self._format_conversation_summary(session_id, state)
        if summary_part:
            parts.append(summary_part)

        # 3. 引用消息
        refs_part = self._format_references(referenced_messages)
        if refs_part:
            parts.append(refs_part)

        # 4. 检索文档
        docs_part = self._format_documents(reranked_results)
        if docs_part:
            parts.append(docs_part)

        # 5. 用户消息（去除引用标记）
        user_part = self._format_user_message(
            user_message,
            has_references=bool(referenced_messages),
        )
        parts.append(user_part)

        return "\n\n".join(parts)

    def _format_system_prefix(self, prefix: str) -> str:
        """格式化系统规则前缀段落。"""
        if not prefix:
            return ""
        return f"{self.SECTION_SYSTEM}\n{prefix}"

    # ✅ 新增：对话摘要格式化
    def _format_conversation_summary(
        self, session_id: Optional[str], state: Optional[GraphState]
    ) -> str:
        """从 session metadata 或向量库中提取对话摘要"""
        if not session_id:
            return ""
        try:
            # 优先从 state 中读取（LangGraph 图内传递的）
            if state:
                metadata = state.get("metadata", {})
                summaries = self._get_attr(
                    state.get("session", {}), "metadata", {}
                ).get("conversation_summaries", [])

            # 从 SessionManager 读取
            if self.session_manager:
                try:
                    session = self.session_manager.get_session(session_id)
                    if session:
                        summaries = (
                            self._get_attr(session, "metadata", {})
                            .get("conversation_summaries", [])
                        )
                except Exception:
                    summaries = []

            if not summaries:
                return ""

            # 取最近 3 条
            recent = summaries[-3:]
            lines = [self.SECTION_SUMMARY]
            for s in recent:
                content = s.get("content", "") if isinstance(s, dict) else str(s)
                if content:
                    lines.append(f"- {content}")
            return "\n".join(lines)

        except Exception as e:
            logger.debug("对话摘要格式化跳过: %s", e)
            return ""

    def _format_references(
        self,
        referenced_messages: List[ReferencedMessage],
    ) -> str:
        """格式化引用消息段落，对长消息做截断。"""
        if not referenced_messages:
            return ""

        lines = [self.SECTION_REFERENCES]
        for ref in referenced_messages:
            content = self._truncate_text(
                ref["content"],
                max_chars=self.reference_truncate_chars,
            )
            lines.append(f"@msg-{ref['msg_id']}: {content}")

        return "\n".join(lines)

    def _format_documents(self, reranked_results: List[Any]) -> str:
        """格式化检索文档段落，每个文档截断展示。"""
        if not reranked_results:
            return ""

        lines = [self.SECTION_DOCUMENTS]
        for idx, doc in enumerate(reranked_results, 1):
            content = self._get_attr(doc, "content", str(doc))
            source = self._get_attr(doc, "source", self.DEFAULT_DOC_SOURCE)
            score = self._get_attr(
                doc, "rerank_score",
                self._get_attr(doc, "score", self.DEFAULT_SCORE),
            )

            truncated = self._truncate_text(
                content,
                max_chars=self.doc_truncate_chars,
            )
            lines.append(
                f"{idx}. {source} (score: {float(score):.2f}): {truncated}"
            )

        return "\n".join(lines)

    def _format_user_message(
        self,
        raw_message: str,
        has_references: bool,
    ) -> str:
        """
        格式化用户消息段落。

        去除 @msg 引用标记；当用户消息全部由引用组成时，
        生成合理的默认意图，避免模型收到空用户消息。
        """
        clean = self.REFERENCE_PATTERN.sub('', raw_message).strip()

        if clean:
            return f"{self.SECTION_USER}\n{clean}"

        # 用户仅发送引用，无实质文本
        if has_references:
            return f"{self.SECTION_USER}\n请参考上述引用消息回答。"

        # 理论上不会到这里（user_message 为空时入口已跳过），防御性保留
        return f"{self.SECTION_USER}\n"

    # ------------------------------------------------------------------
    # 通用辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        """
        统一从 dict 或对象中安全获取属性。

        消除项目中反复出现的 `isinstance(doc, dict) ... else getattr(...)` 模式。
        """
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int = DEFAULT_DOC_TRUNCATE_CHARS,
    ) -> str:
        """
        智能截断文本，优先在句子边界断开。

        回退优先级：换行 → 中文句号 → 英文句号 → 空格 → 硬截断

        Args:
            text: 原始文本
            max_chars: 最大保留字符数

        Returns:
            截断后的文本；若实际发生截断则追加 TRUNCATE_SUFFIX
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # 按优先级寻找断点，要求断点不低于 max_chars * 60%
        min_pos = int(max_chars * 0.6)
        for sep in ("\n", "。", ".", " "):
            pos = truncated.rfind(sep)
            if pos >= min_pos:
                return truncated[:pos] + ContextMergerNode.TRUNCATE_SUFFIX

        return truncated + ContextMergerNode.TRUNCATE_SUFFIX

    @staticmethod
    def _normalize_results(results: Any) -> List[Any]:
        """
        规范化检索结果，确保始终返回列表。

        处理 None、空列表、falsy 值的边界情况。
        """
        if results is None:
            return []
        if isinstance(results, list):
            return results
        # 异常类型：记录并降级
        logger.warning(
            "reranked_results 类型异常，期望 list 实际 %s，已降级为空列表",
            type(results).__name__,
        )
        return []