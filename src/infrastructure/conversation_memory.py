"""
对话记忆模块 - 长对话摘要 + 向量化存储 + 检索

设计：当会话消息超过阈值时，生成摘要并存储为向量，
下次检索时同时搜索知识库和历史摘要，解决长期对话上下文丢失问题。

使用方式：
    from src.infrastructure.conversation_memory import ConversationMemory
    memory = ConversationMemory()
    memory.maybe_summarize(session_id, history, llm_client, embedder)
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("conversation_memory")

# 摘要阈值：消息数超过此值触发
SUMMARY_THRESHOLD = 20

# 每个会话最多保留的摘要数量
MAX_SUMMARIES = 5

# 摘要检索时的权重
SUMMARY_SEARCH_WEIGHT = 0.3

_SUMMARY_PROMPT = """请用 2-3 句话总结以下对话的核心主题、用户关注的技术方向、已解决的问题。

对话历史：
{history}

要求：
- 只输出总结文本，不要加任何前缀或解释
- 包含具体的技术关键词（如框架名、库名、方法名）
- 如果对话中用户表达了偏好（如"喜欢简洁回答"），也要包含"""


class SummaryRecord:
    """单条摘要记录"""

    def __init__(self, content: str, timestamp: datetime):
        self.content = content
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {"content": self.content, "timestamp": self.timestamp.isoformat()}


class ConversationMemory:
    """
    对话记忆管理器。

    - 消息 >= 阈值 → 调 LLM 生成摘要 → Embedding → 存 ChromaDB
    - 检索时多搜一个 summary 集合
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        threshold: int = SUMMARY_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.threshold = threshold
        self._last_summary_at: Dict[str, int] = {}  # session_id → 上次摘要时的消息数
        self._lock = threading.Lock()

    def _summary_collection_name(self, session_id: str) -> str:
        return f"chat_summary_{session_id}"

    def should_summarize(self, session_id: str, msg_count: int) -> bool:
        """判断是否该生成摘要了"""
        if msg_count < self.threshold:
            return False
        last = self._last_summary_at.get(session_id, 0)
        return (msg_count - last) >= self.threshold

    def maybe_summarize(
        self,
        session_id: str,
        history: List[Any],
        llm_client: Any,
        embedder: Any,
        session_manager: Any = None,
    ) -> Optional[str]:
        """
        若满足条件，生成摘要并存入向量库。

        Args:
            session_id: 会话 ID
            history: 消息列表（Message 对象或 dict）
            llm_client: LLM 客户端（需有 .chat() 方法）
            embedder: 嵌入客户端（需有 .execute() 方法）
            session_manager: SessionManager 实例（用于更新 metadata）

        Returns:
            摘要文本，或不满足条件时返回 None
        """
        with self._lock:
            msg_count = len(history)
            if not self.should_summarize(session_id, msg_count):
                return None

            logger.info(
                "触发对话摘要 | session=%s | messages=%d",
                session_id[:12], msg_count,
            )

            try:
                summary = self._generate_summary(history, llm_client)
                if not summary:
                    return None

                # 存到 ChromaDB
                if self.vector_store and embedder:
                    self._store_summary(session_id, summary, embedder)

                # 更新 session metadata
                if session_manager:
                    self._update_session_metadata(
                        session_id, summary, session_manager
                    )

                self._last_summary_at[session_id] = msg_count
                logger.info(
                    "摘要已生成并存储 | session=%s | len=%d",
                    session_id[:12], len(summary),
                )
                return summary

            except Exception as e:
                logger.error("摘要生成失败 | session=%s | %s", session_id[:12], e)
                return None

    def _generate_summary(self, history: List[Any], llm_client: Any) -> str:
        """调 LLM 生成摘要"""
        history_text = "\n".join(
            f"{self._get_role(h)}: {self._get_content(h)}"
            for h in history[-30:]  # 只取最近 30 条
        )

        prompt = _SUMMARY_PROMPT.format(history=history_text)

        try:
            response = llm_client.chat(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            # 兼容多种返回格式
            if hasattr(response, "content"):
                return response.content.strip()
            if isinstance(response, dict):
                return response.get("content", "").strip()
            return str(response).strip()
        except Exception as e:
            logger.warning("LLM 摘要失败，使用兜底: %s", e)
            return self._fallback_summary(history)

    def _fallback_summary(self, history: List[Any]) -> str:
        """降级：从历史中提取关键词兜底"""
        roles_keywords = {}
        for h in history[-20:]:
            role = self._get_role(h)
            content = self._get_content(h)
            if role == "user" and len(content) > 10:
                roles_keywords.setdefault("user_topics", [])
                # 简单取前 60 字符
                roles_keywords["user_topics"].append(content[:60])

        topics = roles_keywords.get("user_topics", [])
        if topics:
            return "用户关注的话题：" + "；".join(topics[-5:])
        return "对话摘要（自动生成）"

    def _store_summary(
        self, session_id: str, summary: str, embedder: Any
    ) -> None:
        """将摘要转向量并存入 ChromaDB"""
        try:
            # 调 embedder 转向量
            emb_result = embedder.execute(
                texts=[summary]
            ) if hasattr(embedder, "execute") else None

            if not emb_result:
                return

            vec = None
            if isinstance(emb_result, dict):
                chunks = emb_result.get("embedded_chunks", [])
                if chunks:
                    vec = (
                        chunks[0].get("embedding")
                        if isinstance(chunks[0], dict)
                        else getattr(chunks[0], "embedding", None)
                    )
            elif hasattr(emb_result, "embedded_chunks"):
                chunks = emb_result.embedded_chunks
                if chunks:
                    vec = getattr(chunks[0], "embedding", None)

            if not vec:
                logger.warning("嵌入向量为空，跳过存储")
                return

            collection_name = self._summary_collection_name(session_id)
            collection = self.vector_store.get_or_create_collection(collection_name)

            ts = datetime.now(timezone.utc).isoformat()
            doc_id = f"summary_{ts}"

            # 限制摘要数量
            existing = collection.get()
            if existing and len(existing.get("ids", [])) >= MAX_SUMMARIES:
                # 删除最旧的那个
                oldest_id = existing["ids"][0]
                collection.delete(ids=[oldest_id])

            collection.add(
                embeddings=[vec],
                documents=[summary],
                metadatas=[{"timestamp": ts, "session_id": session_id}],
                ids=[doc_id],
            )
            logger.debug("摘要已存入 ChromaDB | collection=%s", collection_name)

        except Exception as e:
            logger.warning("摘要向量存储失败: %s", e)

    def _update_session_metadata(
        self, session_id: str, summary: str, session_manager: Any
    ) -> None:
        """更新会话的 metadata 字段"""
        try:
            session = session_manager.get_session(session_id)
            if session and hasattr(session, "metadata"):
                existing = session.metadata.get("conversation_summaries", []) if session.metadata else []
                existing.append({
                    "content": summary,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                if len(existing) > MAX_SUMMARIES:
                    existing = existing[-MAX_SUMMARIES:]
                session_manager.update_session(
                    session_id,
                    metadata={"conversation_summaries": existing,
                              **(session.metadata or {})},
                )
        except Exception as e:
            logger.debug("更新 session metadata 失败: %s", e)

    def search_summaries(
        self, session_id: str, query_vector: List[float], top_k: int = 3
    ) -> List[str]:
        """
        从会话摘要集合中检索相关摘要。

        Args:
            session_id: 会话 ID
            query_vector: 查询向量
            top_k: 返回数量

        Returns:
            相关摘要文本列表
        """
        try:
            if not self.vector_store:
                return []

            collection_name = self._summary_collection_name(session_id)
            collection = self.vector_store.get_or_create_collection(collection_name)

            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, 5),
            )

            if results and results.get("documents"):
                docs = results["documents"]
                if docs and docs[0]:
                    return [str(d) for d in docs[0]]

            return []

        except Exception as e:
            logger.debug("摘要检索失败 | session=%s | %s", session_id[:12], e)
            return []

    @staticmethod
    def _get_role(msg: Any) -> str:
        if isinstance(msg, dict):
            return msg.get("role", "unknown")
        return getattr(msg, "role", "unknown")

    @staticmethod
    def _get_content(msg: Any) -> str:
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")


# 全局单例（可选）
_default_memory: Optional[ConversationMemory] = None
_memory_lock = threading.Lock()


def get_conversation_memory(
    vector_store: Any = None,
) -> ConversationMemory:
    global _default_memory
    with _memory_lock:
        if _default_memory is None:
            _default_memory = ConversationMemory(vector_store=vector_store)
        elif vector_store and _default_memory.vector_store is None:
            _default_memory.vector_store = vector_store
        return _default_memory
