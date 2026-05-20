"""
SSE 流式输出封装（修正版）
- 修复所有 4 个已验证问题
- 对齐项目实际配置、依赖和导入路径
"""
from __future__ import annotations

import json
import logging
import time
from enum import StrEnum
from typing import Any, AsyncGenerator, Dict, Optional

from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from src.core.config import settings

logger = logging.getLogger(__name__)


class SSEEvent(StrEnum):
    NODE_OUTPUT = "node_output"
    ANSWER_CHUNK = "answer_chunk"
    STATUS = "status"          # ← 新增：中间状态推送
    DONE = "done"
    ERROR = "error"


def _safe_truncate(obj: Any, max_length: int = 500) -> Dict[str, Any]:
    """安全截断，附带 is_truncated 标记"""
    result: Dict[str, Any] = {}
    try:
        if isinstance(obj, str):
            is_truncated = len(obj) > max_length
            result["content"] = obj[:max_length] if is_truncated else obj
            result["full_length"] = len(obj)
        else:
            summary = json.dumps(obj, ensure_ascii=False, default=str)
            is_truncated = len(summary) > max_length
            result["content"] = summary[:max_length] if is_truncated else summary
            result["full_length"] = len(summary)
        result["is_truncated"] = is_truncated
    except Exception:
        result["content"] = "[无法序列化]"
        result["is_truncated"] = False
        result["full_length"] = 0
    return result


async def stream_chat_response(
        query: str,
        session_id: str,
        request_id: str,
        graph: CompiledStateGraph,
        config: dict,
        sm,
        kb_id: str = "default",  # ✅ 新增：知识库 ID
        last_event_id: Optional[str] = None,  # ✅ 新增：支持断点续传
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    流式输出聊天响应（SSE 格式），基于 LangGraph astream_events 实现 Token 级流式。

    Args:
        kb_id: 知识库 ID（用于向量检索定位正确的 collection）
        last_event_id: 客户端最后接收的事件ID（用于断线重连）
    """
    start_time = time.monotonic()
    total_tokens = 0
    full_answer = ""
    event_counter = 0  # ✅ 新增：事件计数器
    last_answer_output = ""  # ✅ 追踪最后一个节点的 answer 输出

    initial_state = {
        "current_input": query,   # ✅ 对齐 GraphState 的字段
        "session_id": session_id,
        "kb_id": kb_id,           # ✅ 新增：注入 kb_id
    }

    # ── 中间状态映射（节点名 → 用户可读状态）──
    NODE_STATUS_MAP = {
        "rewrite_query":     "🔍 正在理解你的问题...",
        "text_embedder":     "🧠 正在将问题转为向量...",
        "vector_search":     "📚 正在检索相关资料...",
        "graphrag_search":   "🕸️ 正在检索知识图谱...",
        "rerank_skill":      "📊 正在精排检索结果...",
        "rag_answer":        "✍️ 正在生成回答...",
        "inject_rules":      "📋 正在加载对话规则...",
        "translator":        "🌐 正在翻译...",
        "text_summarizer":   "📝 正在摘要...",
    }

    try:
        async for event in graph.astream_events(
                initial_state, config=config, version="v2"
        ):
            event_counter += 1
            event_id = f"{request_id}-{event_counter}"  # ✅ 新增：唯一事件ID

            # ✅ 新增：如果指定了 last_event_id，跳过已发送的事件
            if last_event_id and event_id <= last_event_id:
                continue

            event_type = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── ✅ 新增：节点开始时推送状态 ──
            if event_type == "on_chain_start" and name in NODE_STATUS_MAP:
                yield ServerSentEvent(
                    id=event_id,
                    event=SSEEvent.STATUS,
                    data=json.dumps({
                        "status": name,
                        "message": NODE_STATUS_MAP[name],
                        "request_id": request_id,
                        "session_id": session_id,
                    }, ensure_ascii=False),
                )

            if event_type == "on_chain_end" and name and data:
                output_data = data.get("output", "")
                # ✅ 从回答节点输出中提取 answer
                if isinstance(output_data, dict) and output_data.get("answer"):
                    last_answer_output = output_data.get("answer", "")
                truncated = _safe_truncate(output_data)
                yield ServerSentEvent(
                    id=event_id,  # ✅ 新增：事件ID
                    event=SSEEvent.NODE_OUTPUT,
                    data=json.dumps(
                        {
                            "node": name,
                            "output": truncated["content"],
                            "is_truncated": truncated["is_truncated"],
                            "request_id": request_id,
                            "session_id": session_id,
                        },
                        ensure_ascii=False,
                    ),
                )

            if event_type == "on_chat_model_stream" and name == "answer":
                chunk = data.get("chunk", {})
                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")
                if content:
                    full_answer += str(content)
                    total_tokens += 1
                    yield ServerSentEvent(
                        id=event_id,  # ✅ 新增：事件ID
                        event=SSEEvent.ANSWER_CHUNK,
                        data=json.dumps(
                            {
                                "chunk": str(content),
                                "request_id": request_id,
                                "session_id": session_id,
                            },
                            ensure_ascii=False,
                        ),
                    )

        # ✅ 若无 streaming chunks，答案通过 DONE 事件发送给前端
        if not full_answer:
            full_answer = last_answer_output or "抱歉，未能生成回答。"

        from src.infrastructure.session_manager import Message
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        user_msg = Message(role="user", content=query, created_at=now)
        assistant_msg = Message(role="assistant", content=full_answer, created_at=now)
        sm.add_message(session_id, user_msg)
        sm.add_message(session_id, assistant_msg)

        # ✅ 新增：触发对话摘要（异步，不阻塞）
        _trigger_summary_async(session_id, sm)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        yield ServerSentEvent(
            id=f"{request_id}-done",  # ✅ 新增：事件ID
            event=SSEEvent.DONE,
            data=json.dumps(
                {
                    "status": "completed",
                    "elapsed_ms": round(elapsed_ms, 2),
                    "total_tokens": total_tokens,
                    "request_id": request_id,
                    "session_id": session_id,
                    "chunk": full_answer,   # ✅ 前端优先读取 chunk
                    "answer": full_answer,  # ✅ 双重保险
                },
                ensure_ascii=False,
            ),
        )

    except Exception as e:
        logger.error(
            f"SSE 流式输出失败 request_id={request_id} session={session_id}: {e}",
            exc_info=True,
        )
        error_detail = "Internal server error" if settings.env != "dev" else str(e)
        yield ServerSentEvent(
            id=f"{request_id}-error",  # ✅ 新增：事件ID
            event=SSEEvent.ERROR,
            data=json.dumps(
                {
                    "error": error_detail,
                    "request_id": request_id,
                    "session_id": session_id,
                },
                ensure_ascii=False,
            ),
        )


# ✅ 新增：异步触发摘要
def _trigger_summary_async(session_id: str, sm) -> None:
    """SSE 流完成后异步触发对话摘要"""
    try:
        from src.infrastructure.conversation_memory import get_conversation_memory
        import threading

        def _do_summary():
            try:
                history = sm.get_message_history(session_id, limit=50)
                if not history:
                    return
                memory = get_conversation_memory()
                if not memory.should_summarize(session_id, len(history)):
                    return
                from src.core.model_client import LiteLLMClient
                from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
                from src.infrastructure.vector_store import get_vector_store

                memory.vector_store = get_vector_store()
                llm = LiteLLMClient()
                embedder = TextEmbedderSkill()
                memory.maybe_summarize(
                    session_id=session_id,
                    history=history,
                    llm_client=llm,
                    embedder=embedder,
                    session_manager=sm,
                )
            except Exception as e:
                logging.getLogger("sse_stream").debug("异步摘要失败: %s", e)

        t = threading.Thread(target=_do_summary, daemon=True)
        t.start()
    except Exception:
        pass


def create_sse_response(
        generator: AsyncGenerator[ServerSentEvent, None],
) -> EventSourceResponse:
    """创建 SSE StreamingResponse"""
    return EventSourceResponse(
        content=generator,
        send_timeout=30,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",  # ✅ 新增：允许跨域（生产环境应限制）
        },
    )