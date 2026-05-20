"""会话路由（优化版 v3）"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src.api.auth import verify_api_key
from src.api.deps import (
    get_owned_session,
    get_session_manager,
    get_dual_retrieval_graph,
    get_request_id,
)
from src.api.sse_stream import stream_chat_response, create_sse_response
from src.api.schemas.session import (
    ChatRequest,
    CreateSessionRequest,
    MessageResponse,
    PaginatedResponse,
    SessionDetailResponse,
    SessionSimpleResponse,
    SessionStatus,
)
from src.infrastructure.session_manager import SessionManager, Message
from src.infrastructure.session_manager import Session as SessionSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


# 同步响应模型
class ChatResponse(BaseModel):
    answer: str

    session_id: str
    elapsed_ms: float = 0.0



@router.post("", response_model=SessionDetailResponse, status_code=201)
def create_session(
    request: CreateSessionRequest,
    user_id: str = Depends(verify_api_key),
    sm: SessionManager = Depends(get_session_manager),
):
    session = sm.create_session(
        user_id=user_id,
        name=request.name,
        knowledge_base_ids=request.knowledge_base_ids,
    )
    return SessionDetailResponse.model_validate(session)


@router.get("", response_model=PaginatedResponse[SessionSimpleResponse])
def list_sessions(
    user_id: str = Depends(verify_api_key),
    sm: SessionManager = Depends(get_session_manager),
    status: SessionStatus = Query(SessionStatus.ACTIVE),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    # StrEnum 是 str 子类，直接传入即可
    total = sm.count_sessions(user_id=user_id, status=status)
    sessions = sm.list_sessions(
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    items = [SessionSimpleResponse.model_validate(s) for s in sessions]
    page = (offset // limit) + 1 if limit else 1
    return PaginatedResponse(items=items, total=total, page=page, size=limit)


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(session: SessionSchema = Depends(get_owned_session)):
    return SessionDetailResponse.model_validate(session)


@router.delete("/{session_id}")
def delete_session(
    session: SessionSchema = Depends(get_owned_session),
    sm: SessionManager = Depends(get_session_manager),
):
    sm.delete_session(session.session_id)
    return {"message": "Session deleted", "session_id": session.session_id}


@router.get("/{session_id}/history", response_model=List[MessageResponse])
def get_message_history(
    session: SessionSchema = Depends(get_owned_session),
    sm: SessionManager = Depends(get_session_manager),
    limit: int = Query(50, ge=1, le=200),
):
    """获取会话消息历史，当前仅支持按 limit 截取最近 N 条消息"""
    messages = sm.get_message_history(
        session.session_id,
        limit=limit,
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return [MessageResponse.model_validate(m) for m in messages]

@router.post("/{session_id}/chat")
def chat(
    request: ChatRequest,
    session: SessionSchema = Depends(get_owned_session),
    sm: SessionManager = Depends(get_session_manager),
    graph: CompiledStateGraph = Depends(get_dual_retrieval_graph),
    request_id: str = Depends(get_request_id),
    last_event_id: Optional[str] = Query(None, description="最后接收的事件ID（断线重连）"),  # ✅ 新增
):
    """发送消息，根据 stream 参数自动选择同步或 SSE 流式响应"""
    config = {"configurable": {"thread_id": session.langgraph_thread_id}}
    start_time = datetime.now(timezone.utc)

    # ✅ 优先级：请求中实时传入的 kb_id > session 历史关联 > default
    kb_id = (
        request.knowledge_base_ids[0] if request.knowledge_base_ids
        else session.knowledge_base_ids[0] if session.knowledge_base_ids
        else "default"
    )

    if request.stream:
        # ---------- SSE 流式响应（生成器内部自动保存消息） ----------
        generator = stream_chat_response(
            query=request.query,
            session_id=session.session_id,
            request_id=request_id,
            graph=graph,
            config=config,
            sm=sm,
            kb_id=kb_id,  # ✅ 新增：传递 kb_id
            last_event_id=last_event_id,  # ✅ 新增：传递断点ID
        )
        return create_sse_response(generator)
    else:
        # ---------- 同步响应 ----------
        result_state = graph.invoke(
            {"current_input": request.query, "session_id": session.session_id, "kb_id": kb_id},
            config=config,
        )
        answer = str(result_state.get("current_output", ""))

        # 保存消息
        user_msg = Message(
            role="user",
            content=request.query,
            created_at=datetime.now(timezone.utc),
        )
        assistant_msg = Message(
            role="assistant",
            content=answer,
            created_at=datetime.now(timezone.utc),
        )
        sm.add_message(session.session_id, user_msg)
        sm.add_message(session.session_id, assistant_msg)

        # ✅ 新增：触发对话摘要（异步，不阻塞响应）
        _trigger_summary_if_needed(session.session_id, sm)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ChatResponse(
            answer=answer,
            session_id=session.session_id,
            elapsed_ms=elapsed,
        )

@router.get("/{session_id}/state")
def get_session_state(
        session: SessionSchema = Depends(get_owned_session),
        sm: SessionManager = Depends(get_session_manager),
):
    """获取LangGraph当前状态（从checkpoint读取）"""
    from src.agent.langgraph.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    thread_id = session.langgraph_thread_id
    config = {"configurable": {"thread_id": thread_id}}

    checkpoint = checkpointer.get(config)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found for this session")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint.get("checkpoint_ns", ""),
        "channel_values": {
            k: str(v)[:500] if isinstance(v, str) else v
            for k, v in checkpoint.get("channel_values", {}).items()
        },
        "session_status": session.status,
    }


# ✅ 新增：对话摘要触发
def _trigger_summary_if_needed(session_id: str, sm: SessionManager) -> None:
    """在保存消息后异步触发摘要生成"""
    try:
        from src.infrastructure.conversation_memory import (
            ConversationMemory, get_conversation_memory
        )
        from src.infrastructure.vector_store import get_vector_store
        from src.core.model_client import LiteLLMClient
        from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill

        history = sm.get_message_history(session_id, limit=50)
        if not history:
            return

        memory = get_conversation_memory(vector_store=get_vector_store())
        if not memory.should_summarize(session_id, len(history)):
            return

        # 异步执行，不阻塞聊天响应
        import threading

        def _do_summary():
            try:
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
                logger.warning("异步摘要生成失败: %s", e)

        t = threading.Thread(target=_do_summary, daemon=True)
        t.start()

    except Exception as e:
        logger.debug("摘要触发检查跳过: %s", e)



