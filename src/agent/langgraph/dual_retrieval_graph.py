"""
src/agent/langgraph/dual_retrieval_graph.py

DualRetrievalGraph - 规则注入 + 向量检索的完整 RAG 流水线 (优化版)

图结构：
    RulesInjectorNode → QueryRewriteNode → TextEmbedderNode → VectorStoreRetrievalNode
    → RerankNode → ContextMergerNode → RagAnswerNode → END
"""

from __future__ import annotations

import logging
import time
import uuid as _uuid
from typing import Any, Dict, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agent.langgraph.state import GraphState, create_initial_state
from src.agent.langgraph.checkpointer import get_checkpointer
from src.agent.langgraph.nodes import RulesInjectorNode, build_skill_node
from src.agent.langgraph.node_impls.context_merger import ContextMergerNode
from src.agent.langgraph.node_impls.vector_store_retrieval import VectorStoreRetrievalNode
from src.skills.base.skill_manager import SkillManager
from src.infrastructure.rules_engine import RulesEngine
from src.infrastructure.session_manager import SessionManager
from src.infrastructure.vector_store import VectorStoreManager
from src.agent.langgraph.nodes import (
    TranslatorNode,
    SummarizerNode,
    TextEmbedderNode,      # ← 新增
    RerankSkillNode,       # ← 新增
    RagAnswerNode,         # ← 新增
    QueryRewriteNode,
    build_skill_node,
    _get_default_skill_manager,
)
logger = logging.getLogger("langgraph.dual_retrieval")

# ---------------------------------------------------------------------------
# 节点命名常量
# ---------------------------------------------------------------------------
NODE_INJECT_RULES = "inject_rules"
NODE_REWRITE_QUERY = "rewrite_query"
NODE_EMBED_QUERY = "embed_query"
NODE_RETRIEVE_CANDIDATES = "retrieve_candidates"
NODE_RERANK = "rerank"
NODE_MERGE_CONTEXT = "merge_context"
NODE_ANSWER = "answer"

# ---------------------------------------------------------------------------
# 配置模型
# ---------------------------------------------------------------------------
class RewriteQueryConfig(BaseModel):
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: int = 500

class EmbedQueryConfig(BaseModel):
    model_name: str = "text-embedding-ada-002"

class RetrieveCandidatesConfig(BaseModel):
    top_k: int = 5
    collection_name: Optional[str] = None  # None 时由节点自动选择

class RerankConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

class RagAnswerConfig(BaseModel):
    model_name: str = "gpt-4"
    temperature: float = 0.5
    max_tokens: int = 1000

class StepsConfig(BaseModel):
    """各节点配置，每字段对应一个节点，默认值取自各节点配置类"""
    rewrite_query: RewriteQueryConfig = Field(default_factory=RewriteQueryConfig)
    embed_query: EmbedQueryConfig = Field(default_factory=EmbedQueryConfig)
    retrieve_candidates: RetrieveCandidatesConfig = Field(default_factory=RetrieveCandidatesConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    rag_answer: RagAnswerConfig = Field(default_factory=RagAnswerConfig)

# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------
class RetrievalSource(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class PipelineResult(BaseModel):
    success: bool
    session_id: str
    thread_id: str
    answer_text: Optional[str] = None
    merged_context: Optional[str] = None
    retrieval_sources: List[RetrievalSource] = Field(default_factory=list)
    applied_rules: List[str] = Field(default_factory=list)
    skill_results: List[Any] = Field(default_factory=list)  # ✅ 修复：改为 List
    error_message: Optional[str] = None
    elapsed_ms: float = 0.0
    system_prefix: Optional[str] = None  # ✅ 新增：规则前缀

# ---------------------------------------------------------------------------
# 图构建（拆分为多个阶段）
# ---------------------------------------------------------------------------

def _add_rules_and_query_stage(
    graph: StateGraph,
    rules_engine: RulesEngine,
    skill_manager: SkillManager,
    cfg: StepsConfig,
) -> None:
    """注入规则 + 查询改写"""
    # 规则注入
    rules_node = RulesInjectorNode(rules_engine=rules_engine)
    graph.add_node(NODE_INJECT_RULES, rules_node)
    graph.set_entry_point(NODE_INJECT_RULES)

    # ✅ 修复：在实例化节点时传入 config，而非 add_node 时
    RewriteNodeClass = build_skill_node("query_rewrite_skill", skill_manager)
    rewrite_node = RewriteNodeClass(config=cfg.rewrite_query.dict())
    graph.add_node(NODE_REWRITE_QUERY, rewrite_node)
    graph.add_edge(NODE_INJECT_RULES, NODE_REWRITE_QUERY)



def _add_embedding_stage(
    graph: StateGraph,
    skill_manager: SkillManager,
    cfg: StepsConfig,
) -> None:
    """文本转向量"""
    # ✅ 修复：使用专用的 TextEmbedderNode
    embed_node = TextEmbedderNode(skill_manager=skill_manager, config=cfg.embed_query.dict())
    graph.add_node(NODE_EMBED_QUERY, embed_node)
    graph.add_edge(NODE_REWRITE_QUERY, NODE_EMBED_QUERY)


def _add_retrieval_stage(
    graph: StateGraph,
    vs_manager: VectorStoreManager,
    skill_manager: SkillManager,
    cfg: StepsConfig,
) -> None:
    """向量检索（已含排序，不再单独使用 VectorSearch 节点）"""
    # ✅ 修复：VectorStoreRetrievalNode 配置已在构造时传入
    from src.agent.langgraph.node_impls.vector_store_retrieval import VectorStoreRetrievalConfig
    retrieve_config = VectorStoreRetrievalConfig(
        top_k=cfg.retrieve_candidates.top_k,
    )
    retrieve_node = VectorStoreRetrievalNode(
        vector_store_manager=vs_manager,
        config=retrieve_config,
    )
    graph.add_node(NODE_RETRIEVE_CANDIDATES, retrieve_node)
    graph.add_edge(NODE_EMBED_QUERY, NODE_RETRIEVE_CANDIDATES)

    # ✅ 修复：使用专用的 RerankSkillNode
    rerank_node = RerankSkillNode(skill_manager=skill_manager, config=cfg.rerank.dict())
    graph.add_node(NODE_RERANK, rerank_node)
    graph.add_edge(NODE_RETRIEVE_CANDIDATES, NODE_RERANK)


def _add_context_and_answer_stage(
    graph: StateGraph,
    skill_manager: SkillManager,
    session_manager: Optional[SessionManager],
    cfg: StepsConfig,
) -> None:
    """上下文合并与最终回答生成"""
    # 上下文合并
    merger_node = ContextMergerNode(session_manager=session_manager)
    graph.add_node(NODE_MERGE_CONTEXT, merger_node)
    graph.add_edge(NODE_RERANK, NODE_MERGE_CONTEXT)

    # ✅ 修复：使用专用的 RagAnswerNode，传入 LLM 客户端
    from src.core.model_client import LiteLLMClient
    llm_client = LiteLLMClient()
    answer_node = RagAnswerNode(
        skill_manager=skill_manager,
        config=cfg.rag_answer.dict(),
        llm_client=llm_client,
    )
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_edge(NODE_MERGE_CONTEXT, NODE_ANSWER)
    graph.add_edge(NODE_ANSWER, END)


def build_dual_retrieval_graph(
    skill_manager: SkillManager,
    rules_engine: RulesEngine,
    session_manager: Optional[SessionManager] = None,
    vector_store_manager: Optional[VectorStoreManager] = None,
    steps_config: Optional[StepsConfig] = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """
    构建并编译 DualRetrievalGraph

    所有外部依赖（包括 VectorStoreManager）均由调用者明确提供，
    若为 None 则调用者应负责提供默认实例（见 run_dual_retrieval_pipeline）。

    Args:
        skill_manager: 技能管理器
        rules_engine: 规则引擎
        session_manager: 会话管理器（可选）
        vector_store_manager: 向量存储管理器（必需，不可为 None）
        steps_config: 节点配置，未提供时使用默认值
        checkpointer: LangGraph checkpointer，为 None 时使用默认实现

    Returns:
        编译后的 CompiledStateGraph
    """
    if vector_store_manager is None:
        raise ValueError("vector_store_manager 不能为 None，请提供有效实例")

    cfg = steps_config or StepsConfig()

    graph = StateGraph(GraphState)

    _add_rules_and_query_stage(graph, rules_engine, skill_manager, cfg)
    _add_embedding_stage(graph, skill_manager, cfg)
    _add_retrieval_stage(graph, vector_store_manager, skill_manager, cfg)
    _add_context_and_answer_stage(graph, skill_manager, session_manager, cfg)

    _checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
    return graph.compile(checkpointer=_checkpointer)

# ---------------------------------------------------------------------------
# 流水线执行
# ---------------------------------------------------------------------------

def run_dual_retrieval_pipeline(
    query: str,
    session_id: str,
    skill_manager: SkillManager,
    rules_engine: RulesEngine,
    session_manager: Optional[SessionManager] = None,
    vector_store_manager: Optional[VectorStoreManager] = None,
    steps_config: Optional[StepsConfig] = None,
    checkpointer: Any = None,
) -> PipelineResult:
    """
    运行 DualRetrieval 流水线，返回标准化结果

    Args:
        query: 用户查询
        session_id: 会话 ID（必需）
        skill_manager: 技能管理器（必需）
        rules_engine: 规则引擎（必需）
        session_manager: 会话管理器（可选，提供后启用 @msg 引用消息解析）
        vector_store_manager: 向量存储管理器（可选，未提供时将使用默认实例）
        steps_config: 各节点配置（可选，使用 Pydantic 模型）
        checkpointer: LangGraph checkpointer（可选）

    Returns:
        PipelineResult: 包含成功标志、最终回答、检索来源、规则列表等字段
    """
    if not session_id:
        raise ValueError("session_id 不能为空")

    # 依赖解析：统一确定默认值，确保显式传入构建函数
    if vector_store_manager is None:
        logger.info("未提供 vector_store_manager，将使用默认实例")
        vector_store_manager = VectorStoreManager.default()

    # 编译图
    try:
        compiled_graph = build_dual_retrieval_graph(
            skill_manager=skill_manager,
            rules_engine=rules_engine,
            session_manager=session_manager,
            vector_store_manager=vector_store_manager,
            steps_config=steps_config,
            checkpointer=checkpointer,
        )
    except Exception as e:
        logger.error("图编译失败: %s", e, exc_info=True)
        return PipelineResult(
            success=False,
            session_id=session_id,
            thread_id="",
            error_message=f"图编译失败: {e}",
        )

    # 准备初始状态
    initial_state = create_initial_state(query, session_id=session_id)
    thread_id = f"dual-retrieval-{session_id}-{_uuid.uuid4().hex[:12]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # ... existing code ...

    start_time = time.perf_counter()
    try:
        result_state = compiled_graph.invoke(initial_state, config=config)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # ✅ 修复：从 RulesEngine 获取实际应用的规则 ID
        if rules_engine and session_id:
            enabled_rules = rules_engine.get_enabled_rules(session_id)
            applied_rules = [r.rule_id for r in enabled_rules]
        else:
            applied_rules = result_state.get("applied_rules", [])

        # 构建标准的检索来源列表
        raw_sources = result_state.get("retrieval_sources", [])
        retrieval_sources = []
        for s in raw_sources:
            if isinstance(s, dict):
                retrieval_sources.append(RetrievalSource(
                    id=s.get("chunk_id", s.get("id", "")),
                    text=s.get("text", ""),
                    metadata=s.get("metadata", {}),
                    score=s.get("score"),
                ))
            else:
                retrieval_sources.append(s)

        # ✅ 修复：提取最终回答文本（兼容字典和字符串两种格式）
        raw_answer = result_state.get("current_output", "")
        if isinstance(raw_answer, dict):
            # 如果返回的是字典，尝试提取 answer 字段
            answer_text = raw_answer.get("answer", raw_answer.get("text", str(raw_answer)))
        elif isinstance(raw_answer, str):
            answer_text = raw_answer
        else:
            answer_text = str(raw_answer)

        # 提取合并后的上下文（若需要）
        merged_context = result_state.get("merged_context", "")

        logger.info(
            "DualRetrieval 流水线完成 | session_id=%s | thread_id=%s | elapsed_ms=%.1f | rules=%d | sources=%d",
            session_id,
            thread_id,
            elapsed_ms,
            len(applied_rules),
            len(retrieval_sources),
        )

        return PipelineResult(
            success=True,
            session_id=session_id,
            thread_id=thread_id,
            answer_text=answer_text,
            merged_context=merged_context,
            retrieval_sources=retrieval_sources,
            applied_rules=applied_rules,
            skill_results=result_state.get("skill_results", []),  # ✅ 修复：使用 list
            elapsed_ms=elapsed_ms,
            system_prefix=result_state.get("system_prefix"),  # ✅ 新增
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "DualRetrieval 流水线失败 | session_id=%s | thread_id=%s | elapsed_ms=%.1f | error=%s",
            session_id,
            thread_id,
            elapsed_ms,
            e,
            exc_info=True,
        )
        return PipelineResult(
            success=False,
            session_id=session_id,
            thread_id=thread_id,
            error_message=str(e),
            elapsed_ms=elapsed_ms,
        )
