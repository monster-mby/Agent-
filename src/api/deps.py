"""依赖注入 - 单例服务与资源管理"""
from functools import lru_cache
from contextvars import ContextVar
from typing import Optional
from src.infrastructure.session_manager import SessionManager, SessionConfig
from src.infrastructure.rules_engine import RulesEngine
from src.infrastructure.vector_store import VectorStoreManager
from src.agent.langgraph.checkpointer import get_checkpointer, async_get_checkpointer
from src.core.config import settings


# ========== request_id 管理（供中间件调用） ==========
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unknown")


def set_request_id(request_id: str):
    """由中间件调用，设置当前请求的 request_id"""
    _request_id_ctx.set(request_id)


def get_request_id() -> str:
    """获取当前请求的 request_id"""
    return _request_id_ctx.get()


# ========== Checkpointer 单例（同步） ==========
def get_checkpointer_singleton():
    """获取 Checkpointer 单例（同步版本）"""
    if not hasattr(get_checkpointer_singleton, "_sync_instance"):
        get_checkpointer_singleton._sync_instance = get_checkpointer(use_async=False)
    return get_checkpointer_singleton._sync_instance


# ========== SessionManager 单例（修正版，不再误用 Depends） ==========
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取 SessionManager 单例"""
    global _session_manager
    if _session_manager is None:
        checkpointer = get_checkpointer_singleton()
        _session_manager = SessionManager(
            config=SessionConfig(db_path=settings.db_path),
            checkpointer=checkpointer,
        )
    return _session_manager


# ========== RulesEngine 单例 ==========
_rules_engine: Optional[RulesEngine] = None


def get_rules_engine() -> RulesEngine:
    """获取 RulesEngine 单例"""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine(db_path=settings.db_path)
    return _rules_engine


def get_vector_store_manager():
    """获取 VectorStoreManager（懒加载）"""
    return VectorStoreManager.default()


# ========== LangGraph 单例（异步） ==========
_graph_singleton = None


async def _build_graph_singleton():
    """应用启动时初始化一次，之后每次请求复用同一个实例"""
    global _graph_singleton

    if _graph_singleton is not None:
        return _graph_singleton

    from src.agent.langgraph.dual_retrieval_graph import build_dual_retrieval_graph

    # ✅ P1 修复：共用 get_skill_manager() 单例，不再单独 new
    skill_manager = get_skill_manager()

    # 确保 RAG 所需技能也已注册
    from src.skills.custom.rag_skills.query_rewrite_skill.skill import QueryRewriteSkill
    from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
    from src.skills.custom.rag_skills.rerank_skill.skill import RerankSkill
    from src.skills.custom.rag_skills.rag_answer.skill import RagAnswerSkill
    for skill_cls in [QueryRewriteSkill, TextEmbedderSkill, RerankSkill, RagAnswerSkill]:
        if not skill_manager.get(skill_cls.name):
            skill_manager.register(skill_cls)

    rules_engine = get_rules_engine()
    sm = get_session_manager()
    vs_manager = get_vector_store_manager()

    # ✅ 核心修复：异步获取异步 Checkpointer
    async_checkpointer = await async_get_checkpointer()

    _graph_singleton = build_dual_retrieval_graph(
        skill_manager=skill_manager,
        rules_engine=rules_engine,
        session_manager=sm,
        vector_store_manager=vs_manager,
        checkpointer=async_checkpointer,
    )

    return _graph_singleton


async def get_dual_retrieval_graph():
    """获取 DualRetrievalGraph 单例（FastAPI 依赖注入）"""
    return await _build_graph_singleton()


# ========== 会话所有权验证依赖 ==========
from fastapi import Depends, HTTPException
from src.api.auth import verify_api_key
from src.infrastructure.session_manager import Session as SessionSchema


def get_owned_session(
    session_id: str,
    user_id: str = Depends(verify_api_key),
    sm: SessionManager = Depends(get_session_manager),
) -> SessionSchema:
    """获取并校验会话所有权，失败自动抛 404/403"""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return session


async def get_owned_rule(
    session_id: str,
    rule_id: str,
    user_id: str = Depends(verify_api_key),
    session: SessionSchema = Depends(get_owned_session),
    re: RulesEngine = Depends(get_rules_engine),
):
    """获取并校验规则所有权，失败自动抛 404/403"""
    rule = re.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.session_id != session_id:
        raise HTTPException(status_code=403, detail="Rule does not belong to this session")
    return rule

# ========== 知识库管理依赖 ==========
from src.infrastructure.kb_manager import KBManager


# ========== SkillManager 依赖 ==========
_skill_manager: Optional["SkillManager"] = None


def get_skill_manager() -> "SkillManager":
    """获取 SkillManager 单例（自动注册核心技能）"""
    global _skill_manager
    if _skill_manager is None:
        from src.skills.base.skill_manager import SkillManager
        _skill_manager = SkillManager()
        # 注册 kb_manager
        from src.skills.custom.rag_skills.kb_manager_skill.skill import KnowledgeBaseManagerSkill
        _skill_manager.register(KnowledgeBaseManagerSkill)
        # 注册金融投研技能
        from src.skills.custom.finance_skills.stock_watcher.skill import StockWatcherSkill
        _skill_manager.register(StockWatcherSkill)
        from src.skills.custom.finance_skills.technology_radar.skill import TechnologyRadarSkill
        _skill_manager.register(TechnologyRadarSkill)
    return _skill_manager


def get_kb_manager() -> KBManager:
    """获取 KBManager 单例"""
    return KBManager.default()


async def get_owned_kb(
    kb_id: str,
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
):
    """
    获取知识库并校验所有权，失败自动抛出 HTTP 异常。
    用法：kb: KnowledgeBase = Depends(get_owned_kb)
    """
    kb = kb_manager.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    # TODO: 等 created_by 字段有真实业务数据后启用
    # if kb.created_by != user_id:
    #     raise HTTPException(status_code=403, detail="Access denied")
    return kb
