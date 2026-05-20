"""
src/agent/langgraph/graphs.py — v3.0 优化版

核心改动：
- BaseNode 接受 config 参数（修复 P0 bug）
- 图构建统一注入 SkillManager（避免重复初始化）
- 提取 _prepare_pipeline 消除 run/stream 重复
- graph.invoke/stream 补充 RunnableConfig
- 输入验证 + PipelineExecutionError 封装
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field  # ← 新增导入

from src.agent.langgraph.state import (
    GraphState,
    create_initial_state,
    ReflectionContext,  # ← 新增导入
)
from src.skills.base.skill_manager import SkillManager

from src.agent.langgraph.nodes import (
    TranslatorNode,
    BaseNode,
    build_skill_node,
    _get_default_skill_manager,
)
from src.agent.langgraph.node_impls import (
    GeneratorNode,           # ← 新增导入
    ExternalFeedbackNode,    # ← 新增导入
    CriticNode,              # ← 新增导入
    ReviserNode,             # ← 新增导入
)
from src.agent.langgraph.checkpointer import get_checkpointer

logger = logging.getLogger("langgraph.graphs")


# ═══════════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# 公共辅助
# ═══════════════════════════════════════════════════════════════

class PipelineExecutionError(Exception):
    """流水线执行异常（封装底层异常 + 上下文）"""

    def __init__(self, message: str, pipeline_name: str, node_name: str = None, cause: Exception = None):
        super().__init__(message)
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.cause = cause


def _validate_input(input_text: str) -> None:
    """输入验证"""
    if input_text is None:
        raise ValueError("input_text 不能为 None")
    if not isinstance(input_text, str):
        raise TypeError(f"input_text 必须为 str，实际为 {type(input_text).__name__}")
    if not input_text.strip():
        raise ValueError("input_text 不能为空字符串")




# ═══════════════════════════════════════════════════════════════
# 便捷调用
# ═══════════════════════════════════════════════════════════════



from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END

# ============================================================================
# 流水线配置 — 已精简为只保留知识库检索相关流水线
# 简单的 A→B 办公流水线已移除（一条 prompt 即可覆盖）
# ============================================================================

PIPELINE_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    # 保留 index 相关流水线交由 LangGraph 图模式处理，这里不需要注册
}


# ============================================================================
# 通用流水线运行函数 — 原来的 _prepare_pipeline 泛化版
# ============================================================================

def run_pipeline(
    input_text: str,
    pipeline_name: str,
    steps_config: Dict[str, Optional[Dict[str, Any]]],
    session_id: Optional[str] = None,
    skill_manager: Optional[SkillManager] = None,
) -> Dict[str, Any]:
    """
    通用流水线执行器
    ...
    """
    # ✅ 先执行输入验证（在获取 skill_manager 之前）
    try:
        _validate_input(input_text)
    except (ValueError, TypeError) as e:
        raise PipelineExecutionError(
            message=str(e),
            pipeline_name=pipeline_name,
            cause=e,
        ) from e

        # ✅ 检查流水线是否存在
    if pipeline_name not in PIPELINE_REGISTRY:
        raise KeyError(f"Unknown pipeline: '{pipeline_name}'")

    sm = skill_manager or _get_default_skill_manager()
    pipeline = PIPELINE_REGISTRY[pipeline_name]

    graph = StateGraph(GraphState)
    prev_node: Optional[str] = None

    for step in pipeline:
        node_name   = step["node_name"]
        skill_name  = step["skill"]
        config_key  = step["config_key"]
        node_config = steps_config.get(config_key) or {}

        NodeClass = build_skill_node(skill_name, sm)
        graph.add_node(node_name, NodeClass(config=node_config))

        if prev_node is None:
            graph.set_entry_point(node_name)
        else:
            graph.add_edge(prev_node, node_name)
        prev_node = node_name

    graph.add_edge(prev_node, END)

    # ✅ 注入 Checkpointer 支持断点续跑
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)
    initial_state = create_initial_state(input_text, session_id=session_id)

    try:
        # ✅ 传入 thread_id 实现会话隔离
        # 🔧 修复：如果未提供 session_id，生成唯一 UUID 避免测试间状态污染
        thread_id = session_id or f"session-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        return compiled.invoke(initial_state, config=config)
    except Exception as e:
        raise PipelineExecutionError(
            message=f"流水线执行失败: {e}",
            pipeline_name=pipeline_name,
            cause=e,
        ) from e


from enum import Enum
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END


# ============================================================================
# RAG 类型枚举
# ============================================================================

class RAGType(str, Enum):
    """RAG 检索类型"""
    STANDARD = "standard"
    GRAPHRAG = "graphrag"
    HYBRID   = "hybrid"


# ============================================================================
# RAG 图配置 — 与 PIPELINE_REGISTRY 同一哲学，用数据描述结构
# ============================================================================

# 所有 RAG 类型共享的前置节点
_RAG_PRE_NODES: List[Dict[str, Any]] = [
    {"node_name": "rewrite_query", "skill": "query_rewrite", "config_key": "rewrite_config"},
]

# 所有 RAG 类型共享的后置节点
_RAG_POST_NODES: List[Dict[str, Any]] = [
    {"node_name": "rerank", "skill": "rerank",      "config_key": "rerank_config"},
    {"node_name": "answer", "skill": "rag_answer",   "config_key": "answer_config"},
]

# 各 RAG 类型的检索节点 — 唯一差异在这里
_RAG_RETRIEVAL_NODES: Dict[RAGType, List[Dict[str, Any]]] = {
    RAGType.STANDARD: [
        {"node_name": "vector_search", "skill": "vector_search", "config_key": "vector_config"},
    ],
    RAGType.GRAPHRAG: [
        {"node_name": "graphrag_search", "skill": "graphrag_searcher", "config_key": "graphrag_config"},
    ],
    RAGType.HYBRID: [
        {"node_name": "vector_search",   "skill": "vector_search",    "config_key": "vector_config"},
        {"node_name": "graphrag_search", "skill": "graphrag_searcher","config_key": "graphrag_config"},
    ],
}


# ============================================================================
# 内部辅助
# ============================================================================

def _add_skill_node(
    graph: StateGraph,
    node_name: str,
    skill_name: str,
    config: Optional[Dict[str, Any]],
    skill_manager: SkillManager,
) -> None:
    """在图上添加一个技能节点（消除 build_skill_node + add_node 重复）"""
    NodeClass = build_skill_node(skill_name, skill_manager)
    graph.add_node(node_name, NodeClass(config=config or {}))


def _build_rag_graph(
        rag_type: RAGType,
        steps_config: Dict[str, Optional[Dict[str, Any]]],
        skill_manager: SkillManager,
        rules_engine=None,  # ✅ 第 357 行：新增参数
) -> Any:
    """
    根据配置构建 RAG 图（内部函数）

    图结构：
        [inject_rules] → rewrite_query → [检索节点...] → rerank → answer → END

    对于 HYBRID，两个检索节点并行执行，LangGraph 在 rerank 处自动 fan-in。
    """
    from src.infrastructure.rules_engine import RulesEngine  # ✅ 第 369 行：导入

    graph = StateGraph(GraphState)
    retrieval_nodes = _RAG_RETRIEVAL_NODES[rag_type]

    # ✅ 第 373-387 行：新增规则注入节点逻辑
    if rules_engine is not None:
        # 添加 RulesInjectorNode 作为第一个节点
        from src.agent.langgraph.node_impls import RulesInjectorNode
        rules_injector = RulesInjectorNode(rules_engine=rules_engine)
        graph.add_node("inject_rules", rules_injector)
        graph.set_entry_point("inject_rules")

        # inject_rules → rewrite_query
        pre_node = _RAG_PRE_NODES[0]
        _add_skill_node(
            graph, pre_node["node_name"], pre_node["skill"],
            steps_config.get(pre_node["config_key"]), skill_manager,
        )
        graph.add_edge("inject_rules", pre_node["node_name"])
    else:
        # 原有逻辑：rewrite_query 作为入口
        pre_node = _RAG_PRE_NODES[0]
        _add_skill_node(
            graph, pre_node["node_name"], pre_node["skill"],
            steps_config.get(pre_node["config_key"]), skill_manager,
        )
        graph.set_entry_point(pre_node["node_name"])

    # --- 检索节点（从 rewrite_query 扇出） ---
    for rn in retrieval_nodes:
        _add_skill_node(
            graph, rn["node_name"], rn["skill"],
            steps_config.get(rn["config_key"]), skill_manager,
        )
        graph.add_edge(pre_node["node_name"], rn["node_name"])

    # --- 后置节点 ---
    prev_name: Optional[str] = None
    for pn in _RAG_POST_NODES:
        _add_skill_node(
            graph, pn["node_name"], pn["skill"],
            steps_config.get(pn["config_key"]), skill_manager,
        )
        if prev_name is None:
            # 第一个后置节点：从所有检索节点扇入
            for rn in retrieval_nodes:
                graph.add_edge(rn["node_name"], pn["node_name"])
        else:
            graph.add_edge(prev_name, pn["node_name"])
        prev_name = pn["node_name"]

    # 🔧 Day 3：添加条件边占位接口（为第八阶段反思模式准备）
    def should_continue_reflection(state: GraphState) -> str:
        """
        条件判断函数：决定是否进入反思循环

        TODO: 第八阶段实现真实逻辑
        - 检查 answer 的质量评分
        - 如果置信度低，返回 "reflect" 重新检索
        - 否则返回 "end" 结束流程

        当前占位：总是返回 "end"
        """
        # 预留字段：state.get("answer_quality_score")
        # 预留字段：state.get("needs_reflection")
        return "end"

    # 定义条件边的路由映射
    conditional_edges_map = {
        "end": END,
        # 第八阶段可能添加："reflect": "rewrite_query"
    }

    # 添加条件边（替代原来的硬编码 add_edge）
    graph.add_conditional_edges(
        source=prev_name,  # answer 节点
        path=should_continue_reflection,
        path_map=conditional_edges_map,
    )
    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)  # ✅ 第 432 行：传入 checkpointer


# ============================================================================
# 公共入口
# ============================================================================

def run_rag_pipeline(
    query: str,
    rag_type: str = "standard",
    steps_config: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    session_id: Optional[str] = None,
    skill_manager: Optional[SkillManager] = None,
    rules_engine=None,  # ✅ 第 444 行：新增参数
) -> Dict[str, Any]:
    """
    运行 RAG 流水线的统一入口

    Args:
        query: 用户查询
        rag_type: 'standard' | 'graphrag' | 'hybrid'
        steps_config: 各节点配置，key 为 config_key（如 'rewrite_config', 'rerank_config'）
        session_id: 可选会话 ID
        skill_manager: 可选技能管理器
        rules_engine: 可选规则引擎实例（用于注入系统规则）

    Returns:
        图执行结果

    Raises:
        ValueError: rag_type 不合法
        PipelineExecutionError: 流水线执行失败
    """
    # 校验 rag_type
    try:
        rag_type_enum = RAGType(rag_type)
    except ValueError:
        raise ValueError(
            f"无效的 rag_type: '{rag_type}'，"
            f"允许值: {[t.value for t in RAGType]}"
        )

    sm = skill_manager or _get_default_skill_manager()
    config = steps_config or {}

    try:
        graph = _build_rag_graph(rag_type_enum, config, sm, rules_engine)  # ✅ 第 476 行：传递 rules_engine
        initial_state = create_initial_state(query, session_id=session_id)

        # ✅ 注入 Checkpointer 支持断点续跑
        checkpointer = get_checkpointer()
        thread_id = session_id or f"session-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state, config=config)

        # ✅ 检查执行结果中是否有错误
        if result.get("error"):
            raise RuntimeError(f"RAG execution failed: {result['error'].message}")

        return result
    except PipelineExecutionError:
        raise
    except Exception as e:
        raise PipelineExecutionError(
            message=f"RAG 流水线执行失败: {e}",
            pipeline_name=f"rag_{rag_type}",
            cause=e,
        ) from e


# ═══════════════════════════════════════════════════════════════
# 配置模型
# ═══════════════════════════════════════════════════════════════
class ReflectionConfig(BaseModel):
    """反思模式配置 — 消除所有硬编码"""
    target_skill: str = Field(..., description="被反思的目标技能名称")
    output_field: str = Field(default="answer", description="目标技能的输出字段名")
    model: str = Field(default="gpt-4", description="LLM 模型名称")
    critic_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Critic 温度(低=稳定)")
    reviser_temperature: float = Field(default=0.5, ge=0.0, le=2.0, description="Reviser 温度(平衡)")
    max_iterations: int = Field(default=2, ge=1, description="最大反思迭代次数")
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="质量分数达标线")
    convergence_threshold: float = Field(default=0.05, ge=0.0, description="收敛检测阈值")


# ═══════════════════════════════════════════════════════════════
# 条件边（可测试）
# ═══════════════════════════════════════════════════════════════
def _is_max_iteration_reached(ctx: ReflectionContext, max_iter: int) -> bool:
    return ctx.iteration >= max_iter


def _is_quality_sufficient(ctx: ReflectionContext, threshold: float) -> bool:
    """检查质量分数是否达标"""
    if ctx.critique is not None and ctx.critique.overall_score is not None:  # ← 修复：改为 critique
        return ctx.critique.overall_score >= threshold
    return False


def _is_converged(ctx: ReflectionContext, threshold: float) -> bool:
    """检查是否收敛"""
    return ctx.check_convergence(threshold=threshold)


def build_should_continue(config: ReflectionConfig):
    """工厂函数：用闭包注入配置，返回可测试的条件边函数"""

    def should_continue(state: GraphState) -> str:
        ctx = state.get("reflection_context")
        if ctx is None:
            logger.warning("should_continue: reflection_context 为空，停止循环")
            return END
        if _is_max_iteration_reached(ctx, config.max_iterations):
            logger.info("should_continue: 达到最大迭代次数 (%d)", ctx.iteration)
            return END
        if _is_quality_sufficient(ctx, config.quality_threshold):
            logger.info("should_continue: 质量分数达标 (≥%.2f)", config.quality_threshold)
            return END
        if _is_converged(ctx, config.convergence_threshold):
            logger.info("should_continue: 收敛检测通过 (<%.2f)", config.convergence_threshold)
            return END
        logger.info("should_continue: 继续反思 (iteration=%d)", ctx.iteration)
        return "reviser"

    return should_continue


# ═══════════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════════
def build_reflection_graph(
        skill_manager: SkillManager,
        llm_client,
        config: Optional[ReflectionConfig] = None,
) -> StateGraph:
    """构建反思模式图
    Args:
        skill_manager: 技能管理器
        llm_client: LLM 客户端
        config: 反思配置（含所有阈值和模型参数）
    Returns:
        编译好的 StateGraph
    """
    cfg = config or ReflectionConfig(target_skill="unknown")
    logger.info("build_reflection_graph: target_skill=%s model=%s", cfg.target_skill, cfg.model)
    graph = StateGraph(GraphState)
    # 节点
    generator = GeneratorNode(
        skill_manager=skill_manager,
        target_skill=cfg.target_skill,
        output_field=cfg.output_field,
    )
    graph.add_node("generator", generator)
    external_feedback = ExternalFeedbackNode(target_skill=cfg.target_skill)
    graph.add_node("external_feedback", external_feedback)
    critic = CriticNode(
        llm_client=llm_client,
        model=cfg.model,
        temperature=cfg.critic_temperature,
    )
    graph.add_node("critic", critic)
    reviser = ReviserNode(
        skill_manager=skill_manager,
        config={
            "llm_client": llm_client,
            "model": cfg.model,
            "temperature": cfg.reviser_temperature,
        },
    )
    graph.add_node("reviser", reviser)
    # 边
    graph.set_entry_point("generator")
    graph.add_edge("generator", "external_feedback")
    graph.add_edge("external_feedback", "critic")
    # 条件边：闭包注入配置
    graph.add_conditional_edges(
        "critic",
        build_should_continue(cfg),
        {"reviser": "reviser", END: END},
    )
    graph.add_edge("reviser", "critic")
    return graph


# ═══════════════════════════════════════════════════════════════
# 流水线入口
# ═══════════════════════════════════════════════════════════════
def compile_reflection_graph(
        target_skill_name: str,
        skill_manager: Optional[SkillManager] = None,
        llm_client: Any = None,
        config: Optional[ReflectionConfig] = None,
        checkpointer: Any = None,
) -> CompiledStateGraph:
    """
    编译反思子图的工厂函数

    Args:
        target_skill_name: 被反思的目标技能名称（如 "rag_answer"）
        skill_manager: 技能管理器（None 则使用默认单例）
        llm_client: LLM 客户端实例
        config: 反思配置（可选，提供自定义阈值和模型参数）
        checkpointer: Checkpointer 实例（None 则使用默认）

    Returns:
        编译好的 LangGraph CompiledStateGraph

    Raises:
        ValueError: target_skill_name 为空

    Example:
        >>> graph = compile_reflection_graph("rag_answer")
        >>> result = graph.invoke({
        ...     "current_input": "什么是机器学习？",
        ...     "session_id": "test-001"
        ... })
    """
    # ── 输入验证 ──
    if not target_skill_name or not isinstance(target_skill_name, str):
        raise ValueError("target_skill_name 必须是非空字符串")

    # ── 依赖解析 ──
    if skill_manager is None:
        skill_manager = _get_default_skill_manager()

    if checkpointer is None:
        checkpointer = get_checkpointer()

    # ── 构建配置 ──
    if config is None:
        config = ReflectionConfig(target_skill=target_skill_name)
    else:
        # 当外部传入的 config.target_skill 与参数不一致时，警告并覆盖
        if config.target_skill != target_skill_name:
            logger.warning(
                "config.target_skill=%s 被 target_skill_name=%s 覆盖",
                config.target_skill,
                target_skill_name,
            )
        config = config.copy(update={"target_skill": target_skill_name})

    logger.info(
        "compile_reflection_graph: target_skill=%s model=%s max_iter=%d",
        config.target_skill,
        config.model,
        config.max_iterations,
    )

    # ── 构建图 ──
    graph_builder = build_reflection_graph(
        skill_manager=skill_manager,
        llm_client=llm_client,
        config=config,
    )

    # ── 编译 ──
    compiled = graph_builder.compile(checkpointer=checkpointer)
    return compiled


def run_reflection_pipeline(
        input_text: str,
        target_skill_name: str = "",
        skill_manager: Optional[SkillManager] = None,
        llm_client: Any = None,
        config: Optional[ReflectionConfig] = None,
        session_id: Optional[str] = None,
        checkpointer: Any = None,
) -> Dict[str, Any]:
    """
    运行反思模式流水线（一站式入口）

    内部委托 compile_reflection_graph 完成编译，本函数负责：
    - 输入验证
    - 初始状态构建
    - 图执行
    - 结果日志与异常封装

    Args:
        input_text: 用户输入文本
        target_skill_name: 目标技能名称（若 config 中已设置可省略，两者都提供时以此为准）
        skill_manager: 技能管理器（None 则使用默认单例）
        llm_client: LLM 客户端实例
        config: 反思配置（可选；target_skill 会与 target_skill_name 对齐）
        session_id: 会话 ID（用于 Checkpointer 线程隔离）
        checkpointer: Checkpointer 实例（None 则使用默认）

    Returns:
        最终状态 dict，包含 reflection_context 等字段

    Raises:
        ValueError: input_text 为空，或无法确定 target_skill
        PipelineExecutionError: 流水线执行过程中的异常封装
    """
    # ─ 输入验证 ──
    if not input_text or not isinstance(input_text, str):
        raise ValueError("input_text 必须是非空字符串")

    # ── 解析 target_skill_name ──
    if not target_skill_name:
        if config is not None and config.target_skill:
            target_skill_name = config.target_skill
        else:
            raise ValueError(
                "无法确定 target_skill：请传入 target_skill_name 参数，"
                "或在 config.target_skill 中设置目标技能名称"
            )

    # ── 编译图 ──
    compiled = compile_reflection_graph(
        target_skill_name=target_skill_name,
        skill_manager=skill_manager,
        llm_client=llm_client,
        config=config,
        checkpointer=checkpointer,
    )

    # ── 初始状态 ──
    initial_state = create_initial_state(input_text, session_id=session_id)

    # ── 执行 ──
    thread_id = session_id or f"reflection-{uuid.uuid4()}"
    run_config = {"configurable": {"thread_id": thread_id}}

    try:
        result = compiled.invoke(initial_state, config=run_config)
    except Exception as exc:
        raise PipelineExecutionError(
            message=f"反思流水线执行失败: {exc}",  # ← 修复：使用正确参数名
            pipeline_name=f"reflection_{target_skill_name}",  # ← 修复：使用正确参数
            cause=exc,  # ← 修复：添加 cause 参数
        ) from exc

    # ── 结构化日志 ──
    ctx = result.get("reflection_context")
    if ctx:
        logger.info(
            "run_reflection_pipeline 完成 | target_skill=%s model=%s "
            "thread_id=%s iteration=%d refined_len=%d",
            target_skill_name,
            config.model if config else "default",
            thread_id,
            ctx.iteration,
            len(ctx.refined_output or ""),
        )
    else:
        logger.warning(
            "run_reflection_pipeline 完成但无 reflection_context | "
            "target_skill=%s thread_id=%s",
            target_skill_name,
            thread_id,
        )

    return result

