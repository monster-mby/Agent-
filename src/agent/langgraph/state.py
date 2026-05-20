"""
LangGraph State 定义 - 所有图共享的统一状态结构
"""

from typing import Any, Dict, List, Optional, TypedDict, Union, Literal
from typing_extensions import Annotated
from datetime import datetime
from uuid import uuid4
import operator

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ── 子结构 ─────────────────────────────────────────

class StateError(BaseModel):
    """结构化错误信息"""
    code: str                          # "SKILL_TIMEOUT" / "LLM_RATE_LIMIT" / "VALIDATION_ERROR"
    message: str
    step_name: Optional[str] = None
    detail: Optional[dict] = None
    traceback: Optional[str] = None
    recoverable: bool = True


class TokenUsage(BaseModel):
    """单次 LLM 调用 token 消耗"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class StepMetadata(BaseModel):
    """单步执行元数据"""
    step_name: str
    elapsed_ms: int = 0
    retries: int = 0
    token_usage: Optional[TokenUsage] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class SkillExecutionRecord(BaseModel):
    """单次技能调用记录（支持同一技能多次调用）"""
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    skill_name: str
    input_summary: str = ""            # 输入摘要，避免 state 膨胀
    output: Any = None
    elapsed_ms: int = 0
    success: bool = True
    error: Optional[StateError] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineResult(BaseModel):
    """流水线最终聚合结果"""
    pipeline_name: str
    pipeline_type: str = "sequential"
    final_output: Any = None
    total_elapsed_ms: int = 0
    success: bool = True
    summary: str = ""
    error: Optional[StateError] = None


# ── 自定义 Reducer ──────────────────────────────────

def token_usage_reducer(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """累加 token 用量"""
    return TokenUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        model=right.model or left.model,
    )


# ── 反思模式 Pydantic 模型 ──────────────────────────

class FeedbackSource(BaseModel):
    """反馈来源追踪"""
    source_type: Literal["user", "evaluator", "llm_judge", "auto"]
    source_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Feedback(BaseModel):
    """外部反馈（结构化）"""
    content: str = Field(..., max_length=10_000)
    source: FeedbackSource
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class CritiquePoint(BaseModel):
    """单条批评点"""
    target_field: str                                    # 被批评的字段/片段
    severity: Literal["critical", "major", "minor", "suggestion"]
    description: str = Field(..., max_length=2_000)
    suggested_fix: Optional[str] = Field(None, max_length=5_000)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class Critique(BaseModel):
    """批评报告"""
    points: List[CritiquePoint] = Field(default_factory=list)
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_feedback_id: Optional[str] = None             # 溯源：来自哪条 feedback
    summary: Optional[str] = Field(None, max_length=5_000, description="批评总结")  # ← 新增
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CritiqueRecord(BaseModel):
    """历史批评快照（用于收敛检测）"""
    critique: Critique
    iteration: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReflectionContext(BaseModel):
    """反思模式上下文（作为 GraphState 的单一子字段）"""
    refined_output: Optional[str] = Field(None, max_length=50_000)
    feedback: Optional[Feedback] = None
    critique: Optional[Critique] = None
    iteration: int = Field(default=0, ge=0)
    critique_history: List[CritiqueRecord] = Field(default_factory=list)
    max_critique_history: int = Field(default=10, ge=1)
    improvement_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_converged: bool = False
    status: Literal["idle", "critiquing", "revising", "awaiting_feedback", "converged", "failed"] = "idle"

    def add_critique(self, critique: Critique) -> None:
        """追加批评记录，自动裁剪历史并计算本轮改进幅度"""
        self.critique = critique
        self.critique_history.append(
            CritiqueRecord(critique=critique, iteration=self.iteration)
        )
        if len(self.critique_history) > self.max_critique_history:
            self.critique_history = self.critique_history[-self.max_critique_history:]

        # ✅ P0 修复：自动计算 improvement_score（用于收敛检测）
        if len(self.critique_history) >= 2:
            prev = self.critique_history[-2].critique.overall_score
            curr = critique.overall_score
            if prev is not None and curr is not None:
                # 改进幅度 = 本轮与上轮的分数差（正数=变好），归一化到 0-1
                self.improvement_score = max(0.0, min(1.0, abs(curr - prev)))

    def check_convergence(self, threshold: float = 0.05) -> bool:
        """基于最近两次迭代的 improvement_score 判断收敛"""
        if self.improvement_score is None or len(self.critique_history) < 2:
            return False
        # 第二次迭代的改进幅度低于阈值 → 收敛
        return self.improvement_score < threshold


# ── GraphState ──────────────────────────────────────

class GraphState(TypedDict):
    """
    LangGraph 统一状态结构（v2.0 优化版）

    设计原则:
    - messages/skill_results 使用 LangGraph 内置 reducer 保证正确合并
    - 可变数据放 State，不可变配置放 RunnableConfig.configurable
    - 所有复杂子结构用 Pydantic BaseModel，State 本体用 TypedDict（保留 reducer 能力）
    - skill_results 改为 List 支持同一技能多次调用
    """

    # ── 对话历史（LangGraph add_messages reducer，按 ID 去重合并）──
    messages: Annotated[List[BaseMessage], add_messages]

    # ── 当前步骤输入输出（节点间传递）──
    current_input: Union[str, dict, None]
    # ✅ 修复：并行节点需要 reducer 来合并更新
    current_output: Annotated[Union[str, dict, None], lambda left, right: right]

    # ── RAG 专用字段 ──
    kb_id: Optional[str]                     # 当前使用的知识库 ID（vector_namespace = f"kb_{kb_id}"）
    rewritten_query: Optional[str]
    search_results: Annotated[List[Any], operator.add]  # 支持并行分支合并（逻辑保留，但目前用 candidates）
    candidates: Optional[List[Any]]          # VectorStoreRetrievalNode 输出的原始检索结果
    reranked_results: Optional[List[Any]]    # RerankSkillNode 输出（备选字段名）
    reranked_docs: Optional[List[Any]]       # RerankSkillNode 实际输出字段名
    query_vector: Optional[List[float]]  # TextEmbedderNode 输出的查询向量，供检索使用

    # ── DualRetrievalGraph 专用字段 ──
    system_prefix: Optional[str]  # RulesInjectorNode 输出的规则前缀
    retrieval_sources: Optional[List[dict]]  # 检索来源：[{doc_name, chunk_id, score}, ...]
    applied_rules: Optional[List[str]]  # 应用的规则 ID 列表
    merged_context: Optional[str]  # ContextMergerNode 拼接的完整上下文



    # ── 技能执行记录（List + operator.add，支持多次调用不覆盖）──
    skill_results: Annotated[List[SkillExecutionRecord], operator.add]

    # ── 流水线最终结果 ──
    pipeline_results: Optional[PipelineResult]

    # ── 错误信息（结构化）──
    error: Optional[StateError]

    # ── 步骤级元数据 ─
    # ✅ 修复：并行节点写入 metadata 时需要 merge reducer
    metadata: Annotated[Dict[str, StepMetadata], lambda left, right: {**left, **right}]

    # ── 运行标识（tracing + 会话关联）──
    run_id: str
    session_id: Optional[str]
    parent_run_id: Optional[str]

    # ── Token 累计（专用 reducer 累加）──
    cumulative_token_usage: Annotated[TokenUsage, token_usage_reducer]

    # ── 中断控制 ──
    cancelled: bool
    cancel_reason: Optional[str]

    # ── Schema 版本（checkpoint 恢复兼容）──
    schema_version: str

    # ── 反思模式专用字段 ──
    reflection_context: Optional[ReflectionContext]       # 反思上下文（Pydantic 模型）


# ── 便捷函数 ────────────────────────────────────────

def create_initial_state(
        input_data: Any,
        session_id: Optional[str] = None,
        schema_version: str = "2.0",
) -> GraphState:
    """
    创建初始状态的便捷函数

    Args:
        input_data: 用户输入数据
        session_id: 会话 ID（可选）
        schema_version: Schema 版本号

    Returns:
        初始化后的 GraphState
    """
    return {
        "messages": [],
        "current_input": input_data,
        "current_output": None,
        # ── RAG 专用字段 ──
        "rewritten_query": None,
        "search_results": [],  # ← 新增
        "reranked_results": None,
        "query_vector": None,  # ← 新增
        # ── DualRetrievalGraph 专用字段 ──
        "system_prefix": None,  # ← 新增
        "retrieval_sources": None,  # ← 新增
        "applied_rules": None,  # ← 新增
        "merged_context": None,  # ← 新增
        # ── 技能执行记录 ──
        "skill_results": [],
        "pipeline_results": None,
        "error": None,
        "metadata": {},
        "run_id": uuid4().hex[:12],
        "session_id": session_id or uuid4().hex[:12],
        "parent_run_id": None,
        "cumulative_token_usage": TokenUsage(),
        "cancelled": False,
        "cancel_reason": None,
        "schema_version": schema_version,
        "reflection_context": None,
    }

