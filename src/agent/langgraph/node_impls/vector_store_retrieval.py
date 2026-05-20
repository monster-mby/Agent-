
"""
src/agent/langgraph/node_impls/vector_store_retrieval.py

向量存储检索节点 - 从 ChromaDB 中检索相关文档，输出统一的候选文档列表供下游使用。
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, PositiveInt

from src.agent.langgraph.state import GraphState, StateError
from src.agent.langgraph.nodes import BaseNode

logger = logging.getLogger("langgraph.node_impls")


# ---------------------------------------------------------------------------
# 依赖抽象：VectorStoreManager 接口
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStoreManagerProtocol(Protocol):
    """VectorStoreManager 必须遵循的协议，便于测试替换"""
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# 节点配置模型
# ---------------------------------------------------------------------------
class VectorStoreRetrievalConfig(BaseModel):
    """向量检索节点的配置"""
    top_k: PositiveInt = Field(default=10, description="检索返回的最大文档数")
    timeout: float = Field(default=5.0, description="检索的超时时间(秒)")


# ---------------------------------------------------------------------------
# 节点内部数据模型
# ---------------------------------------------------------------------------
class RetrievalInput(BaseModel):
    """from state 提取的检索输入"""
    query_vector: Optional[List[float]] = None
    kb_id: str = "default"
    top_k: int = 10

    @classmethod
    def from_state(cls, state: GraphState, config: VectorStoreRetrievalConfig) -> "RetrievalInput":
        qv = state.get("query_vector")

        # ✅ 修复 1：回退逻辑 - 从上游 TextEmbedder 的 current_output 中提取向量
        if qv is None:
            current_output = state.get("current_output")
            if isinstance(current_output, dict):
                qv = current_output.get("embedding") or current_output.get("query_embedding")
            elif isinstance(current_output, list) and current_output:
                first = current_output[0]
                if isinstance(first, dict):
                    qv = first.get("embedding")

        # ✅ 修复 2：将 kb_id 映射为 ChromaDB 实际的 vector_namespace
        raw_kb_id = state.get("kb_id", "default") or "default"
        if raw_kb_id == "default":
            namespace = "default"
            logger.info("RetrievalInput.from_state: kb_id=%s → namespace=%s (no KB selected)", raw_kb_id, namespace)
        else:
            from src.infrastructure.kb_manager import generate_vector_namespace
            namespace = generate_vector_namespace(raw_kb_id)
            logger.info("RetrievalInput.from_state: kb_id=%s → namespace=%s", raw_kb_id, namespace)

        return cls(
            query_vector=qv,
            kb_id=namespace,
            top_k=config.top_k,
        )


class RetrievalOutput(BaseModel):
    """检索输出，仅保留 candidates 供下游使用"""
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="检索到的候选文档列表")


# ---------------------------------------------------------------------------
# 优化后的节点
# ---------------------------------------------------------------------------
class VectorStoreRetrievalNode(BaseNode):
    """
    向量存储检索节点
    ...（保留原有注释）...
    """
    skill_name = "vector_retrieval"

    def __init__(
        self,
        vector_store_manager: VectorStoreManagerProtocol,
        config: Optional[VectorStoreRetrievalConfig] = None,
    ):
        cfg = config or VectorStoreRetrievalConfig()
        super().__init__(skill_manager=None, config=cfg.dict())
        self._vs_manager = vector_store_manager
        self._config_model = cfg
        self._current_state = None  # ✅ 新增：缓存完整 state

    def __call__(self, state: GraphState, config=None) -> dict:
        """
        覆盖父类入口：缓存完整 state 供 adapt_input 使用，
        因为父类 __call__ 默认只传入 raw_input 字符串。
        """
        # ✅ 新增：调试日志，查看 state 中有哪些键
        logger.info(f"VectorStoreRetrievalNode: State keys = {list(state.keys())}")
        logger.info(f"VectorStoreRetrievalNode: query_vector present = {'query_vector' in state}")
        if 'query_vector' in state:
            qv = state.get('query_vector')
            logger.info(f"VectorStoreRetrievalNode: query_vector type = {type(qv)}, len = {len(qv) if qv else 0}")

        self._current_state = state
        try:
            return super().__call__(state, config)
        finally:
            self._current_state = None

    def _get_skill(self):
        """✅ 新增：本节点不使用外部 SkillManager，直接返回自身作为执行体"""
        return self

    # ------------------------------ 适配输入 ------------------------------
    def adapt_input(self, raw_input, skill=None, config=None) -> RetrievalInput:
        """✅ 修正：从缓存的全量 state 提取 query_vector / kb_id"""
        if self._current_state is not None:
            return RetrievalInput.from_state(self._current_state, self._config_model)
        logger.warning("VectorStoreRetrievalNode: _current_state is None, returning empty RetrievalInput")
        return RetrievalInput()

# ... existing code ...

    # def _get_skill(self):
    #     """本节点不使用外部技能，直接返回自身作为执行体"""
    #     return self
    # ------------------------------ 核心执行 ------------------------------
    def execute(self, input_data: RetrievalInput) -> RetrievalOutput:
        if not input_data.query_vector:
            logger.warning("查询向量缺失，跳过检索 (kb_id=%s)", input_data.kb_id)
            return RetrievalOutput()

        # ✅ 新增：检查集合是否存在，若不存在则自动创建
        try:
            if not self._vs_manager.collection_exists(input_data.kb_id):
                logger.warning(
                    "集合 '%s' 不存在，自动创建空集合。请先向知识库添加文档。",
                    input_data.kb_id
                )
                self._vs_manager.get_or_create_collection(input_data.kb_id)
        except Exception as exc:
            logger.error("自动创建集合失败 kb_id=%s error=%s", input_data.kb_id, exc)

        try:
            results = self._vs_manager.search(
                collection_name=input_data.kb_id,
                query_embedding=input_data.query_vector,
                top_k=input_data.top_k,
            )
        except Exception as exc:
            logger.error(
                "向量检索失败 kb_id=%s top_k=%d error=%s",
                input_data.kb_id, input_data.top_k, exc,
                exc_info=True,
            )
            # 返回空结果
            return RetrievalOutput()

        # 防御性转换
        candidates = []
        for idx, doc in enumerate(results):
            try:
                candidates.append({
                    "chunk_id": doc.get("id", f"doc_{idx}"),
                    "text": doc.get("text", ""),
                    "embedding": doc.get("embedding", []),
                    "metadata": doc.get("metadata", {}),
                })
            except Exception:
                logger.warning("无效的检索结果项: %s", doc)

        logger.info(
            "向量检索完成 kb_id=%s top_k=%d result_count=%d",
            input_data.kb_id, input_data.top_k, len(candidates),
        )
        return RetrievalOutput(candidates=candidates)

    # ------------------------------ 提取输出 ------------------------------
    def extract_output(self, result: RetrievalOutput, skill=None) -> dict:
        """将结构化输出转换为 state 可合并的字典格式"""
        output = {"candidates": result.candidates}
        # ✅ 新增：如果没有检索到结果，在 state 中标记错误供下游检查
        if not result.candidates:
            output["error"] = StateError(
                code="VECTOR_RETRIEVAL_EMPTY",
                message="向量检索未返回任何结果，请检查知识库中是否有文档",
                step_name="vector_retrieval",
                recoverable=True,
            )
        return output
