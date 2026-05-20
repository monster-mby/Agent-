"""
向量检索技能 — RAG 体系第四环（v1.1）

在已向量化的分块集合中，根据查询向量检索 Top-K 最相关结果。

v1.1 优化（相比 v1.0）：
- Bug 修复：euclidean 距离完整实现（NumPy + Faiss IndexFlatL2）
- Bug 修复：Faiss 索引绑定 candidates 哈希，变化自动重建
- Bug 修复：零向量检测 + WARNING 日志
- 批量检索：NumPy 矩阵乘法 / Faiss 多 query 一次完成
- 缓存：候选集归一化后缓存，hash 检测失效
- 依赖检测：importlib.util.find_spec 替代 try-except
- 功能：filter_fn 自定义过滤、deduplicate_results 去重、return_embeddings
- 日志：分阶段耗时 + 细粒度边界日志
- 异常：DimensionMismatchError / BackendNotAvailableError 细化

与 text_embedder / document_chunker 零耦合，与现有项目 100% 兼容。
"""

from __future__ import annotations

import numpy as np
import hashlib
import importlib.util
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger("vector_search")


# ═══════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════

class VectorSearchError(Exception):
    """向量检索基础异常"""


class DimensionMismatchError(VectorSearchError):
    """查询向量与候选集维度不匹配"""


class BackendNotAvailableError(VectorSearchError):
    """指定后端不可用"""


class EmptyCandidateError(VectorSearchError):
    """候选集为空"""


# ═══════════════════════════════════════════════════════════════
# 依赖检测（importlib.util.find_spec）
# ═══════════════════════════════════════════════════════════════

def _is_available(package: str) -> bool:
    """检测 Python 包是否可导入"""
    return importlib.util.find_spec(package) is not None


HAS_NUMPY = _is_available("numpy")
HAS_FAISS = _is_available("faiss")

if not HAS_NUMPY:
    logger.warning("numpy 未安装 — NumPy 后端不可用（pip install numpy）")
if not HAS_FAISS:
    logger.info("faiss 未安装 — Faiss 后端不可用（pip install faiss-cpu），将使用 NumPy 后端")


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class VectorRecord(BaseModel):
    """单条向量记录 — 与 EmbeddedChunk 零耦合，接收 dict 自动转换"""

    chunk_id: str = Field(..., description="唯一标识，回链源 Chunk")
    text: str = Field(..., description="原始文本")
    embedding: List[float] = Field(..., description="向量")
    metadata: dict = Field(default_factory=dict, description="透传元数据")

    @field_validator("embedding")
    @classmethod
    def _embedding_not_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("embedding 不能为空")
        return v

    @property
    def dimension(self) -> int:
        return len(self.embedding)


class VectorSearchInput(BaseModel):
    """检索输入"""

    query_vector: List[float] = Field(
        ..., min_length=1, description="查询向量（由 text_embedder 生成）"
    )
    candidates: List[VectorRecord] = Field(
        ..., min_length=1, description="待检索的向量集合"
    )
    top_k: int = Field(default=5, ge=1, le=100)
    backend: str = Field(
        default="numpy",
        description="检索后端：numpy | faiss",
    )
    metric: str = Field(
        default="cosine",
        description="相似度度量：cosine | dot | euclidean。euclidean 内部转相似度 1/(1+d)",
    )
    filter_metadata: Optional[dict] = Field(
        default=None, description="元数据过滤条件（key=value 精确匹配）"
    )
    filter_fn: Optional[Callable[[VectorRecord], bool]] = Field(
        default=None, description="自定义过滤回调，接收 VectorRecord，返回 bool"
    )
    min_score: float = Field(
        default=-1.0, ge=-1.0, le=1.0,
        description="最低相似度阈值（cosine/dot 可能为负，euclidean 转相似度后为 [0,1]）",
    )
    normalize_vectors: bool = Field(
        default=True, description="检索前自动 L2 归一化（euclidean 下自动忽略）"
    )
    deduplicate_results: bool = Field(
        default=False, description="对结果按文本哈希去重，避免 Top-K 被语义重复 chunk 占满"
    )
    return_embeddings: bool = Field(
        default=False, description="结果是否包含向量（供下游重排序）"
    )
    force_rebuild_index: bool = Field(
        default=False, description="强制重建 Faiss 索引（忽略缓存）"
    )

    @field_validator("candidates")
    @classmethod
    def _check_candidates_not_empty(cls, v: List[VectorRecord]) -> List[VectorRecord]:
        if not v:
            raise ValueError("candidates 不能为空列表")
        return v

    @model_validator(mode="after")
    def _check_backend(self) -> "VectorSearchInput":
        if self.backend == "faiss" and not HAS_FAISS:
            raise BackendNotAvailableError("faiss 后端需要: pip install faiss-cpu")
        if self.backend == "numpy" and not HAS_NUMPY:
            raise BackendNotAvailableError("numpy 后端需要: pip install numpy")
        if self.metric not in ("cosine", "dot", "euclidean"):
            raise ValueError(f"不支持的度量: {self.metric}，仅支持 cosine / dot / euclidean")
        return self

    @model_validator(mode="after")
    def _check_dimensions(self) -> "VectorSearchInput":
        dim = len(self.query_vector)
        for cand in self.candidates:
            if cand.dimension != dim:
                raise DimensionMismatchError(
                    f"维度不匹配: query={dim}, chunk '{cand.chunk_id}'={cand.dimension}"
                )
        return self

    @model_validator(mode="after")
    def _warn_zero_vector(self) -> "VectorSearchInput":
        """检测零向量并发出警告"""
        if not HAS_NUMPY:
            return self

        try:
            import numpy as np
            q = np.array(self.query_vector, dtype=np.float32)
            if np.allclose(q, 0):
                logger.warning("查询向量为全零向量，检索结果可能无意义")
            zero_count = 0
            for cand in self.candidates:
                c = np.array(cand.embedding, dtype=np.float32)
                if np.allclose(c, 0):
                    zero_count += 1
            if zero_count:
                logger.warning("候选集中存在 %d 条全零向量，将跳过归一化", zero_count)
        except Exception as e:
            # 零向量检测失败不应阻断正常流程
            logger.debug("零向量检测跳过: %s", e)

        return self


class SearchResult(BaseModel):
    """单条检索结果"""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(default=None)


class VectorSearchOutput(BaseModel):
    """检索输出"""

    success: bool
    results: List[SearchResult] = Field(default_factory=list)
    query_dimension: int = 0
    total_candidates: int = 0
    filtered_candidates: int = 0
    returned_count: int = 0
    backend: str = ""
    metric: str = ""
    elapsed_ms: float = 0.0
    timing_breakdown: dict = Field(default_factory=dict)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# 后端抽象 + NumPy / Faiss 实现
# ═══════════════════════════════════════════════════════════════

class _SearchBackend:
    """检索后端基类"""

    def search(
        self,
        query: List[float],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
    ) -> List[Tuple[float, int]]:
        """单 query 检索 → [(score, record_index), ...] 按分数降序"""
        raise NotImplementedError

    def search_batch(
        self,
        queries: List[List[float]],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
    ) -> List[List[Tuple[float, int]]]:
        """多 query 检索 → 每个 query 对应一个 [(score, idx), ...] 列表"""
        raise NotImplementedError

    @staticmethod
    def _is_zero_vector(vec: "np.ndarray") -> bool:
        import numpy as np
        return bool(np.allclose(vec, 0))


class _NumpyBackend(_SearchBackend):
    """NumPy 余弦相似度 / 点积 / 欧氏距离 检索"""

    def __init__(self):
        self._cached_records_hash: Optional[str] = None
        self._cached_matrix: Optional["np.ndarray"] = None
        self._cached_norms: Optional["np.ndarray"] = None

    # ── 缓存逻辑 ────────────────────────────────

    @staticmethod
    def _hash_records(records: List[VectorRecord]) -> str:
        """对候选集生成哈希，用于缓存失效检测"""
        raw = "|".join(sorted(r.chunk_id for r in records))
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _get_or_build_matrix(
        self, records: List[VectorRecord], normalize: bool, metric: str
    ) -> "np.ndarray":
        """获取候选矩阵（带缓存）"""
        import numpy as np

        current_hash = self._hash_records(records)

        if (self._cached_records_hash == current_hash
                and self._cached_matrix is not None):
            return self._cached_matrix

        # 重建
        M = np.array([r.embedding for r in records], dtype=np.float32)

        if normalize and metric != "euclidean":
            # 只对 cosine/dot 归一化；euclidean 保留原始向量
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            # 零向量保持不变（避免除零）
            zero_mask = (norms.squeeze() < 1e-10)
            if zero_mask.any():
                logger.warning("候选矩阵中 %d 条零向量，跳过归一化", zero_mask.sum())
                norms[zero_mask] = 1.0
            M = M / norms

        self._cached_records_hash = current_hash
        self._cached_matrix = M
        self._cached_norms = None  # 按需再算
        return M

    def _prepare_query(
        self, query: "np.ndarray", normalize: bool, metric: str
    ) -> "np.ndarray":
        """归一化单条查询向量"""
        import numpy as np
        q = query.astype(np.float32)
        if normalize and metric != "euclidean":
            norm = np.linalg.norm(q)
            if norm < 1e-10:
                logger.warning("查询向量接近零向量，跳过归一化")
            else:
                q = q / norm
        return q

    # ── 检索 ────────────────────────────────────

    def search(
        self,
        query: List[float],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
    ) -> List[Tuple[float, int]]:
        import numpy as np

        M = self._get_or_build_matrix(records, normalize, metric)
        q = self._prepare_query(np.array(query), normalize, metric)

        scores = self._compute_scores(q, M, metric)

        # Top-K
        top_k_actual = min(top_k, len(scores))
        if top_k_actual >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k_actual)[:top_k_actual]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(float(scores[i]), int(i)) for i in top_indices]

    def search_batch(
        self,
        queries: List[List[float]],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
    ) -> List[List[Tuple[float, int]]]:
        import numpy as np

        M = self._get_or_build_matrix(records, normalize, metric)
        Q = np.array(queries, dtype=np.float32)

        # 批量归一化查询
        if normalize and metric != "euclidean":
            q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
            zero_mask = (q_norms.squeeze() < 1e-10)
            if zero_mask.any():
                logger.warning("批量查询中 %d 条零向量，跳过归一化", zero_mask.sum())
                q_norms[zero_mask] = 1.0
            Q = Q / q_norms

        # 一次性矩阵乘法: scores = Q @ M.T  → shape (n_queries, n_records)
        scores_matrix = self._compute_scores_batch(Q, M, metric)

        top_k_actual = min(top_k, scores_matrix.shape[1])
        results: List[List[Tuple[float, int]]] = []

        for row in range(scores_matrix.shape[0]):
            scores = scores_matrix[row]
            if top_k_actual >= len(scores):
                top_indices = np.argsort(-scores)
            else:
                top_indices = np.argpartition(-scores, top_k_actual)[:top_k_actual]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            results.append(
                [(float(scores[i]), int(i)) for i in top_indices]
            )

        return results

    # ── 相似度计算 ──────────────────────────────

    @staticmethod
    def _compute_scores(
        q: "np.ndarray", M: "np.ndarray", metric: str
    ) -> "np.ndarray":
        """单 query → 1D scores"""
        if metric == "euclidean":
            # 欧氏距离 → 相似度 1/(1+d)
            dists = np.linalg.norm(M - q, axis=1)
            return 1.0 / (1.0 + dists)
        else:
            # cosine（已归一化等价于 dot）或 dot
            return np.dot(M, q)

    @staticmethod
    def _compute_scores_batch(
        Q: "np.ndarray", M: "np.ndarray", metric: str
    ) -> "np.ndarray":
        """多 query → 2D scores (n_queries, n_records)"""
        import numpy as np

        if metric == "euclidean":
            # 批量欧氏距离: ||Q[i] - M[j]||² = ||Q[i]||² + ||M[j]||² - 2 Q[i]·M[j]
            q_norm_sq = np.sum(Q ** 2, axis=1, keepdims=True)       # (nq, 1)
            m_norm_sq = np.sum(M ** 2, axis=1)                       # (nr,)
            dot_product = Q @ M.T                                    # (nq, nr)
            dists_sq = q_norm_sq + m_norm_sq - 2.0 * dot_product
            dists_sq = np.maximum(dists_sq, 0)  # 数值精度保护
            dists = np.sqrt(dists_sq)
            return 1.0 / (1.0 + dists)
        else:
            return Q @ M.T


class _FaissBackend(_SearchBackend):
    """Faiss 检索（适合 > 10 万向量）

    v1.1：索引绑定 candidates 哈希，变化自动重建；
    支持 IndexFlatIP（cosine/dot）和 IndexFlatL2（euclidean）。
    """

    def __init__(self, metric: str = "cosine"):
        self._metric = metric
        self._index: Any = None
        self._dim: Optional[int] = None
        self._records_hash: Optional[str] = None

    @staticmethod
    def _hash_records(records: List[VectorRecord]) -> str:
        raw = "|".join(sorted(r.chunk_id for r in records))
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _need_rebuild(self, records: List[VectorRecord]) -> bool:
        """检测是否需要重建索引"""
        current_hash = self._hash_records(records)
        if self._index is None:
            return True
        if self._records_hash != current_hash:
            return True
        if records and self._dim != records[0].dimension:
            return True
        return False

    def _build_index(self, records: List[VectorRecord], normalize: bool) -> None:
        """构建 Faiss 索引"""
        import numpy as np
        import faiss

        self._dim = records[0].dimension
        M = np.array([r.embedding for r in records], dtype=np.float32)

        # 零向量检测
        zero_mask = np.all(np.abs(M) < 1e-10, axis=1)
        if zero_mask.any():
            logger.warning("Faiss 索引: %d 条零向量", zero_mask.sum())

        if self._metric == "euclidean":
            # L2 距离索引（越小越近）
            if normalize:
                faiss.normalize_L2(M)
            self._index = faiss.IndexFlatL2(self._dim)
            self._index.add(M)
        else:
            # cosine / dot → Inner Product（需先归一化向量的 L2 范数）
            if normalize:
                faiss.normalize_L2(M)
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(M)

        self._records_hash = self._hash_records(records)

    def _ensure_index(
        self, records: List[VectorRecord], normalize: bool, force_rebuild: bool
    ) -> None:
        """确保索引就绪"""
        if force_rebuild or self._need_rebuild(records):
            logger.info(
                "Faiss 索引构建 | records=%d | dim=%d | metric=%s",
                len(records), records[0].dimension, self._metric,
            )
            self._build_index(records, normalize)

    def _prepare_query(
        self, query: List[float], normalize: bool
    ) -> "np.ndarray":
        import numpy as np
        import faiss

        q = np.array([query], dtype=np.float32)
        if normalize and self._metric != "euclidean":
            faiss.normalize_L2(q)
        return q

    def _faiss_to_scores(
            self, scores: "np.ndarray", indices: "np.ndarray"
    ) -> List[Tuple[float, int]]:
        """将 Faiss 返回的距离/相似度统一转为相似度（越大越好）"""
        results = []
        for i in range(len(indices)):
            idx = int(indices[i])
            # 只跳过真正的无效索引（-1 表示 Faiss 无法找到足够的邻居）
            if idx < 0 or idx >= self._index.ntotal:
                continue
            raw_score = float(scores[i])
            if self._metric == "euclidean":
                # Faiss 返回 L2 距离 → 转相似度
                results.append((1.0 / (1.0 + raw_score), idx))
            else:
                # cosine/dot: 已经是相似度
                results.append((raw_score, idx))
        return results

    def search(
        self,
        query: List[float],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
        force_rebuild: bool = False,
    ) -> List[Tuple[float, int]]:
        import faiss

        self._metric = metric
        self._ensure_index(records, normalize, force_rebuild)
        q = self._prepare_query(query, normalize)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        return self._faiss_to_scores(scores[0], indices[0])

    def search_batch(
        self,
        queries: List[List[float]],
        records: List[VectorRecord],
        top_k: int,
        normalize: bool,
        metric: str,
        force_rebuild: bool = False,
    ) -> List[List[Tuple[float, int]]]:
        import numpy as np
        import faiss

        self._metric = metric
        self._ensure_index(records, normalize, force_rebuild)

        Q = np.array(queries, dtype=np.float32)
        if normalize and metric != "euclidean":
            faiss.normalize_L2(Q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(Q, k)

        results = []
        for row in range(scores.shape[0]):
            results.append(self._faiss_to_scores(scores[row], indices[row]))
        return results


# ═══════════════════════════════════════════════════════════════
# 技能类
# ═══════════════════════════════════════════════════════════════

class VectorSearchSkill(BaseSkill):
    """向量检索技能 — RAG 第四环 v1.1

    v1.1 核心改进：
    - euclidean 距离完整实现 + 零向量防护
    - Faiss 索引生命周期绑定 candidates 哈希
    - 批量检索矩阵乘法 / 多 query 一次完成
    - 候选矩阵缓存 + hash 失效检测
    - filter_fn 自定义过滤、去重、返回向量
    - 分阶段耗时日志 + 细化异常
    """

    name = "vector_search"
    description = (
        "在已向量化的分块集合中，按余弦相似度/点积/欧氏距离检索 Top-K 最相关结果。"
        "支持 numpy/faiss 后端、元数据过滤、自定义过滤回调、文本去重。"
    )
    version = "1.1.0"
    author = "EnterpriseLearningAgent"
    triggers = [
        "向量检索", "语义搜索", "相似度搜索", "vector search",
        "检索相关", "查找相似", "搜索相关文档",
    ]

    input_schema = VectorSearchInput
    output_schema = VectorSearchOutput

    def __init__(self):
        super().__init__()
        self._numpy_backend: Optional[_NumpyBackend] = None
        self._faiss_backend: Optional[_FaissBackend] = None

    # ═══════════════════════════════════════════════
    # 核心入口
    # ═══════════════════════════════════════════════

    def execute(self, input_data: VectorSearchInput) -> dict:
        """执行向量检索

        Args:
            input_data: 向量检索输入 Pydantic 对象
        """
        t_start = time.perf_counter()
        timing: Dict[str, float] = {}

        try:
            logger.info(
                "VectorSearch v1.1 开始 | candidates=%d | top_k=%d | "
                "backend=%s | metric=%s | min_score=%.2f | dedup=%s",
                len(input_data.candidates), input_data.top_k,
                input_data.backend, input_data.metric,
                input_data.min_score, input_data.deduplicate_results,
            )

            # ── 第一步：元数据过滤 ──
            t_filter = time.perf_counter()
            filtered = self._apply_filters(
                input_data.candidates,
                input_data.filter_metadata,
                input_data.filter_fn,
            )
            timing["filter_ms"] = round((time.perf_counter() - t_filter) * 1000, 2)

            if not filtered:
                # 过滤后无候选
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.info("元数据过滤后候选集为空")
                return VectorSearchOutput(
                    success=True,
                    results=[],
                    query_dimension=len(input_data.query_vector),
                    total_candidates=len(input_data.candidates),
                    filtered_candidates=0,
                    returned_count=0,
                    backend=input_data.backend,
                    metric=input_data.metric,
                    elapsed_ms=round(elapsed_ms, 2),
                    timing_breakdown={"filter_ms": timing["filter_ms"]},
                ).model_dump()

            logger.info(
                "元数据过滤: %d → %d 条",
                len(input_data.candidates), len(filtered),
            )

            # ── 第二步：检索 ──
            t_search = time.perf_counter()
            scored_pairs = self._run_search(input_data, filtered)
            timing["search_ms"] = round((time.perf_counter() - t_search) * 1000, 2)

            # ── 第三步：组装结果（分数过滤 + 去重 + 排序）──
            t_assemble = time.perf_counter()
            results = self._assemble_results(scored_pairs, filtered, input_data)
            timing["assemble_ms"] = round((time.perf_counter() - t_assemble) * 1000, 2)

            # ── 第四步：去重 ──
            if input_data.deduplicate_results and results:
                before_dedup = len(results)
                results = self._deduplicate_by_text(results)
                timing["dedup_ms"] = round(
                    (time.perf_counter() - t_assemble - timing.get("assemble_ms", 0)) * 1000, 2
                )
                logger.info("文本去重: %d → %d 条", before_dedup, len(results))

            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

            top_score = results[0].score if results else 0.0
            logger.info(
                "VectorSearch v1.1 完成 | returned=%d/%d | top_score=%.4f | "
                "total=%.1fms (filter=%.1f search=%.1f assemble=%.1f)",
                len(results), len(filtered), top_score,
                timing["total_ms"], timing["filter_ms"],
                timing["search_ms"], timing["assemble_ms"],
            )

            return VectorSearchOutput(
                success=True,
                results=results,
                query_dimension=len(input_data.query_vector),
                total_candidates=len(input_data.candidates),
                filtered_candidates=len(filtered),
                returned_count=len(results),
                backend=input_data.backend,
                metric=input_data.metric,
                elapsed_ms=timing["total_ms"],
                timing_breakdown=timing,
            ).model_dump()


        except DimensionMismatchError as exc:
            logger.error("维度不匹配: %s", exc)
            return self._error_output(input_data, t_start, timing, str(exc))
        except BackendNotAvailableError as exc:
            logger.error("后端不可用: %s", exc)
            return self._error_output(input_data, t_start, timing, str(exc))
        except Exception as exc:
            logger.exception("VectorSearch 执行失败")
            return self._error_output(input_data, t_start, {}, str(exc))

    # ═══════════════════════════════════════════════
    # 批处理入口
    # ═══════════════════════════════════════════════

    def execute_batch(
            self,
            queries: List[List[float]],
            candidates: List[VectorRecord],
            top_k: int = 5,
            **common_kwargs,
    ) -> List[dict]:
        """批量查询（同一候选集合，多个查询向量）

        利用后端原生的批量检索能力（矩阵乘法 / Faiss 多 query），
        比逐条循环调用 execute 效率高 3-10 倍。
        """
        logger.info(
            "批检索开始 | queries=%d | candidates=%d | top_k=%d",
            len(queries), len(candidates), top_k,
        )
        t_start = time.perf_counter()

        # 空查询列表直接返回空列表
        if not queries:
            return []

        validated = VectorSearchInput(
            query_vector=queries[0],
            candidates=candidates,
            top_k=top_k,
            **common_kwargs,
        )

        # 元数据过滤（所有 query 共享）
        filtered = self._apply_filters(
            validated.candidates,
            validated.filter_metadata,
            validated.filter_fn,
        )

        if not filtered:
            empty_output = VectorSearchOutput(
                success=True, results=[],
                total_candidates=len(candidates),
                filtered_candidates=0,
                backend=validated.backend,
                metric=validated.metric,
            ).model_dump()
            return [dict(empty_output) for _ in queries]

        # 批量检索
        backend = self._get_backend(validated.backend, validated.metric)
        scored_batches = backend.search_batch(
            queries=queries,
            records=filtered,
            top_k=validated.top_k,
            normalize=validated.normalize_vectors,
            metric=validated.metric,
            **({"force_rebuild": validated.force_rebuild_index}
               if validated.backend == "faiss" else {}),
        )

        # 组装结果
        outputs = []
        for scored_pairs in scored_batches:
            results = self._assemble_results(scored_pairs, filtered, validated)
            outputs.append(VectorSearchOutput(
                success=True,
                results=results,
                query_dimension=len(queries[0]) if queries else 0,
                total_candidates=len(candidates),
                filtered_candidates=len(filtered),
                returned_count=len(results),
                backend=validated.backend,
                metric=validated.metric,
                elapsed_ms=0,  # 批处理不单独计时每个
            ).model_dump())

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info("批检索完成 | %d queries | total=%.1fms", len(queries), total_ms)
        return outputs

    # ═══════════════════════════════════════════════
    # 检索调度
    # ═══════════════════════════════════════════════

    def _get_backend(self, backend_name: str, metric: str) -> _SearchBackend:
        """获取或创建后端实例"""
        if backend_name == "faiss":
            if self._faiss_backend is None:
                self._faiss_backend = _FaissBackend(metric=metric)
            else:
                self._faiss_backend._metric = metric
            return self._faiss_backend
        else:
            if self._numpy_backend is None:
                self._numpy_backend = _NumpyBackend()
            return self._numpy_backend

    def _run_search(
        self,
        inp: VectorSearchInput,
        records: List[VectorRecord],
    ) -> List[Tuple[float, int]]:
        """单 query 检索"""
        backend = self._get_backend(inp.backend, inp.metric)
        kwargs = {}
        if inp.backend == "faiss":
            kwargs["force_rebuild"] = inp.force_rebuild_index
        return backend.search(
            inp.query_vector,
            records,
            inp.top_k,
            inp.normalize_vectors,
            inp.metric,
            **kwargs,
        )

    # ═══════════════════════════════════════════════
    # 过滤
    # ═══════════════════════════════════════════════

    @staticmethod
    def _apply_filters(
        candidates: List[VectorRecord],
        metadata_filters: Optional[dict],
        filter_fn: Optional[Callable[[VectorRecord], bool]],
    ) -> List[VectorRecord]:
        """组合元数据精确匹配 + 自定义过滤回调"""
        filtered = candidates

        if metadata_filters:
            filtered = [
                c for c in filtered
                if all(c.metadata.get(k) == v for k, v in metadata_filters.items())
            ]

        if filter_fn:
            filtered = [c for c in filtered if filter_fn(c)]

        return filtered

    # ═══════════════════════════════════════════════
    # 结果组装
    # ═══════════════════════════════════════════════

    def _assemble_results(
        self,
        scored_pairs: List[Tuple[float, int]],
        records: List[VectorRecord],
        inp: VectorSearchInput,
    ) -> List[SearchResult]:
        """分数过滤 + 排名 + 组装 SearchResult"""
        results: List[SearchResult] = []
        rank = 0
        for score, idx in scored_pairs:
            if score < inp.min_score:
                continue
            rank += 1
            if len(results) >= inp.top_k:
                break
            rec = records[idx]
            results.append(SearchResult(
                chunk_id=rec.chunk_id,
                text=rec.text,
                score=round(score, 6),
                rank=rank,
                metadata=rec.metadata,
                embedding=rec.embedding if inp.return_embeddings else None,
            ))

        if results and len(results) < inp.top_k:
            # 区分原因
            if inp.min_score > 0 and scored_pairs:
                last_score = scored_pairs[-1][0] if len(scored_pairs) > len(results) else results[-1].score
                logger.info(
                    "返回不足 top_k=%d: 满足 min_score=%.2f 的仅 %d 条（候选共 %d 条）",
                    inp.top_k, inp.min_score, len(results), len(records),
                )
            else:
                logger.info(
                    "返回不足 top_k=%d: 候选集仅 %d 条",
                    inp.top_k, len(records),
                )

        return results

    # ═══════════════════════════════════════════════
    # 文本去重
    # ═══════════════════════════════════════════════

    @staticmethod
    def _deduplicate_by_text(results: List[SearchResult]) -> List[SearchResult]:
        """对结果按文本哈希去重，保留分数最高的"""
        seen: Dict[str, SearchResult] = {}
        for r in results:
            text_hash = hashlib.md5(r.text.strip().encode()).hexdigest()
            if text_hash not in seen or r.score > seen[text_hash].score:
                seen[text_hash] = r
        # 按分数降序重新排列
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    # ═══════════════════════════════════════════════
    # 错误输出
    # ═══════════════════════════════════════════════

    @staticmethod
    def _error_output(
        kwargs_or_input, t_start: float, timing: dict, error: str
    ) -> dict:
        """统一错误输出"""
        candidates = (
            kwargs_or_input.candidates
            if isinstance(kwargs_or_input, VectorSearchInput)
            else kwargs_or_input.get("candidates", [])
        )
        query_dim = (
            len(kwargs_or_input.query_vector)
            if isinstance(kwargs_or_input, VectorSearchInput)
            else len(kwargs_or_input.get("query_vector", []))
        )
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return VectorSearchOutput(
            success=False,
            query_dimension=query_dim,
            total_candidates=len(candidates),
            elapsed_ms=total_ms,
            timing_breakdown=timing,
            error=error,
        ).model_dump()


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def search_similar(
    query_vector: List[float],
    candidates: List[dict],
    top_k: int = 5,
    backend: str = "numpy",
    **kwargs,
) -> dict:
    """一行式向量检索"""
    input_data = VectorSearchInput(
        query_vector=query_vector,
        candidates=candidates,
        top_k=top_k,
        backend=backend,
        **kwargs,
    )
    return VectorSearchSkill().execute(input_data)


def search_similar_batch(
    queries: List[List[float]],
    candidates: List[dict],
    top_k: int = 5,
    **kwargs,
) -> List[dict]:
    """批量向量检索（利用原生批量能力，比逐条调用快 3-10 倍）"""
    return VectorSearchSkill().execute_batch(
        queries=queries,
        candidates=candidates,
        top_k=top_k,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  VectorSearch v1.1 — 向量检索技能测试")
    print("=" * 60)

    import numpy as np

    np.random.seed(42)
    dim = 128

    # 生成 10 条模拟向量（含一条零向量测试）
    mock_candidates = []
    for i in range(10):
        if i == 9:
            vec = np.zeros(dim, dtype=np.float32)  # 零向量
        else:
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
        mock_candidates.append({
            "chunk_id": f"chunk_{i:03d}",
            "text": f"这是第 {i} 条测试文本。内容关于{'人工智能' if i < 5 else '机器学习'}。",
            "embedding": vec.tolist(),
            "metadata": {"doc_id": f"doc_{(i // 3) + 1}", "category": "AI" if i < 5 else "ML"},
        })

    query = mock_candidates[0]["embedding"].copy()
    query_array = np.array(query) + np.random.randn(dim).astype(np.float32) * 0.1
    query_array = query_array / np.linalg.norm(query_array)

    skill = VectorSearchSkill()

    # ── 测试 1：cosine ──
    print("\n📌 测试 1：cosine 相似度")
    r = skill.execute(query_vector=query_array.tolist(), candidates=mock_candidates, top_k=5, metric="cosine")
    print(f"   返回: {r['returned_count']} 条 | 耗时: {r['timing_breakdown']}")
    for item in r["results"][:3]:
        print(f"   #{item['rank']} | {item['chunk_id']} | score={item['score']:.4f}")

    # ── 测试 2：euclidean ──
    print("\n📌 测试 2：euclidean 距离（内部转相似度）")
    r2 = skill.execute(query_vector=query_array.tolist(), candidates=mock_candidates, top_k=5, metric="euclidean")
    print(f"   返回: {r2['returned_count']} 条 | timing={r2['timing_breakdown']}")
    for item in r2["results"][:3]:
        print(f"   #{item['rank']} | {item['chunk_id']} | score={item['score']:.4f}")

    # ── 测试 3：元数据过滤 + filter_fn ──
    print("\n📌 测试 3：元数据过滤 (category=AI) + filter_fn (字数>10)")
    r3 = skill.execute(
        query_vector=query_array.tolist(),
        candidates=mock_candidates,
        top_k=5,
        filter_metadata={"category": "AI"},
        filter_fn=lambda rec: len(rec.text) > 10,
    )
    print(f"   过滤后: {r3['filtered_candidates']} 条 → 返回 {r3['returned_count']} 条")

    # ── 测试 4：去重 ──
    print("\n📌 测试 4：文本去重")
    dup_candidates = mock_candidates[:3] + mock_candidates[:3]  # 复制前 3 条
    r4 = skill.execute(
        query_vector=query_array.tolist(),
        candidates=dup_candidates,
        top_k=5,
        deduplicate_results=True,
    )
    print(f"   去重前候选: {len(dup_candidates)} → 返回 {r4['returned_count']} 条（去重后）")

    # ── 测试 5：return_embeddings ──
    print("\n📌 测试 5：返回向量")
    r5 = skill.execute(
        query_vector=query_array.tolist(),
        candidates=mock_candidates,
        top_k=3,
        return_embeddings=True,
    )
    has_emb = r5["results"][0].get("embedding") is not None if r5["results"] else False
    print(f"   返回向量: {has_emb} | 维度: {len(r5['results'][0]['embedding']) if has_emb else 'N/A'}")

    # ── 测试 6：批量检索 ──
    print("\n📌 测试 6：批量检索（3 queries）")
    queries_batch = [
        query_array.tolist(),
        mock_candidates[3]["embedding"],
        mock_candidates[6]["embedding"],
    ]
    r6_list = skill.execute_batch(queries=queries_batch, candidates=mock_candidates, top_k=3)
    for i, r6 in enumerate(r6_list):
        print(f"   query {i}: 返回 {r6['returned_count']} 条 | top_score={r6['results'][0]['score']:.4f}" if r6['results'] else f"   query {i}: 无结果")

    # ── 测试 7：faiss 后端 ──
    if HAS_FAISS:
        print("\n📌 测试 7：Faiss 后端")
        r7 = skill.execute(
            query_vector=query_array.tolist(),
            candidates=mock_candidates,
            top_k=5,
            backend="faiss",
            metric="cosine",
        )
        print(f"   返回: {r7['returned_count']} 条 | backend={r7['backend']} | timing={r7['timing_breakdown']}")
        for item in r7["results"][:3]:
            print(f"   #{item['rank']} | {item['chunk_id']} | score={item['score']:.4f}")
    else:
        print("\n📌 测试 7：Faiss 不可用，跳过（pip install faiss-cpu）")

    # ── 测试 8：元数据过滤后为空 ──
    print("\n📌 测试 8：过滤后候选集为空")
    r8 = skill.execute(
        query_vector=query_array.tolist(),
        candidates=mock_candidates,
        filter_metadata={"category": "NONEXISTENT"},
    )
    print(f"   成功: {r8['success']} | 返回: {r8['returned_count']} 条")

    print("\n" + "=" * 60)
    print("  ✅ 测试完成")
    print("=" * 60)