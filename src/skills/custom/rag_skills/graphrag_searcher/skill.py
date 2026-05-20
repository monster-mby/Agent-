"""
GraphRAG 知识图谱检索技能（生产优化版）

优化点清单：
  P0: faiss 向量搜索替代全量遍历（带 numpy 降级）
  P0: tiktoken 精确 Token 截断（带字符数降级）
  P0: 缓存失效机制（基于文件 mtime）+ 线程安全锁
  P1: tenacity LLM 重试（带降级）
  P1: 客户端构建失败显式告警日志
  P1: 嵌入客户端接口适配（处理单数/复数签名）
  P1: 客户端解耦 → 模块级 inject 函数
  P1: execute 入口查询日志
  P2: jinja2 Prompt 模板化（带字符串降级）
  P2: Pydantic 输出模型
  P2: jieba 中文分词 fallback
  P3: itertuples 替代 iterrows
  P3: 社区倒排索引预构建

完全匹配 BaseSkill 接口：
  - 无 setup()，配置从环境变量 + inject 函数读取
  - execute(**kwargs) → Dict[str, Any]，同步方法
  - 模块级 _searcher_cache 单例（带失效检测）
  - 通过 SkillManager.call("graphrag_searcher", query="...", mode="local") 调用
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)



# ============================================================
# Pydantic 输出模型（可选依赖）
# ============================================================
try:
    from pydantic import BaseModel, Field

    class SearchSource(BaseModel):
        type: str = Field(..., description="来源类型: entity / community")
        name: str = Field(..., description="来源名称")

    class GraphRAGSearcherOutput(BaseModel):
        answer: str = Field(default="", description="回答文本")
        sources: List[SearchSource] = Field(default_factory=list, description="信息来源列表")
        mode: str = Field(default="local", description="检索模式")
        entity_count: int = Field(default=0, description="匹配到的实体数")
        community_count: int = Field(default=0, description="匹配到的社区数")
        error: Optional[str] = Field(default=None, description="错误信息")


    class GraphRAGSearcherInput(BaseModel):
        """GraphRAG 知识图谱检索的输入模型"""
        query: str = Field(..., description="用户问题")
        mode: str = Field(
            default="local",
            description='检索模式："local"（实体级精确检索）/ "global"（社区级全局摘要）/ "hybrid"（融合两种视角）'
        )
        top_k_entities: int = Field(
            default=10,
            description="Local 模式取前 k 个实体"
        )
        top_k_communities: int = Field(
            default=5,
            description="Global 模式取前 k 个社区"
        )
        max_context_tokens: int = Field(
            default=6000,
            description="上下文最大 token 数"
        )

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    SearchSource = None
    GraphRAGSearcherOutput = None




# ============================================================
# 可选依赖检测 & 降级
# ============================================================

def _check_faiss():
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _check_tiktoken():
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


def _check_tenacity():
    try:
        import tenacity  # noqa: F401
        return True
    except ImportError:
        return False


def _check_jieba():
    try:
        import jieba  # noqa: F401
        return True
    except ImportError:
        return False


HAS_FAISS = _check_faiss()
HAS_TIKTOKEN = _check_tiktoken()
HAS_TENACITY = _check_tenacity()
HAS_JIEBA = _check_jieba()

if HAS_FAISS:
    import faiss
if HAS_TIKTOKEN:
    import tiktoken
if HAS_TENACITY:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger.info(
    "GraphRAG Searcher 依赖检测: faiss=%s, tiktoken=%s, tenacity=%s, jieba=%s",
    HAS_FAISS, HAS_TIKTOKEN, HAS_TENACITY, HAS_JIEBA,
)


# ============================================================
# Prompt 模板（jinja2 优先，字符串降级）
# ============================================================

_LOCAL_SEARCH_TEMPLATE = """你是一个知识图谱助手。请基于以下知识图谱上下文回答用户问题。
如果上下文中没有相关信息，请诚实说明「当前知识库中未找到相关信息」。

## 知识图谱上下文
{{ context }}

## 用户问题
{{ query }}

## 回答要求
1. 优先基于上下文回答，引用具体的实体和关系
2. 如果上下文不足，明确指出缺少哪类信息
3. 回答简洁清晰"""

_GLOBAL_SEARCH_TEMPLATE = """你是一个知识图谱助手。以下是文档集中多个主题社区的摘要信息。
请基于这些摘要，从全局视角综合回答用户问题。

## 社区摘要
{{ context }}

## 用户问题
{{ query }}

## 回答要求
1. 综合不同社区的视角
2. 指出各主题之间的关联和整体图景
3. 回答要有宏观概括力"""

_HYBRID_MERGE_TEMPLATE = """请综合以下两个视角来回答用户问题，形成完整且有深度的回答。

## 视角 1（实体级细节）
{{ local_answer }}

## 视角 2（全局社区视角）
{{ global_answer }}

## 用户问题
{{ query }}

请给出一个融合两种视角的完整回答。"""


def _render_prompt(template_str: str, **kwargs) -> str:
    """渲染 Prompt 模板（优先 jinja2，否则用 str.replace）"""
    try:
        from jinja2 import Template
        return Template(template_str).render(**kwargs)
    except ImportError:
        result = template_str
        for key, val in kwargs.items():
            result = result.replace("{{ " + key + " }}", str(val))
        return result


# ============================================================
# 客户端注入（模块级，解耦硬编码）
# ============================================================

_inject_llm_client = None
_inject_embedding_client = None
_inject_lock = threading.Lock()


def configure_graphrag_searcher(
    llm_client: Any = None,
    embedding_client: Any = None,
) -> None:
    """
    注入外部客户端。在应用启动时调用一次即可。

    用法：
        from src.skills.custom.rag_skills.graphrag_searcher import skill
        skill.configure_graphrag_searcher(
            llm_client=my_llm_instance,
            embedding_client=my_embedding_instance,
        )

    Args:
        llm_client: 实现了 .generate(prompt) → str 的对象
        embedding_client: 实现了 .execute(texts=[...]) → dict 或 .embed(text) → list 的对象
    """
    global _inject_llm_client, _inject_embedding_client
    with _inject_lock:
        if llm_client is not None:
            _inject_llm_client = llm_client
            logger.info("GraphRAG Searcher: LLM 客户端已注入 (%s)", type(llm_client).__name__)
        if embedding_client is not None:
            _inject_embedding_client = embedding_client
            logger.info("GraphRAG Searcher: Embedding 客户端已注入 (%s)", type(embedding_client).__name__)


def _clear_injected_clients():
    """清空注入的客户端（主要用于测试）"""
    global _inject_llm_client, _inject_embedding_client
    with _inject_lock:
        _inject_llm_client = None
        _inject_embedding_client = None


# ============================================================
# 内部引擎（模块级缓存 + 自动失效）
# ============================================================

_searcher_cache: Optional["_GraphRAGEngine"] = None
_cache_lock = threading.Lock()


class _GraphRAGEngine:
    """
    内部引擎：持有 faiss 索引、parquet 引用、LLM/Embedding 客户端。

    缓存策略：
      - 模块级单例，避免每次调用重新加载
      - 每次查询前检查 artifacts 目录 mtime，若变更则自动重建
      - 线程安全：读写均加锁
    """

    def __init__(self, graphrag_root: Path):
        import pandas as pd

        self.graphrag_root = graphrag_root
        self.artifacts_dir = graphrag_root / "output" / "artifacts"

        if not self._index_exists():
            raise FileNotFoundError(
                f"GraphRAG 索引不存在于 {self.artifacts_dir}。"
                f"请先用 GraphRAGIndexerSkill 构建索引，或执行: graphrag index --root {graphrag_root}"
            )

        # 记录索引的 mtime，用于后续失效检测
        self._artifacts_mtime = self._get_artifacts_mtime()

        # 加载数据（仅加载必要列）
        logger.info("加载 GraphRAG 索引产物...")
        self.entities = pd.read_parquet(
            self.artifacts_dir / "entities.parquet",
            columns=["title", "type", "description", "embedding"],
        )
        self.relationships = pd.read_parquet(
            self.artifacts_dir / "relationships.parquet",
            columns=["source", "target", "description"],
        )
        self.communities = pd.read_parquet(
            self.artifacts_dir / "communities.parquet",
            columns=["community", "title", "embedding"],
        )
        self.community_reports = pd.read_parquet(
            self.artifacts_dir / "community_reports.parquet",
        )

        # 预处理：实体名统一用 title 列
        if "title" not in self.entities.columns and hasattr(self.entities, "columns"):
            # 兼容旧索引格式
            if "name" in self.entities.columns:
                self.entities["title"] = self.entities["name"]
            else:
                self.entities["title"] = ""

        # 构建 faiss 实体索引
        self._entity_index, self._entity_names = self._build_entity_faiss_index()

        # 构建 faiss 社区索引
        self._community_index, self._community_ids = self._build_community_faiss_index()

        # 预构建社区倒排索引
        self._entity_to_communities = self._build_entity_community_index()

        # LLM / Embedding 客户端
        self.llm_client = _inject_llm_client or self._auto_build_llm_client()
        self.embedding_client = _inject_embedding_client or self._auto_build_embedding_client()

        # 如果注入的客户端为 None，记录警告但继续（允许降级到关键词匹配）
        if self.llm_client is None:
            logger.warning("LLM 客户端未配置，LLM 调用将返回提示信息")
        if self.embedding_client is None:
            logger.warning("Embedding 客户端未配置，将仅使用关键词匹配")

        # tiktoken encoder
        self._tokenizer = None
        if HAS_TIKTOKEN:
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("tiktoken encoder 初始化失败，将使用字符数估算")

        logger.info(
            "GraphRAG 引擎初始化完成: %d 实体, %d 关系, %d 社区",
            len(self.entities), len(self.relationships), len(self.communities),
        )

    # ── 索引存在性 ──────────────────────────────────────────

    def _index_exists(self) -> bool:
        success_marker = self.artifacts_dir / "_SUCCESS"
        if not success_marker.exists():
            return False
        required = ["entities.parquet", "relationships.parquet", "communities.parquet",
                     "community_reports.parquet"]
        return all((self.artifacts_dir / f).exists() for f in required)

    def _get_artifacts_mtime(self) -> float:
        """获取 artifacts 目录下最新文件的修改时间"""
        max_mtime = 0.0
        for f in self.artifacts_dir.glob("*.parquet"):
            mtime = f.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        return max_mtime

    def is_stale(self) -> bool:
        """检查索引是否已过期（文件被更新）"""
        return self._get_artifacts_mtime() > self._artifacts_mtime

    # ── Faiss 索引构建 ──────────────────────────────────────

    def _build_entity_faiss_index(self) -> Tuple[Any, np.ndarray]:
        """构建实体的 Faiss 向量索引"""
        embeddings = []
        names = []
        for row in self.entities.itertuples():
            emb = getattr(row, "embedding", None)
            name = getattr(row, "title", "") or getattr(row, "title", "")
            if emb is not None and len(emb) > 0:
                embeddings.append(emb)
                names.append(name)

        if not embeddings:
            logger.warning("未找到任何实体 embedding，将仅使用关键词匹配")
            return None, np.array([])

        embeddings_arr = np.array(embeddings, dtype=np.float32)

        if HAS_FAISS and len(embeddings_arr) > 0:
            dim = embeddings_arr.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner Product（等价于 cosine 如果向量已归一化）
            # Faiss 推荐先 L2 归一化再做 inner product = cosine similarity
            faiss.normalize_L2(embeddings_arr)
            index.add(embeddings_arr)
            logger.debug("Faiss 实体索引构建完成: %d 个向量", len(embeddings_arr))
        else:
            # 降级：存 numpy 数组
            index = embeddings_arr
            if len(embeddings_arr) > 0:
                # L2 归一化
                norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                index = embeddings_arr / norms
            logger.debug("NumPy 实体索引构建完成 (faiss 不可用): %d 个向量", len(embeddings_arr))

        return index, np.array(names)

    def _build_community_faiss_index(self) -> Tuple[Any, np.ndarray]:
        """构建社区的 Faiss 向量索引"""
        embeddings = []
        ids = []
        for row in self.communities.itertuples():
            emb = getattr(row, "embedding", None)
            cid = getattr(row, "community", None)
            if emb is not None and len(emb) > 0:
                embeddings.append(emb)
                ids.append(cid)

        if not embeddings:
            return None, np.array([])

        embeddings_arr = np.array(embeddings, dtype=np.float32)

        if HAS_FAISS and len(embeddings_arr) > 0:
            dim = embeddings_arr.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings_arr)
            index.add(embeddings_arr)
        else:
            index = embeddings_arr
            if len(embeddings_arr) > 0:
                norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                index = embeddings_arr / norms

        return index, np.array(ids)

    # ── 社区倒排索引 ────────────────────────────────────────

    def _build_entity_community_index(self) -> Dict[str, List]:
        """构建 实体名 → 社区ID列表 的倒排索引"""
        index: Dict[str, List] = {}
        for row in self.community_reports.itertuples():
            content = str(getattr(row, "full_content", ""))
            cid = getattr(row, "community", None)
            # 快速匹配：检查实体名是否出现在报告内容中
            for entity_name in self._entity_names:
                if entity_name and entity_name in content:
                    index.setdefault(str(entity_name), []).append(cid)
        logger.debug("社区倒排索引构建完成: %d 个实体有社区归属", len(index))
        return index

    # ── 客户端自动构建（降级兜底） ──────────────────────────

    def _auto_build_llm_client(self):
        """尝试自动构建 LLM 客户端（仅当未注入时）"""
        try:
            from src.core.model_client import ModelClient
            client = ModelClient()
            logger.info("LLM 客户端自动构建: ModelClient")
            return client
        except Exception as e:
            logger.warning("LLM 客户端自动构建失败，将无法生成回答: %s", e)
            return None

    def _auto_build_embedding_client(self):
        """尝试自动构建嵌入客户端（仅当未注入时）"""
        try:
            from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
            client = TextEmbedderSkill()
            logger.info("Embedding 客户端自动构建: TextEmbeddingSkill")
            return client
        except Exception as e:
            logger.warning("Embedding 客户端自动构建失败，将使用关键词匹配: %s", e)
            return None

    # ==================== Local Search ========================

    def local_search(self, query: str, top_k: int, max_tokens: int) -> dict:
        top_entities = self._find_top_entities(query, top_k)
        context = self._build_local_context(top_entities, max_tokens)

        prompt = _render_prompt(
            _LOCAL_SEARCH_TEMPLATE,
            context=context,
            query=query,
        )

        answer = self._generate(prompt)

        return {
            "answer": answer or "",
            "sources": [{"type": "entity", "name": e["name"]} for e in top_entities],
            "mode": "local",
            "entity_count": len(top_entities),
            "community_count": 0,
            "error": None,
        }

    def _build_local_context(self, top_entities: list, max_tokens: int) -> str:
        parts = [f"## 找到 {len(top_entities)} 个相关实体\n"]
        for e in top_entities[:5]:
            parts.append(f"- **{e['name']}** (类型: {e.get('type', 'N/A')})")
            desc = e.get("description", "")
            if desc:
                parts.append(f"  描述: {desc[:300]}")

        entity_names = {e["name"] for e in top_entities}
        related_rels = []
        for row in self.relationships.itertuples():
            source = str(getattr(row, "source", ""))
            target = str(getattr(row, "target", ""))
            if source in entity_names or target in entity_names:
                related_rels.append({
                    "source": source,
                    "target": target,
                    "description": str(getattr(row, "description", "")),
                })
        if related_rels:
            parts.append(f"\n## 实体间关系 (共 {len(related_rels)} 条)\n")
            for r in related_rels[:10]:
                parts.append(f"- {r['source']} → {r['target']}: {r['description'][:200]}")

        # 用倒排索引快速查社区
        entity_communities = []
        for name in entity_names:
            cids = self._entity_to_communities.get(name, [])
            for cid in cids:
                report_row = self.community_reports[
                    self.community_reports["community"] == cid
                ]
                if not report_row.empty:
                    entity_communities.append(report_row.iloc[0].to_dict())

        if entity_communities:
            parts.append(f"\n## 相关社区摘要\n")
            for c in entity_communities[:3]:
                parts.append(str(c.get("full_content", ""))[:500])

        context = "\n".join(parts)
        return self._truncate_by_tokens(context, max_tokens)

    # ==================== Global Search =======================

    def global_search(self, query: str, top_k: int) -> dict:
        top_communities = self._find_top_communities(query, top_k)
        summaries = self._gather_community_summaries(top_communities)

        context = "\n\n---\n\n".join(summaries)

        prompt = _render_prompt(
            _GLOBAL_SEARCH_TEMPLATE,
            context=context,
            query=query,
        )

        answer = self._generate(prompt)

        return {
            "answer": answer or "",
            "sources": [{"type": "community", "name": c["title"]} for c in top_communities],
            "mode": "global",
            "entity_count": 0,
            "community_count": len(top_communities),
            "error": None,
        }

    # ==================== Hybrid Search =======================

    def hybrid_search(self, query: str, top_k_entities: int,
                      top_k_communities: int, max_tokens: int) -> dict:
        local_result = self.local_search(query, top_k_entities, max_tokens // 2)
        global_result = self.global_search(query, top_k_communities)

        prompt = _render_prompt(
            _HYBRID_MERGE_TEMPLATE,
            local_answer=local_result["answer"],
            global_answer=global_result["answer"],
            query=query,
        )

        answer = self._generate(prompt)

        return {
            "answer": answer or "",
            "sources": local_result["sources"] + global_result["sources"],
            "mode": "hybrid",
            "entity_count": local_result.get("entity_count", 0),
            "community_count": global_result.get("community_count", 0),
            "error": None,
        }

    # ==================== 向量搜索 ============================

    def _find_top_entities(self, query: str, top_k: int) -> List[dict]:
        """用 faiss/numpy 做向量搜索；无嵌入客户端则降级为关键词匹配"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None or self._entity_index is None or len(self._entity_names) == 0:
            return self._fallback_entity_match(query, top_k)

        query_vec = np.array([query_embedding], dtype=np.float32)
        # L2 归一化
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        if HAS_FAISS and hasattr(self._entity_index, 'search'):
            scores, indices = self._entity_index.search(query_vec, min(top_k, len(self._entity_names)))
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self._entity_names):
                    name = str(self._entity_names[idx])
                    entity_row = self.entities[self.entities["title"] == name]
                    entity_type = ""
                    entity_desc = ""
                    if not entity_row.empty:
                        entity_type = str(entity_row.iloc[0].get("type", ""))
                        entity_desc = str(entity_row.iloc[0].get("description", ""))
                    results.append({
                        "name": name,
                        "type": entity_type,
                        "description": entity_desc,
                        "similarity": float(scores[0][i]),
                    })
        else:
            # numpy 降级：批量计算余弦相似度
            if isinstance(self._entity_index, np.ndarray):
                sims = np.dot(self._entity_index, query_vec.T).flatten()
                top_indices = np.argsort(sims)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    if idx < len(self._entity_names):
                        name = str(self._entity_names[idx])
                        results.append({
                            "name": name,
                            "type": "",
                            "description": "",
                            "similarity": float(sims[idx]),
                        })
            else:
                results = []

        return results

    def _find_top_communities(self, query: str, top_k: int) -> List[dict]:
        """用 faiss/numpy 搜索社区"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None or self._community_index is None or len(self._community_ids) == 0:
            results = []
            for row in self.communities.head(top_k).itertuples():
                results.append({
                    "id": getattr(row, "community", ""),
                    "title": getattr(row, "title", f"Community {getattr(row, 'community', 'N/A')}"),
                })
            return results

        query_vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        if HAS_FAISS and hasattr(self._community_index, 'search'):
            scores, indices = self._community_index.search(query_vec, min(top_k, len(self._community_ids)))
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self._community_ids):
                    cid = self._community_ids[idx]
                    comm_row = self.communities[self.communities["community"] == cid]
                    title = f"Community {cid}"
                    if not comm_row.empty:
                        title = str(comm_row.iloc[0].get("title", title))
                    results.append({
                        "id": cid,
                        "title": title,
                        "similarity": float(scores[0][i]),
                    })
        else:
            results = []

        return results

    # ── 降级匹配 ────────────────────────────────────────────

    def _fallback_entity_match(self, query: str, top_k: int) -> List[dict]:
        """无嵌入时的关键词匹配（jieba 中文分词优先）"""
        if HAS_JIEBA:
            import jieba
            keywords = set(jieba.lcut(query.lower()))
        else:
            keywords = set(query.lower().split())

        results = []
        for row in self.entities.itertuples():
            name = str(getattr(row, "title", "") or "")
            if not name:
                continue
            score = sum(1 for kw in keywords if kw in name.lower())
            if score > 0:
                results.append({
                    "name": name,
                    "type": str(getattr(row, "type", "")),
                    "description": str(getattr(row, "description", "")),
                    "similarity": float(score),
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _gather_community_summaries(self, top_communities: List[dict]) -> List[str]:
        summaries = []
        for c in top_communities:
            report = self.community_reports[
                self.community_reports["community"] == c["id"]
            ]
            if not report.empty:
                summaries.append(str(report.iloc[0].get("full_content", "")))
        return summaries

    # ==================== LLM 调用（含 tenacity 重试） =======

    def _generate(self, prompt: str) -> str:
        """调用 LLM 生成回答，失败时自动重试（如果 tenacity 可用）"""
        if self.llm_client is None:
            return "⚠️ 未配置 LLM 客户端，无法生成回答。请调用 configure_graphrag_searcher() 注入 llm_client。"

        if HAS_TENACITY:
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
                reraise=True,
            )
            def _do_generate():
                return self._call_llm_once(prompt)

            try:
                return _do_generate()
            except Exception as e:
                logger.error("LLM 调用失败（已重试 3 次）: %s", e)
                return f"⚠️ LLM 调用失败: {e}"
        else:
            try:
                return self._call_llm_once(prompt)
            except Exception as e:
                logger.error("LLM 调用失败: %s", e)
                return f"⚠️ LLM 调用失败: {e}"

    def _call_llm_once(self, prompt: str) -> str:
        """单次 LLM 调用（适配多种客户端接口）"""
        if hasattr(self.llm_client, "generate"):
            return self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, "complete"):
            return self.llm_client.complete(prompt)
        elif hasattr(self.llm_client, "__call__"):
            return self.llm_client(prompt)
        else:
            return str(self.llm_client)

    # ==================== 嵌入获取 ============================

    def _get_embedding(self, text: str):
        """获取文本嵌入向量（适配多种客户端签名）"""
        if self.embedding_client is None:
            return None

        try:
            # 优先尝试 execute（更通用的接口，且避免 MagicMock 误判）
            if hasattr(self.embedding_client, "execute"):
                try:
                    result = self.embedding_client.execute(texts=[text])
                    if isinstance(result, dict):
                        embeddings = result.get("embeddings", [])
                        if embeddings and len(embeddings) > 0:
                            emb = embeddings[0]
                            if isinstance(emb, list) and len(emb) > 0:
                                return emb
                except (TypeError, AttributeError):
                    pass  # 签名不匹配，继续尝试

                try:
                    result = self.embedding_client.execute(text=text)
                    if isinstance(result, dict):
                        emb = result.get("embedding", result.get("vector"))
                        if isinstance(emb, list) and len(emb) > 0:
                            return emb
                except (TypeError, AttributeError):
                    pass

            # 然后尝试 embed
            if hasattr(self.embedding_client, "embed"):
                result = self.embedding_client.embed(text)
                if isinstance(result, list) and len(result) > 0:
                    return result
        except Exception as e:
            logger.warning("获取 embedding 失败: %s", e)

        return None

    # ==================== Token 截断 ==========================

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """用 tiktoken 精确截断（降级为字符数估算）"""
        if self._tokenizer is not None:
            tokens = self._tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self._tokenizer.decode(truncated_tokens)
        else:
            # 降级：近似 1 token ≈ 2 字符（中英文混合场景）
            max_chars = max_tokens * 2
            if len(text) <= max_chars:
                return text
            return text[:max_chars]


# ============================================================
# 技能类（对外接口）
# ============================================================

class GraphRAGSearcherSkill(BaseSkill):
    """知识图谱检索技能"""

    # ============ 元数据 ============
    name: str = "graphrag_searcher"
    description: str = (
        "使用知识图谱进行实体感知检索。支持三种模式：\n"
        "- local: 实体级精确检索（找实体邻居+关系，适合「A 和 B 什么关系？」）\n"
        "- global: 社区级全局摘要（适合「整个知识库的核心主题是什么？」）\n"
        "- hybrid: 融合 local + global 两种视角"
    )
    version: str = "0.2.0"
    author: str = "dev-team"
    triggers: list[str] = [
        "知识图谱", "图谱检索", "实体关系", "graphrag search",
        "哪些实体", "实体之间", "知识库里有",
    ]
    changelog: list[Dict[str, str]] = [
        {"version": "0.2.0", "change": (
            "P0: faiss 向量搜索 / tiktoken 截断 / 缓存失效 + 线程安全; "
            "P1: tenacity 重试 / 客户端注入 / 嵌入适配 / 查询日志; "
            "P2: jinja2 Prompt 模板 / Pydantic 输出 / jieba 中文分词; "
            "P3: itertuples + 社区倒排索引"
        )},
        {"version": "0.1.0", "change": "初始版本"},
    ]

    # 输入输出模型（供 SkillManager 校验用）
    input_schema = GraphRAGSearcherInput if HAS_PYDANTIC else None
    output_schema = GraphRAGSearcherOutput if HAS_PYDANTIC else None

    # ============ 初始化 ============
    def __init__(self):
        super().__init__()
        global _searcher_cache
        with _cache_lock:
            graphrag_root = Path(os.getenv("GRAPHRAG_ROOT", "./graphrag_data")).resolve()

            # 缓存失效检测
            if _searcher_cache is not None and _searcher_cache.is_stale():
                logger.info("检测到索引已更新，清空缓存并重建引擎")
                _searcher_cache = None

            if _searcher_cache is None:
                _searcher_cache = _GraphRAGEngine(graphrag_root)

            self._engine = _searcher_cache

    # ============ 核心执行 ============
    def execute(
            self,
            input_data: GraphRAGSearcherInput,
    ) -> Dict[str, Any]:
        """
        执行知识图谱检索。

        Args:
            input_data: GraphRAG 知识图谱检索输入 Pydantic 对象

        Returns:
            {"answer": str, "sources": list, "mode": str, "entity_count": int,
             "community_count": int, "error": str | None}
        """
        query = input_data.query
        mode = input_data.mode
        top_k_entities = input_data.top_k_entities
        top_k_communities = input_data.top_k_communities
        max_context_tokens = input_data.max_context_tokens

        logger.info(
            "GraphRAG 检索: mode=%s, top_k_entities=%d, top_k_communities=%d, query=%.100s...",
            mode, top_k_entities, top_k_communities, query,
        )

        # 再次检查缓存是否失效
        with _cache_lock:
            if self._engine.is_stale():
                logger.warning("运行时检测到索引变更，请重新调用（下次请求将自动重建）")
                global _searcher_cache
                _searcher_cache = None

        if mode == "local":
            result = self._engine.local_search(query, top_k_entities, max_context_tokens)
        elif mode == "global":
            result = self._engine.global_search(query, top_k_communities)
        elif mode == "hybrid":
            result = self._engine.hybrid_search(
                query, top_k_entities, top_k_communities, max_context_tokens,
            )
        else:
            result = {
                "answer": "",
                "sources": [],
                "mode": mode,
                "entity_count": 0,
                "community_count": 0,
                "error": f"未知模式: {mode}，可选: local / global / hybrid",
            }

        # Pydantic 输出校验
        if HAS_PYDANTIC and self.output_schema:
            try:
                validated = self.output_schema(**result)
                return validated.model_dump()
            except Exception as e:
                logger.warning("输出校验失败: %s", e)

        return result