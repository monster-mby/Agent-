"""
文档索引流水线 — Loader → Chunker → Embedder → VectorStore

职责：
    1. 管理索引全生命周期的状态（KB + Document）
    2. 协调加载 → 切分 → 向量化 → 存储四阶段
    3. 分批写入 + upsert 保证幂等性
    4. 每阶段独立计时，结构化日志
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

from src.infrastructure.kb_manager import (
    DocumentStatus,
    IndexingStatus,
    update_document_status,
    update_kb_status,
)
from src.infrastructure.vector_store import (
    CollectionNotFoundError,
    VectorStoreError,
    vector_store,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════

class IndexingPipelineError(Exception):
    """索引流水线通用异常基类"""
    pass


class DocumentLoadError(IndexingPipelineError):
    """文档加载阶段失败"""

    def __init__(self, file_path: str, detail: str) -> None:
        super().__init__(f"文档加载失败: file={file_path!r} | {detail}")
        self.file_path = file_path
        self.detail = detail


class ChunkingError(IndexingPipelineError):
    """文档切分阶段失败"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"文档切分失败: {detail}")
        self.detail = detail


class EmbeddingError(IndexingPipelineError):
    """向量化阶段失败"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"向量化失败: {detail}")
        self.detail = detail


class VectorStoreWriteError(IndexingPipelineError):
    """向量存储写入失败"""

    def __init__(self, namespace: str, detail: str) -> None:
        super().__init__(f"向量存储写入失败: namespace={namespace!r} | {detail}")
        self.namespace = namespace
        self.detail = detail


class EmptyChunksError(IndexingPipelineError):
    """切分后无有效内容"""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            f"文档切分后无有效内容（可能为空文件或不支持的格式）: doc_id={doc_id!r}"
        )
        self.doc_id = doc_id


# ═══════════════════════════════════════════════════════
# Pydantic I/O 模型
# ═══════════════════════════════════════════════════════

@dataclass
class StepTiming:
    """单阶段耗时"""
    step: str
    elapsed_ms: float


class IndexingPipelineInput(BaseModel):
    """索引流水线输入"""
    kb_id: str = Field(..., min_length=1, description="知识库 ID")
    doc_id: str = Field(..., min_length=1, description="文档 ID")
    file_path: str = Field(..., min_length=1, description="原始文件路径")
    vector_namespace: str = Field(..., min_length=1, description="ChromaDB 集合名")
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="每批写入向量库的 chunk 数量",
    )


class IndexingPipelineOutput(BaseModel):
    """索引流水线输出"""
    success: bool
    message: str
    chunk_count: int = 0
    total_elapsed_ms: float = 0.0
    step_timings: List[StepTiming] = Field(default_factory=list)
    error: str = ""


# ═══════════════════════════════════════════════════════
# Skill 接口协议（解耦具体实现）
# ═══════════════════════════════════════════════════════

class LoaderProtocol(Protocol):
    """文档加载器接口"""

    def execute(self, input_data: Any) -> Dict[str, Any]:
        ...


class ChunkerProtocol(Protocol):
    """文档切分器接口"""

    def execute(self, input_data: Any) -> Dict[str, Any]:
        ...


class EmbedderProtocol(Protocol):
    """文本向量化接口"""

    def execute(self, input_data: Any) -> Dict[str, Any]:
        ...


# ═══════════════════════════════════════════════════════
# 索引流水线
# ═══════════════════════════════════════════════════════

class IndexingPipeline:
    """
    文档索引流水线（依赖注入，可测试）

    使用方式:

        # 生产：注入真实 Skill
        pipeline = IndexingPipeline(
            loader=DocumentLoaderSkill(),
            chunker=DocumentChunkerSkill(),
            embedder=TextEmbedderSkill(),
        )
        result = pipeline.run(input_data)

        # 测试：注入 Mock
        pipeline = IndexingPipeline(
            loader=mock_loader,
            chunker=mock_chunker,
            embedder=mock_embedder,
        )
    """

    def __init__(
        self,
        loader: LoaderProtocol,
        chunker: ChunkerProtocol,
        embedder: EmbedderProtocol,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder

    # ═══════════════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════════════

    def run(self, inp: IndexingPipelineInput) -> IndexingPipelineOutput:
        """
        执行完整索引流水线（状态管理 + 核心逻辑）

        状态机:
            PENDING → PROCESSING → DONE  (成功)
                                → ERROR  (失败)
        """
        t0 = time.perf_counter()

        logger.info(
            "索引流水线启动 | kb_id=%s doc_id=%s file=%s namespace=%s",
            inp.kb_id, inp.doc_id, inp.file_path, inp.vector_namespace,
        )

        # 1. 标记处理中
        self._set_processing(inp.kb_id, inp.doc_id)

        try:
            output = self._execute_pipeline(inp)
            output.total_elapsed_ms = (time.perf_counter() - t0) * 1000

            # 成功：更新最终状态
            self._set_done(inp.kb_id, inp.doc_id, output.chunk_count)

            logger.info(
                "索引流水线完成 | kb_id=%s doc_id=%s chunks=%d elapsed=%.1fms",
                inp.kb_id, inp.doc_id, output.chunk_count, output.total_elapsed_ms,
            )
            return output

        except IndexingPipelineError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "索引流水线失败 | kb_id=%s doc_id=%s error_type=%s error=%s elapsed=%.1fms",
                inp.kb_id, inp.doc_id, type(exc).__name__, exc, elapsed_ms,
            )
            self._set_error_safe(inp.kb_id, inp.doc_id)
            return IndexingPipelineOutput(
                success=False,
                message="索引失败",
                total_elapsed_ms=round(elapsed_ms, 2),
                error=str(exc),
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "索引流水线异常（非预期）| kb_id=%s doc_id=%s elapsed=%.1fms",
                inp.kb_id, inp.doc_id, elapsed_ms,
            )
            self._set_error_safe(inp.kb_id, inp.doc_id)
            return IndexingPipelineOutput(
                success=False,
                message="索引失败（系统异常）",
                total_elapsed_ms=round(elapsed_ms, 2),
                error="内部错误，请查看日志",
            )

    # ═══════════════════════════════════════════════════
    # 核心流水线（纯逻辑，不管理状态）
    # ═══════════════════════════════════════════════════

    def _execute_pipeline(self, inp: IndexingPipelineInput) -> IndexingPipelineOutput:
        """执行 Load → Chunk → Embed → Store 四阶段"""
        step_timings: List[StepTiming] = []

        # --- Phase 1: Load ---
        t1 = time.perf_counter()
        load_result = self._load_document(inp.file_path)
        step_timings.append(StepTiming("load", (time.perf_counter() - t1) * 1000))

        # --- Phase 2: Chunk ---
        t2 = time.perf_counter()
        chunks = self._chunk_document(load_result["content"])
        step_timings.append(StepTiming("chunk", (time.perf_counter() - t2) * 1000))

        if not chunks:
            raise EmptyChunksError(inp.doc_id)

        # --- Phase 3: Embed ---
        t3 = time.perf_counter()
        embedded = self._embed_chunks(inp, chunks)
        step_timings.append(StepTiming("embed", (time.perf_counter() - t3) * 1000))

        # --- Phase 4: Store ---
        t4 = time.perf_counter()
        self._store_vectors(inp, embedded, chunks)
        step_timings.append(StepTiming("store", (time.perf_counter() - t4) * 1000))

        return IndexingPipelineOutput(
            success=True,
            message="索引完成",
            chunk_count=len(chunks),
            step_timings=step_timings,
        )

    # ═══════════════════════════════════════════════════
    # Phase 1: Load
    # ═══════════════════════════════════════════════════

    def _load_document(self, file_path: str) -> Dict[str, Any]:
        """
        加载原始文档

        Returns:
            {"content": str, ...}

        Raises:
            DocumentLoadError
        """
        logger.debug("Phase 1/4 加载文档 | file=%s", file_path)
        try:
            from src.skills.custom.rag_skills.document_loader.skill import (
                DocumentLoaderInput,
            )

            result = self._loader.execute(DocumentLoaderInput(file_path=file_path))
        except Exception as exc:
            raise DocumentLoadError(file_path, str(exc)) from exc

        if result.get("error"):
            raise DocumentLoadError(file_path, result["error"])
        if not result.get("content"):
            raise DocumentLoadError(file_path, "文档内容为空")

        logger.debug(
            "Phase 1/4 完成 | file=%s content_len=%d",
            file_path, len(result["content"]),
        )
        return result

    # ═══════════════════════════════════════════════════
    # Phase 2: Chunk
    # ═══════════════════════════════════════════════════

    def _chunk_document(self, content: str) -> List[Dict[str, Any]]:
        """
        将文档内容切分为 chunks

        Returns:
            [{"index": int, "text": str}, ...]

        Raises:
            ChunkingError
        """
        logger.debug("Phase 2/4 切分文档 | content_len=%d", len(content))
        try:
            from src.skills.custom.rag_skills.document_chunker.skill import (
                DocumentChunkerInput,
            )

            result = self._chunker.execute(DocumentChunkerInput(text=content))
        except Exception as exc:
            raise ChunkingError(str(exc)) from exc

        if not result.get("success"):
            raise ChunkingError(result.get("error", "未知错误"))

        chunks = result.get("chunks", [])
        logger.debug("Phase 2/4 完成 | chunk_count=%d", len(chunks))
        return chunks

    # ═══════════════════════════════════════════════════
    # Phase 3: Embed
    # ═══════════════════════════════════════════════════

    def _embed_chunks(
        self,
        inp: IndexingPipelineInput,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        将 chunks 向量化

        Returns:
            [{"chunk_id": str, "text": str, "embedding": [...], "metadata": {...}}, ...]

        Raises:
            EmbeddingError
        """
        logger.debug(
            "Phase 3/4 向量化 | kb_id=%s doc_id=%s chunk_count=%d",
            inp.kb_id, inp.doc_id, len(chunks),
        )
        try:
            from src.skills.custom.rag_skills.text_embedder.skill import (
                EmbeddingCandidate,
                TextEmbedderInput,
            )

            file_name = Path(inp.file_path).name
            candidates = [
                EmbeddingCandidate(
                    chunk_id=f"{inp.doc_id}_{chunk['index']}",
                    text=chunk["text"],
                    metadata={
                        "kb_id": inp.kb_id,
                        "doc_id": inp.doc_id,
                        "file_name": file_name,
                    },
                )
                for chunk in chunks
            ]

            result = self._embedder.execute(TextEmbedderInput(candidates=candidates))
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

        if not result.get("success"):
            raise EmbeddingError(result.get("error", "未知错误"))

        embedded = result.get("embedded_chunks", [])
        if not embedded or len(embedded) != len(chunks):
            raise EmbeddingError(
                f"向量化结果数量不匹配: expected={len(chunks)}, got={len(embedded)}"
            )

        logger.debug("Phase 3/4 完成 | vectors=%d", len(embedded))
        return embedded

    # ═══════════════════════════════════════════════════
    # Phase 4: Store
    # ═══════════════════════════════════════════════════

    def _store_vectors(
        self,
        inp: IndexingPipelineInput,
        embedded: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        分批将向量写入 ChromaDB（upsert 保证幂等性）

        Raises:
            VectorStoreWriteError
        """
        logger.debug(
            "Phase 4/4 存储向量 | namespace=%s total=%d batch_size=%d",
            inp.vector_namespace, len(embedded), inp.batch_size,
        )

        try:
            collection = vector_store.get_or_create_collection(inp.vector_namespace)
        except VectorStoreError as exc:
            raise VectorStoreWriteError(inp.vector_namespace, str(exc)) from exc

        batch_size = inp.batch_size
        total_batches = (len(embedded) + batch_size - 1) // batch_size
        all_chunk_ids: List[str] = []

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(embedded))
            batch_slice = embedded[start:end]

            ids = [item["chunk_id"] for item in batch_slice]
            embeddings = [item["embedding"] for item in batch_slice]
            metadatas = [item["metadata"] for item in batch_slice]
            documents = [item["text"] for item in batch_slice]

            try:
                collection.upsert(  # ← upsert 保证幂等：相同 id 自动覆盖
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                all_chunk_ids.extend(ids)
                logger.debug(
                    "Phase 4/4 批次写入 | batch=%d/%d count=%d",
                    batch_idx + 1, total_batches, len(ids),
                )
            except Exception as exc:
                # 尝试回滚：删除本批及之前已写入的 chunk_ids
                logger.warning(
                    "向量存储批次失败，尝试回滚 | batch=%d/%d namespace=%s",
                    batch_idx + 1, total_batches, inp.vector_namespace,
                )
                self._rollback_chunks(collection, all_chunk_ids + ids)
                raise VectorStoreWriteError(
                    inp.vector_namespace,
                    f"批次 {batch_idx + 1}/{total_batches} 写入失败: {exc}",
                ) from exc

        logger.debug(
            "Phase 4/4 完成 | namespace=%s total=%d batches=%d",
            inp.vector_namespace, len(embedded), total_batches,
        )

    def _rollback_chunks(self, collection, chunk_ids: List[str]) -> None:
        """尝试删除已写入的 chunk（best-effort）"""
        if not chunk_ids:
            return
        try:
            collection.delete(ids=chunk_ids)
            logger.info("已回滚 %d 个 chunk | ids=%s...", len(chunk_ids), chunk_ids[:5])
        except Exception as exc:
            logger.error("回滚失败（需手动清理）| error=%s ids=%s", exc, chunk_ids[:10])

    # ═══════════════════════════════════════════════════
    # 状态管理（try-except 保护，避免状态更新异常掩盖根因）
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _set_processing(kb_id: str, doc_id: str) -> None:
        """标记 KB 和文档为「处理中」"""
        logger.debug("更新状态 → PROCESSING | kb_id=%s doc_id=%s", kb_id, doc_id)
        try:
            update_kb_status(kb_id, IndexingStatus.PROCESSING)
        except Exception as exc:
            logger.warning("更新 KB 状态失败（继续执行）| kb_id=%s error=%s", kb_id, exc)
        try:
            update_document_status(doc_id, DocumentStatus.PROCESSING)
        except Exception as exc:
            logger.warning("更新文档状态失败（继续执行）| doc_id=%s error=%s", doc_id, exc)

    @staticmethod
    def _set_done(kb_id: str, doc_id: str, chunk_count: int) -> None:
        """标记 KB 和文档为「完成」"""
        logger.debug(
            "更新状态 → DONE/INDEXED | kb_id=%s doc_id=%s chunks=%d",
            kb_id, doc_id, chunk_count,
        )
        try:
            update_kb_status(kb_id, IndexingStatus.DONE)
        except Exception as exc:
            logger.error("更新 KB 状态为 DONE 失败 | kb_id=%s error=%s", kb_id, exc)
        try:
            update_document_status(doc_id, DocumentStatus.INDEXED)
            # 可选：同步更新 chunk_count（如果 CRUD 层支持）
            # update_document_chunk_count(doc_id, chunk_count)
        except Exception as exc:
            logger.error("更新文档状态为 INDEXED 失败 | doc_id=%s error=%s", doc_id, exc)

    @staticmethod
    def _set_error_safe(kb_id: str, doc_id: str) -> None:
        """
        安全地将 KB 和文档标记为错误状态

        即使状态更新本身抛异常也不影响上层（best-effort）
        """
        logger.debug("更新状态 → ERROR/FAILED | kb_id=%s doc_id=%s", kb_id, doc_id)
        try:
            update_kb_status(kb_id, IndexingStatus.ERROR)
        except Exception as exc:
            logger.error("更新 KB 状态为 ERROR 失败 | kb_id=%s error=%s", kb_id, exc)
        try:
            update_document_status(doc_id, DocumentStatus.FAILED)
        except Exception as exc:
            logger.error("更新文档状态为 FAILED 失败 | doc_id=%s error=%s", kb_id, exc)


# ═══════════════════════════════════════════════════════
# 便捷函数（保持向后兼容）
# ═══════════════════════════════════════════════════════

def run_indexing_pipeline(
    kb_id: str,
    doc_id: str,
    file_path: str,
    vector_namespace: str,
    *,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    执行文档索引流水线（便捷函数，向后兼容旧调用方）

    新代码推荐直接使用 IndexingPipeline 类以获得更好的可测试性。
    """
    from src.skills.custom.rag_skills.document_loader.skill import DocumentLoaderSkill
    from src.skills.custom.rag_skills.document_chunker.skill import DocumentChunkerSkill
    from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill

    pipeline = IndexingPipeline(
        loader=DocumentLoaderSkill(),
        chunker=DocumentChunkerSkill(),
        embedder=TextEmbedderSkill(),
    )

    inp = IndexingPipelineInput(
        kb_id=kb_id,
        doc_id=doc_id,
        file_path=file_path,
        vector_namespace=vector_namespace,
        batch_size=batch_size,
    )

    output = pipeline.run(inp)
    return output.model_dump()