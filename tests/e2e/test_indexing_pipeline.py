"""
知识库索引全流程 E2E 测试

覆盖场景：
    - Happy Path: 正常文档索引全链路
    - 幂等性: 同文档索引两次不产生重复
    - StepTiming: 四阶段耗时完整性
    - 不同 batch_size: 边界覆盖
    - 失败场景: 空文件、文件不存在
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.infrastructure.kb_manager import (
    DocumentStatus,
    IndexingStatus,
    get_knowledge_base,
    get_document,
    KnowledgeBaseResponse,
    DocumentResponse,
)
from src.infrastructure.vector_store import (
    CollectionNotFoundError,
    VectorStoreManager,
)
from src.infrastructure.indexing_pipeline import (
    IndexingPipeline,
    IndexingPipelineInput,
    IndexingPipelineOutput,
    EmptyChunksError,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.e2e  # 整个模块标记为 E2E


# ═══════════════════════════════════════════════════════
# Happy Path
# ═══════════════════════════════════════════════════════

class TestHappyPath:
    """正常流程：创建 → 索引 → 验证"""

    def test_full_lifecycle(
        self,
        isolated_kb_and_doc: tuple[KnowledgeBaseResponse, DocumentResponse, Path],
        indexing_pipeline: IndexingPipeline,
        in_memory_vector_store: VectorStoreManager,
    ):
        """
        完整端到端测试:
            1. 已有 KB + 文档（fixture 创建）
            2. 执行索引流水线
            3. 验证 KB 状态 → DONE
            4. 验证文档状态 → INDEXED
            5. 验证 ChromaDB 中确实有向量数据
            6. 验证 chunk_count 一致性
            7. 验证 StepTiming 四阶段完整
        """
        kb, doc, test_file = isolated_kb_and_doc

        # --- Act: 执行索引 ---
        inp = IndexingPipelineInput(
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            file_path=str(test_file),
            vector_namespace=kb.vector_namespace,
        )
        output = indexing_pipeline.run(inp)

        # --- Assert: 返回值 ---
        assert output.success, f"索引应成功 | error={output.error}"
        assert output.chunk_count > 0, "chunk_count 应为正数"
        assert output.total_elapsed_ms > 0, "total_elapsed_ms 应为正数"
        logger.info(
            "索引完成 | chunks=%d elapsed=%.1fms",
            output.chunk_count, output.total_elapsed_ms,
        )

        # --- Assert: KB 状态 ---
        final_kb = get_knowledge_base(kb.kb_id)
        assert final_kb.indexing_status == IndexingStatus.DONE, (
            f"KB 状态应为 DONE，实际: {final_kb.indexing_status}"
        )

        # --- Assert: 文档状态 ---
        final_doc = get_document(doc.doc_id)
        assert final_doc.status == DocumentStatus.INDEXED, (
            f"文档状态应为 INDEXED，实际: {final_doc.status}"
        )

        # --- Assert: StepTiming 四阶段完整性 ---
        step_names = {s.step for s in output.step_timings}
        assert step_names == {"load", "chunk", "embed", "store"}, (
            f"StepTiming 应包含 load/chunk/embed/store，实际: {step_names}"
        )
        for step in output.step_timings:
            assert step.elapsed_ms > 0, (
                f"阶段 {step.step} 的 elapsed_ms 应 > 0，实际: {step.elapsed_ms}"
            )

        # --- Assert: ChromaDB 向量数据 ---
        collection = in_memory_vector_store.get_collection(kb.vector_namespace)
        assert collection is not None, f"Collection 应存在: {kb.vector_namespace}"

        db_count = collection.count()
        assert db_count == output.chunk_count, (
            f"ChromaDB 文档数 ({db_count}) 应与 chunk_count ({output.chunk_count}) 一致"
        )

        # 验证具体 chunk 内容存在
        sample_ids = [f"{doc.doc_id}_{i}" for i in range(min(2, output.chunk_count))]
        if sample_ids:
            try:
                result = collection.get(ids=sample_ids)
                assert len(result["ids"]) == len(sample_ids), (
                    f"应能查到 {len(sample_ids)} 个 chunks，实际: {len(result['ids'])}"
                )
                # 验证 metadata 包含 kb_id
                for meta in result["metadatas"]:
                    assert meta["kb_id"] == kb.kb_id
                    assert meta["doc_id"] == doc.doc_id
                logger.info("ChromaDB 数据验证通过 | ids=%s", sample_ids)
            except Exception as exc:
                pytest.fail(f"查询 ChromaDB 失败: {exc}")


# ═══════════════════════════════════════════════════════
# 幂等性
# ═══════════════════════════════════════════════════════

class TestIdempotency:
    """幂等性：同文档多次索引不产生重复数据"""

    def test_same_document_twice_no_duplicates(
        self,
        isolated_kb_and_doc: tuple[KnowledgeBaseResponse, DocumentResponse, Path],
        indexing_pipeline: IndexingPipeline,
        in_memory_vector_store: VectorStoreManager,
    ):
        """同一文档索引两次，ChromaDB 中向量数不变"""
        kb, doc, test_file = isolated_kb_and_doc

        inp = IndexingPipelineInput(
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            file_path=str(test_file),
            vector_namespace=kb.vector_namespace,
        )

        # 第一次索引
        output1 = indexing_pipeline.run(inp)
        assert output1.success
        count1 = in_memory_vector_store.get_collection(kb.vector_namespace).count()

        # 第二次索引（相同 chunk_ids，upsert 应覆盖不新增）
        output2 = indexing_pipeline.run(inp)
        assert output2.success
        count2 = in_memory_vector_store.get_collection(kb.vector_namespace).count()

        assert count1 == count2, (
            f"幂等性失败：第一次 {count1} 条，第二次 {count2} 条（应为相同）"
        )
        logger.info("幂等性验证通过 | count=%d", count1)


# ═══════════════════════════════════════════════════════
# 不同 batch_size
# ═══════════════════════════════════════════════════════

class TestBatchSize:
    """验证不同 batch_size 下流水线正常工作"""

    @pytest.mark.parametrize("batch_size", [
        pytest.param(1, id="batch_size=1"),
        pytest.param(50, id="batch_size=50"),
        pytest.param(200, id="batch_size=200"),
    ])
    def test_various_batch_sizes(
        self,
        isolated_kb_and_doc: tuple[KnowledgeBaseResponse, DocumentResponse, Path],
        indexing_pipeline: IndexingPipeline,
        in_memory_vector_store: VectorStoreManager,
        batch_size: int,
    ):
        """batch_size=1/50/200 均应成功且数据完整"""
        kb, doc, test_file = isolated_kb_and_doc

        inp = IndexingPipelineInput(
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            file_path=str(test_file),
            vector_namespace=kb.vector_namespace,
            batch_size=batch_size,
        )

        output = indexing_pipeline.run(inp)
        assert output.success, f"batch_size={batch_size} 索引失败: {output.error}"

        collection = in_memory_vector_store.get_collection(kb.vector_namespace)
        assert collection.count() == output.chunk_count, (
            f"batch_size={batch_size}: ChromaDB 数量 ({collection.count()}) != chunk_count ({output.chunk_count})"
        )
        logger.info(
            "batch_size=%d 验证通过 | chunks=%d", batch_size, output.chunk_count,
        )


# ═══════════════════════════════════════════════════════
# 失败场景
# ═══════════════════════════════════════════════════════

class TestFailureScenarios:
    """异常场景：验证状态回滚 & 向量库无脏数据"""

    def test_empty_file_fails_gracefully(
        self,
        tmp_path: Path,
        indexing_pipeline: IndexingPipeline,
        in_memory_vector_store: VectorStoreManager,
    ):
        """
        空文件 → EmptyChunksError → KB=ERROR, Doc=FAILED → 向量库无数据
        """
        from src.infrastructure.kb_manager import create_knowledge_base, add_document_to_kb

        # 手动创建（不用 isolated_kb_and_doc，因为需要空文件）
        kb = create_knowledge_base(name=f"e2e-empty-{_short_id()}", description="空文件测试")
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")  # 空文件
        doc = add_document_to_kb(kb.kb_id, empty_file.name, str(empty_file))

        inp = IndexingPipelineInput(
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            file_path=str(empty_file),
            vector_namespace=kb.vector_namespace,
        )

        output = indexing_pipeline.run(inp)

        # --- Assert: 流水线返回失败 ---
        assert not output.success, "空文件索引应返回 success=False"
        assert "无有效内容" in output.error or "EmptyChunksError" in output.error or "empty" in output.error.lower(), (
            f"错误信息应提示空内容，实际: {output.error}"
        )

        # --- Assert: KB 状态 → ERROR ---
        final_kb = get_knowledge_base(kb.kb_id)
        assert final_kb.indexing_status == IndexingStatus.ERROR, (
            f"空文件索引失败后 KB 状态应为 ERROR，实际: {final_kb.indexing_status}"
        )

        # --- Assert: 文档状态 → FAILED ---
        final_doc = get_document(doc.doc_id)
        assert final_doc.status == DocumentStatus.FAILED, (
            f"空文件索引失败后文档状态应为 FAILED，实际: {final_doc.status}"
        )

        # --- Assert: 向量库无残留 ---
        assert not in_memory_vector_store.collection_exists(kb.vector_namespace), (
            "空文件索引失败后不应创建向量集合"
        )

        # Cleanup
        from src.infrastructure.kb_manager import delete_knowledge_base
        delete_knowledge_base(kb.kb_id)

    def test_nonexistent_file_fails(
        self,
        indexing_pipeline: IndexingPipeline,
        in_memory_vector_store: VectorStoreManager,
    ):
        """
        文件不存在 → DocumentLoadError → KB=ERROR, Doc=FAILED
        """
        from src.infrastructure.kb_manager import create_knowledge_base, add_document_to_kb

        kb = create_knowledge_base(name=f"e2e-nofile-{_short_id()}", description="文件不存在测试")
        fake_path = "/nonexistent/path/ghost_file.txt"
        doc = add_document_to_kb(kb.kb_id, "ghost_file.txt", fake_path)

        inp = IndexingPipelineInput(
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            file_path=fake_path,
            vector_namespace=kb.vector_namespace,
        )

        output = indexing_pipeline.run(inp)

        assert not output.success
        assert "加载失败" in output.error or "not found" in output.error.lower() or "NoSuchFile" in output.error, (
            f"错误信息应提示文件不存在，实际: {output.error}"
        )

        final_kb = get_knowledge_base(kb.kb_id)
        assert final_kb.indexing_status == IndexingStatus.ERROR

        final_doc = get_document(doc.doc_id)
        assert final_doc.status == DocumentStatus.FAILED

        assert not in_memory_vector_store.collection_exists(kb.vector_namespace)

        from src.infrastructure.kb_manager import delete_knowledge_base
        delete_knowledge_base(kb.kb_id)


def _short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]