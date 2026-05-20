"""
E2E 测试共享 fixtures

提供：
    - 内存数据库 Session
    - 内存 ChromaDB 向量存储
    - 隔离的 KB + 文档
    - 测试后自动清理
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import chromadb
import pytest

from src.infrastructure.vector_store import (
    VectorStoreManager,
    VectorStoreSettings,
)
from src.infrastructure.kb_manager import (
    create_knowledge_base,
    add_document_to_kb,
    delete_knowledge_base,
    KnowledgeBaseResponse,
    DocumentResponse,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Session 级：内存基础设施（所有 E2E 测试共享）
# ═══════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def in_memory_vector_store() -> VectorStoreManager:
    """
    创建内存版 ChromaDB 向量存储（session 级单例）

    所有 E2E 测试共享同一个客户端实例，通过不同的 namespace 隔离。
    """
    client = chromadb.Client()  # ← 纯内存，无持久化

    store = VectorStoreManager(
        client=client,
        settings=VectorStoreSettings(
            persist_path=":memory:",
            is_persistent=False,
            allow_reset=True,  # 测试环境允许 reset
        ),
    )
    logger.info("内存 ChromaDB 已就绪（session 级）")
    return store


# ═══════════════════════════════════════════════════════
# Function 级：隔离的 KB + 文档 + 测试文件
# ═══════════════════════════════════════════════════════

@pytest.fixture
def isolated_kb_and_doc(
    tmp_path: Path,
    in_memory_vector_store: VectorStoreManager,
) -> Generator[tuple[KnowledgeBaseResponse, DocumentResponse, Path], None, None]:
    """
    创建隔离的知识库 + 文档 + 测试文件

    Yields:
        (kb, doc, test_file_path)

    Teardown:
        - 删除知识库（含关联文档）
        - 删除对应的 ChromaDB collection
        - tmp_path 自动清理测试文件
    """
    # --- Setup ---
    kb = create_knowledge_base(
        name=f"e2e-test-{_short_id()}",
        description="E2E 自动化测试",
    )

    test_file = tmp_path / "test_sample.txt"
    test_file.write_text(
        "这是第一段测试文本，用于验证知识库索引流水线。\n\n"
        "这是第二段测试文本，包含不同的内容。\n\n"
        "这是第三段，确保能产生多个 chunk。\n"
        "Python 是一门优雅的编程语言。人工智能正在改变世界。\n"
        "知识库是 RAG 系统的核心组件。向量数据库存储嵌入表示。\n"
    )

    doc = add_document_to_kb(
        kb_id=kb.kb_id,
        file_name=test_file.name,
        file_path=str(test_file),
    )

    logger.info(
        "E2E Fixture 就绪 | kb_id=%s doc_id=%s file=%s",
        kb.kb_id, doc.doc_id, test_file,
    )

    yield kb, doc, test_file

    # --- Teardown ---
    try:
        delete_knowledge_base(kb.kb_id)
        logger.info("已删除测试 KB | kb_id=%s", kb.kb_id)
    except Exception as exc:
        logger.warning("删除 KB 失败（可能已被测试删除）| kb_id=%s error=%s", kb.kb_id, exc)

    try:
        in_memory_vector_store.delete_collection(kb.vector_namespace)
        logger.info("已删除测试 Collection | namespace=%s", kb.vector_namespace)
    except Exception as exc:
        logger.warning(
            "删除 Collection 失败（可能不存在）| namespace=%s error=%s",
            kb.vector_namespace, exc,
        )


@pytest.fixture
def indexing_pipeline(in_memory_vector_store: VectorStoreManager):
    """
    创建索引流水线实例（注入真实 Skill + 内存向量存储）

    注意：此 fixture 使用真实 DocumentLoader / Chunker / Embedder，
    如需 Mock Embedding，使用 `indexing_pipeline_mocked` fixture。
    """
    from src.skills.custom.rag_skills.document_loader.skill import DocumentLoaderSkill
    from src.skills.custom.rag_skills.document_chunker.skill import DocumentChunkerSkill
    from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
    from src.infrastructure.indexing_pipeline import IndexingPipeline

    # 注入内存向量存储到全局（或通过 MonkeyPatch）
    import src.infrastructure.indexing_pipeline as ip_module
    original = ip_module.vector_store
    ip_module.vector_store = in_memory_vector_store

    pipeline = IndexingPipeline(
        loader=DocumentLoaderSkill(),
        chunker=DocumentChunkerSkill(),
        embedder=TextEmbedderSkill(),
    )

    yield pipeline

    # 恢复
    ip_module.vector_store = original


def _short_id() -> str:
    """生成短随机 ID（避免 KB 名过长）"""
    import uuid
    return uuid.uuid4().hex[:8]