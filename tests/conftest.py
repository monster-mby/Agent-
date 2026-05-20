"""
真实环境测试的自动跑腿员（fixtures）—— 优化版

pytest 在跑测试前，自动帮你做好：
  1. 从 .env.test 读 API 密钥
  2. 创建 2 个翻译官（RealLLMClient + RealEmbeddingClient）
  3. 把翻译官塞进技能管理器
  4. 找到测试文档路径
  5. 没填密钥 → 自动 skip，不崩

优化要点：
  - 日志统一用 logging，可通过 --log-cli-level=INFO 控制
  - 新增 semantic_test_sentences / company_report_doc / expected_facts fixture
  - 新增异常测试用 fixture：invalid_api_key_llm_client
  - 所有 fixture 加类型标注
  - 模块级常量消除魔法字符串
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Any

import pytest
from dotenv import load_dotenv

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 环境变量加载
# ============================================================================
load_dotenv(Path(__file__).parent.parent / ".env.test")

# ============================================================================
# 常量
# ============================================================================
_DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL = "gpt-4o-mini"
_DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DOC_GLOB_PATTERNS = ["*.txt", "*.md"]

# ============================================================================
# Fixture: 大模型翻译官
# ============================================================================

@pytest.fixture(scope="module")
def real_llm_client() -> Generator[Optional[object], None, None]:
    """准备大模型翻译官（模块级，只创建一次）。

    从 .env.test 读取：
      - LLM_API_KEY（必须）
      - LLM_BASE_URL（可选，默认 OpenAI）
      - LLM_MODEL（可选，默认 gpt-4o-mini）

    密钥缺失 → pytest.skip，不崩。
    """
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        pytest.skip("未配置 LLM_API_KEY，跳过真实环境测试")

    from tests.real_clients import RealLLMClient

    client = RealLLMClient(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL),
        model=os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL),
    )

    logger.info("✅ RealLLMClient 就绪: model=%s", client.model if hasattr(client, "model") else "?")
    yield client


# ============================================================================
# Fixture: 向量翻译官
# ============================================================================

@pytest.fixture(scope="module")
def real_embedding_client() -> Generator[Optional[object], None, None]:
    """准备向量翻译官（模块级，只创建一次）。

    密钥优先级：EMBEDDING_API_KEY > LLM_API_KEY（自动复用）。
    """
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")

    if not api_key:
        pytest.skip("未配置 EMBEDDING_API_KEY 且未配置 LLM_API_KEY，跳过真实环境测试")

    from tests.real_clients import RealEmbeddingClient

    client = RealEmbeddingClient(
        api_key=api_key,
        base_url=os.getenv("EMBEDDING_BASE_URL", _DEFAULT_EMBEDDING_BASE_URL),
        model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
    )

    key_source = "EMBEDDING_API_KEY" if os.getenv("EMBEDDING_API_KEY") else "LLM_API_KEY（复用）"
    logger.info("✅ RealEmbeddingClient 就绪: model=%s, key=%s",
                client.model if hasattr(client, "model") else "?", key_source)
    yield client


# ============================================================================
# Fixture: 技能管理器（装满所有技能 + 注入真实客户端）
# ============================================================================

@pytest.fixture(scope="module")
def real_skill_manager(
    real_llm_client: object,
    real_embedding_client: object,
) -> Generator[Dict[str, Any], None, None]:
    """准备装满技能的工具箱（模块级，只创建一次）。

    自动做的事：
      1. 注册全部 5 个 RAG 技能
      2. 把真实 Embedding 客户端注入 TextEmbedderSkill
      3. 返回 {"manager": ..., "llm_client": ...} 字典，供测试使用

    注意：RagAnswerSkill 的 llm_client 需要在调用时通过 input_data 传入，
         而不是注入到技能实例属性中。
    """
    from src.skills.base.skill_manager import SkillManager
    from src.skills.custom.rag_skills.document_loader.skill import DocumentLoaderSkill
    from src.skills.custom.rag_skills.document_chunker.skill import DocumentChunkerSkill
    from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
    from src.skills.custom.rag_skills.vector_search.skill import VectorSearchSkill
    from src.skills.custom.rag_skills.rag_answer.skill import RagAnswerSkill

    manager = SkillManager()

    # 注册全部 RAG 技能
    manager.register(DocumentLoaderSkill)
    manager.register(DocumentChunkerSkill)
    manager.register(TextEmbedderSkill)
    manager.register(VectorSearchSkill)
    manager.register(RagAnswerSkill)
    logger.info("📦 已注册 5 个 RAG 技能")

    # 注入真实向量客户端 → TextEmbedderSkill
    text_embedder = manager._registry.get("text_embedder")
    if text_embedder:
        text_embedder._client = real_embedding_client
        text_embedder._build_client = lambda *a, **k: real_embedding_client
        logger.info("🔗 已注入 RealEmbeddingClient → TextEmbedderSkill")

    # 返回管理器和 LLM 客户端（供测试调用时使用）
    yield {
        "manager": manager,
        "llm_client": real_llm_client,
    }


# ============================================================================
# Fixture: 测试文档目录 & 文档列表
# ============================================================================

@pytest.fixture(scope="module")
def sample_document_dir() -> Generator[Path, None, None]:
    """测试文档目录（模块级）。

    路径：tests/fixtures/real_docs/
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "real_docs"

    if not fixtures_dir.exists():
        pytest.skip(f"测试文档目录不存在: {fixtures_dir}")

    logger.info("📁 测试文档目录: %s", fixtures_dir)
    yield fixtures_dir


@pytest.fixture(scope="module")
def sample_documents(sample_document_dir: Path) -> Generator[List[Path], None, None]:
    """测试文档路径列表（模块级）。

    自动扫描 *.txt 和 *.md 文件，至少需要一个。
    """
    docs: List[Path] = []
    for pattern in _DOC_GLOB_PATTERNS:
        docs.extend(sample_document_dir.glob(pattern))

    if not docs:
        pytest.skip(f"测试文档目录中没有找到可用的文档文件: {sample_document_dir}")

    logger.info("📄 找到 %d 个测试文档: %s", len(docs), [d.name for d in docs])
    yield docs


# ============================================================================
# 新增 Fixture: 语义相似度测试句对（考题2）
# ============================================================================

@pytest.fixture(scope="module")
def semantic_test_sentences() -> List[str]:
    """语义相似度测试用的 3 句话。

    前两句语义相似，第三句与前两句无关。
    用于验证嵌入模型是否能正确识别语义距离。
    """
    return [
        "今天天气很好，适合出门",
        "今天是晴天，阳光很好",
        "我晚上想吃火锅",
    ]


# ============================================================================
# 新增 Fixture: 定位 company_report.txt（考题4 & 5）
# ============================================================================

@pytest.fixture(scope="module")
def company_report_doc(sample_documents: List[Path]) -> Path:
    """定位 company_report.txt，找不到则 skip。

    匹配规则：文件名含 "company" 或 "report"。
    """
    for doc in sample_documents:
        name = doc.name.lower()
        if "company" in name or "report" in name:
            logger.info("🏢 定位到公司报告文档: %s", doc.name)
            return doc

    pytest.skip(
        f"未找到 company_report.txt，跳过相关测试。"
        f"可用文档: {[d.name for d in sample_documents]}"
    )


# ============================================================================
# 新增 Fixture: 文档预期事实（考题4 & 5）
# ============================================================================

@pytest.fixture(scope="module")
def expected_facts() -> Dict[str, Dict[str, str]]:
    """文档中预期可检索的关键事实。

    与 tests/fixtures/real_docs/company_report.txt 内容一一对应。
    修改文档时务必同步更新这里，避免测试因内容变更而静默失败。

    结构：
        {
            "company_report.txt": {
                "revenue": "100亿元",      # Q3 总营收
                "ceo": "张明辉",            # CEO 姓名
                "headcount": "15,000人",     # 员工总数
            }
        }
    """
    return {
        "company_report.txt": {
            "revenue": "100亿元",
            "ceo": "张明辉",
            "headcount": "15,000人",
        }
    }


# ============================================================================
# 新增 Fixture: 无效 API Key 的 LLM 客户端（异常路径测试）
# ============================================================================

@pytest.fixture(scope="module")
def invalid_api_key_llm_client() -> Generator[object, None, None]:
    """创建一个使用无效 API Key 的 LLM 客户端。

    用于验证系统在鉴权失败时能否返回清晰错误，而非崩溃。
    """
    from tests.real_clients import RealLLMClient

    client = RealLLMClient(
        api_key="sk-invalid-key-for-testing",
        base_url=os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL),
        model=os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL),
    )

    logger.info("⚠️ 已创建无效 API Key 的 LLM 客户端（用于异常测试）")
    yield client


@pytest.fixture(scope="module")
def invalid_api_key_llm_client() -> Generator[object, None, None]:
    """创建一个使用无效 API Key 的 LLM 客户端。

    用于验证系统在鉴权失败时能否返回清晰错误，而非崩溃。
    """
    from tests.real_clients import RealLLMClient

    client = RealLLMClient(
        api_key="sk-invalid-key-for-testing",
        base_url=os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL),
        model=os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL),
    )

    logger.info("⚠️ 已创建无效 API Key 的 LLM 客户端（用于异常测试）")
    yield client


# ============================================================================
# Fixture: 自动清理 Checkpoint（防止跨测试状态累积）
# ============================================================================

@pytest.fixture(autouse=True)
def reset_checkpointer():
    """每个测试前自动重置 checkpointer，防止 skill_results 跨测试累积。

    根因：SqliteSaver 持久化到 checkpoints.sqlite，operator.add reducer
    会把新结果追加到旧记录后面，导致 len(skill_results) 不符合预期。

    解决方案：每个测试前调用 reset_checkpointer() 清除缓存实例，
    下次 get_checkpointer() 会重新创建干净的数据库连接。
    """
    from src.agent.langgraph.checkpointer import reset_checkpointer

    # 重置默认 SQLite checkpointer
    reset_checkpointer(db_path="data/checkpoints.sqlite", use_async=False)

    # 如果测试使用了内存数据库，也重置
    try:
        reset_checkpointer(db_path=":memory:", use_async=False)
    except Exception:
        pass  # 内存库可能不存在，忽略

    yield