"""
GraphRAG Searcher Skill 完整测试套件

覆盖场景：
  ✅ 基础注册 & 元数据校验
  ✅ 客户端注入 / 自动构建 / 降级
  ✅ faiss 向量搜索（含 numpy 降级）
  ✅ tiktoken Token 截断（含字符估算降级）
  ✅ 缓存命中 / 失效重建
  ✅ 线程安全锁
  ✅ Local / Global / Hybrid 三种模式
  ✅ 中文 jieba 分词 fallback
  ✅ LLM 重试（tenacity）
  ✅ Prompt 模板渲染（jinja2 / str.replace 降级）
  ✅ Pydantic 输出模型
  ✅ SkillManager.call() 集成
  ✅ 索引缺失时的报错
  ✅ 嵌入获取接口适配（单数/复数签名）
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.skills.custom.rag_skills.graphrag_searcher import skill as graphrag_skill
from src.skills.custom.rag_skills.graphrag_searcher.skill import (
    GraphRAGSearcherSkill,
    _GraphRAGEngine,
    configure_graphrag_searcher,
    _clear_injected_clients,
    _render_prompt,
    HAS_FAISS,
    HAS_TIKTOKEN,
    HAS_TENACITY,
    HAS_JIEBA,
    HAS_PYDANTIC,
)


# ============================================================
# 模拟数据工厂
# ============================================================

def make_mock_entities(n: int = 10):
    """创建模拟实体 DataFrame"""
    np.random.seed(42)
    embeddings = np.random.randn(n, 128).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return pd.DataFrame({
        "title": [f"Entity_{i}" for i in range(n)],
        "type": ["Person" if i % 2 == 0 else "Organization" for i in range(n)],
        "description": [f"Description for Entity_{i}" for i in range(n)],
        "embedding": [embeddings[i].tolist() for i in range(n)],
    })


def make_mock_relationships(entities_df, n: int = 15):
    """创建模拟关系 DataFrame"""
    titles = entities_df["title"].tolist()
    sources = [titles[i % len(titles)] for i in range(n)]
    targets = [titles[(i + 1) % len(titles)] for i in range(n)]
    return pd.DataFrame({
        "source": sources,
        "target": targets,
        "description": [f"Relationship {i} between {s} and {t}" for i, (s, t) in enumerate(zip(sources, targets))],
    })


def make_mock_communities(entities_df, n: int = 5):
    """创建模拟社区 DataFrame"""
    np.random.seed(99)
    embeddings = np.random.randn(n, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return pd.DataFrame({
        "community": list(range(n)),
        "title": [f"Community_{i}: Topic {i}" for i in range(n)],
        "embedding": [embeddings[i].tolist() for i in range(n)],
    })


def make_mock_community_reports(communities_df, entities_df):
    """创建模拟社区报告 DataFrame"""
    rows = []
    entity_names = entities_df["title"].tolist()
    for _, comm in communities_df.iterrows():
        cid = comm["community"]
        # 包含一些实体名
        involved_entities = entity_names[cid % len(entity_names):(cid + 2) % len(entity_names) + 1]
        content = (
            f"## Community {cid} Report\n\n"
            f"This community involves: {', '.join(involved_entities)}.\n"
            f"Summary for community {cid}."
        )
        rows.append({
            "community": cid,
            "title": comm["title"],
            "full_content": content,
        })
    return pd.DataFrame(rows)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_data_dir(tmp_path):
    """创建模拟的 GraphRAG artifacts 目录（含 _SUCCESS 标记）"""
    artifacts_dir = tmp_path / "output" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # 写入 _SUCCESS
    (artifacts_dir / "_SUCCESS").write_text("ok")

    entities = make_mock_entities(10)
    relationships = make_mock_relationships(entities, 15)
    communities = make_mock_communities(entities, 5)
    community_reports = make_mock_community_reports(communities, entities)

    entities.to_parquet(artifacts_dir / "entities.parquet")
    relationships.to_parquet(artifacts_dir / "relationships.parquet")
    communities.to_parquet(artifacts_dir / "communities.parquet")
    community_reports.to_parquet(artifacts_dir / "community_reports.parquet")

    return tmp_path


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端"""
    client = mock.MagicMock()
    client.generate.return_value = "这是一个模拟的回答。"
    return client


@pytest.fixture
def mock_embedding_client():
    """模拟嵌入客户端（复数签名）"""
    client = mock.MagicMock()
    client.execute.return_value = {
        "embeddings": [np.random.randn(128).astype(np.float32).tolist()],
    }
    return client


@pytest.fixture(autouse=True)
def clear_cache_and_injects():
    """每个测试前清空模块级状态"""
    global graphrag_skill
    _clear_injected_clients()
    graphrag_skill._searcher_cache = None
    yield
    _clear_injected_clients()
    graphrag_skill._searcher_cache = None


@pytest.fixture
def engine_with_mocks(mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
    """注入客户端并创建引擎"""
    monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
    configure_graphrag_searcher(
        llm_client=mock_llm_client,
        embedding_client=mock_embedding_client,
    )
    engine = _GraphRAGEngine(mock_data_dir)
    return engine, mock_llm_client, mock_embedding_client


# ============================================================
# 元数据 & 注册测试
# ============================================================

class TestSkillMetadata:
    def test_skill_has_required_attrs(self):
        assert GraphRAGSearcherSkill.name == "graphrag_searcher"
        assert len(GraphRAGSearcherSkill.description) > 0
        assert GraphRAGSearcherSkill.version == "0.2.0"
        assert "local" in GraphRAGSearcherSkill.description
        assert "global" in GraphRAGSearcherSkill.description
        assert len(GraphRAGSearcherSkill.triggers) >= 3

    def test_skill_can_be_registered(self):
        from src.skills.base.skill_manager import SkillManager
        manager = SkillManager()
        manager.register(GraphRAGSearcherSkill)
        assert "graphrag_searcher" in manager.get_all()


# ============================================================
# 客户端注入测试
# ============================================================

class TestClientInjection:
    def test_inject_llm_client(self):
        fake_llm = mock.MagicMock()
        configure_graphrag_searcher(llm_client=fake_llm)
        assert graphrag_skill._inject_llm_client is fake_llm

    def test_inject_embedding_client(self):
        fake_emb = mock.MagicMock()
        configure_graphrag_searcher(embedding_client=fake_emb)
        assert graphrag_skill._inject_embedding_client is fake_emb

    def test_clear_injected(self):
        configure_graphrag_searcher(llm_client=mock.MagicMock())
        _clear_injected_clients()
        assert graphrag_skill._inject_llm_client is None
        assert graphrag_skill._inject_embedding_client is None

    def test_thread_safe_inject(self):
        """多线程同时注入不应崩溃"""
        def inject():
            configure_graphrag_searcher(llm_client=mock.MagicMock())

        threads = [threading.Thread(target=inject) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 无异常即通过
        assert graphrag_skill._inject_llm_client is not None


# ============================================================
# Prompt 渲染测试
# ============================================================

class TestPromptRendering:
    def test_render_with_jinja2(self):
        result = _render_prompt("Hello {{ name }}!", name="World")
        assert result == "Hello World!"

    def test_render_multiple_vars(self):
        result = _render_prompt("{{ a }} + {{ b }} = {{ c }}", a=1, b=2, c=3)
        assert result == "1 + 2 = 3"

    def test_render_no_vars(self):
        result = _render_prompt("Plain text")
        assert result == "Plain text"


# ============================================================
# 引擎初始化测试
# ============================================================

class TestEngineInit:
    def test_engine_loads_data(self, mock_data_dir, mock_llm_client, mock_embedding_client):
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )
        engine = _GraphRAGEngine(mock_data_dir)
        assert len(engine.entities) == 10
        assert len(engine.relationships) == 15
        assert len(engine.communities) == 5
        assert len(engine.community_reports) == 5
        assert engine.llm_client is mock_llm_client
        assert engine.embedding_client is mock_embedding_client

    def test_engine_fails_without_index(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="索引不存在"):
            _GraphRAGEngine(tmp_path)

    def test_engine_faiss_index_built(self, mock_data_dir, mock_llm_client, mock_embedding_client):
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )
        engine = _GraphRAGEngine(mock_data_dir)
        assert engine._entity_index is not None
        if HAS_FAISS:
            assert hasattr(engine._entity_index, 'search')
        else:
            assert isinstance(engine._entity_index, np.ndarray)

    def test_engine_community_inverted_index(self, mock_data_dir, mock_llm_client, mock_embedding_client):
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )
        engine = _GraphRAGEngine(mock_data_dir)
        # 至少有部分实体有社区归属
        assert isinstance(engine._entity_to_communities, dict)


# ============================================================
# 缓存机制测试
# ============================================================

class TestCacheMechanism:
    def test_cache_singleton(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        skill1 = GraphRAGSearcherSkill()
        skill2 = GraphRAGSearcherSkill()

        # 同一个引擎实例
        assert skill1._engine is skill2._engine

    def test_cache_stale_detection(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        skill = GraphRAGSearcherSkill()
        original_engine = skill._engine

        # 模拟索引更新：写入新文件
        import time
        time.sleep(0.1)  # 确保 mtime 有变化
        new_file = mock_data_dir / "output" / "artifacts" / "updated.parquet"
        new_file.write_bytes(b"new data")

        # 下次初始化检测到 stale
        graphrag_skill._searcher_cache = None  # 手动清空模拟重建
        skill2 = GraphRAGSearcherSkill()
        # 引擎应重建（新的 mtime）
        assert skill2._engine._artifacts_mtime >= original_engine._artifacts_mtime


# ============================================================
# Local Search 测试
# ============================================================

class TestLocalSearch:
    def test_local_search_returns_answer(self, engine_with_mocks):
        engine, mock_llm, mock_emb = engine_with_mocks
        result = engine.local_search("Entity_0", top_k=3, max_tokens=2000)

        assert "answer" in result
        assert result["mode"] == "local"
        assert len(result["sources"]) > 0
        assert result["sources"][0]["type"] == "entity"
        mock_llm.generate.assert_called_once()

    def test_local_search_finds_top_entities(self, engine_with_mocks):
        engine, mock_llm, mock_emb = engine_with_mocks
        result = engine.local_search("Entity_0 Entity_1", top_k=5, max_tokens=2000)

        assert result["entity_count"] == 5

    def test_local_search_no_embedding_fallback(self, mock_data_dir, mock_llm_client, monkeypatch):
        """无嵌入客户端时应降级为关键词匹配"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(llm_client=mock_llm_client)
        # 不注入 embedding_client

        engine = _GraphRAGEngine(mock_data_dir)
        result = engine.local_search("Entity_0", top_k=3, max_tokens=2000)

        assert "answer" in result
        assert len(result["sources"]) > 0  # 关键词匹配找到了 Entity_0


# ============================================================
# Global Search 测试
# ============================================================

class TestGlobalSearch:
    def test_global_search_returns_answer(self, engine_with_mocks):
        engine, mock_llm, mock_emb = engine_with_mocks
        result = engine.global_search("overall topics", top_k=3)

        assert "answer" in result
        assert result["mode"] == "global"
        assert result["community_count"] == 3
        assert len(result["sources"]) > 0
        assert result["sources"][0]["type"] == "community"

    def test_global_search_sources_are_communities(self, engine_with_mocks):
        engine, mock_llm, mock_emb = engine_with_mocks
        result = engine.global_search("Topic", top_k=2)

        for src in result["sources"]:
            assert src["type"] == "community"


# ============================================================
# Hybrid Search 测试
# ============================================================

class TestHybridSearch:
    def test_hybrid_combines_both(self, engine_with_mocks):
        engine, mock_llm, mock_emb = engine_with_mocks
        result = engine.hybrid_search("Entity_0", top_k_entities=3, top_k_communities=2, max_tokens=4000)

        assert result["mode"] == "hybrid"
        assert result["entity_count"] > 0
        assert result["community_count"] > 0
        assert len(result["sources"]) > 0
        # hybrid 调用了 3 次 LLM: local + global + merge
        assert mock_llm.generate.call_count == 3


# ============================================================
# Token 截断测试
# ============================================================

class TestTokenTruncation:
    def test_tiktoken_truncation(self, engine_with_mocks):
        engine, _, _ = engine_with_mocks
        long_text = "hello " * 5000  # ~5000 tokens
        truncated = engine._truncate_by_tokens(long_text, max_tokens=100)
        assert len(truncated) < len(long_text)

    def test_short_text_not_truncated(self, engine_with_mocks):
        engine, _, _ = engine_with_mocks
        short = "hello world"
        result = engine._truncate_by_tokens(short, max_tokens=1000)
        assert result == short

    def test_no_tiktoken_fallback_to_chars(self, mock_data_dir, mock_llm_client, monkeypatch):
        """如果 tiktoken 不可用，应该按字符估算"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(llm_client=mock_llm_client)

        with mock.patch("src.skills.custom.rag_skills.graphrag_searcher.skill.HAS_TIKTOKEN", False):
            engine = _GraphRAGEngine(mock_data_dir)
            # 手动置空 tokenizer
            engine._tokenizer = None

            long_text = "x" * 5000
            truncated = engine._truncate_by_tokens(long_text, max_tokens=100)
            # 100 tokens ≈ 200 chars
            assert len(truncated) <= 200


# ============================================================
# LLM 重试测试
# ============================================================

class TestLLMRetry:
    def test_retry_on_failure(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        # 前两次失败，第三次成功
        mock_llm_client.generate.side_effect = [
            ConnectionError("network error"),
            TimeoutError("timeout"),
            "最终成功的回答",
        ]

        engine = _GraphRAGEngine(mock_data_dir)
        result = engine._generate("test prompt")

        if HAS_TENACITY:
            assert "最终成功" in result
            assert mock_llm_client.generate.call_count == 3
        else:
            # 无 tenacity 时只调一次，失败就返回错误
            assert mock_llm_client.generate.call_count == 1

    def test_retry_exhausted(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )
        mock_llm_client.generate.side_effect = ConnectionError("always failing")

        engine = _GraphRAGEngine(mock_data_dir)
        result = engine._generate("test")

        # 应该返回错误信息，不抛异常
        assert "失败" in result or "LLM" in result

    def test_no_llm_client_returns_warning(self, mock_data_dir, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        # 不注入 LLM 客户端，自动构建也会因 mock 环境失败
        # 手动创建一个没有 llm_client 的 engine
        configure_graphrag_searcher(embedding_client=mock_embedding_client)
        engine = _GraphRAGEngine(mock_data_dir)
        engine.llm_client = None  # 手动置空

        result = engine._generate("test")
        assert "未配置" in result


# ============================================================
# 嵌入获取适配测试
# ============================================================

class TestEmbeddingAdaptation:
    def test_execute_with_plural_signature(self, mock_data_dir, mock_llm_client, monkeypatch):
        """嵌入客户端使用 execute(texts=[...]) 复数签名"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        emb_client = mock.MagicMock()
        emb_client.execute.return_value = {
            "embeddings": [[0.1] * 128],
        }
        configure_graphrag_searcher(llm_client=mock_llm_client, embedding_client=emb_client)

        engine = _GraphRAGEngine(mock_data_dir)
        embedding = engine._get_embedding("test query")
        assert embedding is not None
        assert len(embedding) == 128

    def test_execute_with_singular_signature(self, mock_data_dir, mock_llm_client, monkeypatch):
        """嵌入客户端使用 execute(text=...) 单数签名"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        emb_client = mock.MagicMock()
        # 第一次调用（复数）失败，第二次（单数）成功
        emb_client.execute.side_effect = [
            TypeError("unexpected keyword argument 'texts'"),
            {"embedding": [0.2] * 128},
        ]
        configure_graphrag_searcher(llm_client=mock_llm_client, embedding_client=emb_client)

        engine = _GraphRAGEngine(mock_data_dir)
        embedding = engine._get_embedding("test query")
        assert embedding is not None
        assert len(embedding) == 128

    def test_embed_method(self, mock_data_dir, mock_llm_client, monkeypatch):
        """嵌入客户端使用 .embed(text) 方法"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        emb_client = mock.MagicMock()
        emb_client.embed.return_value = [0.3] * 128
        # 移除 execute 方法
        del emb_client.execute
        configure_graphrag_searcher(llm_client=mock_llm_client, embedding_client=emb_client)

        engine = _GraphRAGEngine(mock_data_dir)
        embedding = engine._get_embedding("test query")
        assert embedding is not None
        assert len(embedding) == 128
        emb_client.embed.assert_called_once_with("test query")

    def test_no_embedding_client_returns_none(self, mock_data_dir, mock_llm_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(llm_client=mock_llm_client)
        engine = _GraphRAGEngine(mock_data_dir)
        engine.embedding_client = None

        result = engine._get_embedding("test")
        assert result is None


# ============================================================
# 降级搜索测试（无 faiss / 无 embedding）
# ============================================================

class TestFallbackSearch:
    def test_fallback_entity_match_basic(self, mock_data_dir, mock_llm_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(llm_client=mock_llm_client)
        engine = _GraphRAGEngine(mock_data_dir)
        engine.embedding_client = None

        results = engine._fallback_entity_match("Entity_0", top_k=3)
        assert len(results) > 0
        # Entity_0 应被匹配到
        names = [r["name"] for r in results]
        assert "Entity_0" in names

    def test_fallback_with_jieba(self, mock_data_dir, mock_llm_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(llm_client=mock_llm_client)
        engine = _GraphRAGEngine(mock_data_dir)
        engine.embedding_client = None

        # 中英文混合查询
        results = engine._fallback_entity_match("Entity 测试", top_k=3)
        # 至少不崩溃
        assert isinstance(results, list)


# ============================================================
# SkillManager.call() 集成
# ============================================================

class TestSkillManagerIntegration:
    def test_call_local_search(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        from src.skills.base.skill_manager import SkillManager
        manager = SkillManager()
        manager.register(GraphRAGSearcherSkill)

        result = manager.call(
            "graphrag_searcher",
            query="Entity_0 和 Entity_1 的关系",
            mode="local",
            top_k_entities=5,
        )

        assert result["status"] == "success"
        assert result["skill"] == "graphrag_searcher"
        assert result["error"] is None
        output = result["output"]
        assert "answer" in output
        assert output["mode"] == "local"

    def test_call_global_search(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        from src.skills.base.skill_manager import SkillManager
        manager = SkillManager()
        manager.register(GraphRAGSearcherSkill)

        result = manager.call(
            "graphrag_searcher",
            query="整个知识库的主题是什么",
            mode="global",
            top_k_communities=3,
        )

        assert result["status"] == "success"
        assert result["output"]["mode"] == "global"

    def test_call_hybrid_search(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        from src.skills.base.skill_manager import SkillManager
        manager = SkillManager()
        manager.register(GraphRAGSearcherSkill)

        result = manager.call(
            "graphrag_searcher",
            query="Entity_0 在全局视角下如何理解",
            mode="hybrid",
        )

        assert result["status"] == "success"
        assert result["output"]["mode"] == "hybrid"

    def test_call_invalid_mode(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        from src.skills.base.skill_manager import SkillManager
        manager = SkillManager()
        manager.register(GraphRAGSearcherSkill)

        result = manager.call(
            "graphrag_searcher",
            query="test",
            mode="invalid_mode",
        )

        assert result["status"] == "success"  # call 层面不报错
        assert result["output"]["error"] is not None  # 但输出里有错误
        assert "未知模式" in result["output"]["error"]


# ============================================================
# 边界 & 异常测试
# ============================================================

class TestEdgeCases:
    def test_empty_query(self, engine_with_mocks):
        engine, _, _ = engine_with_mocks
        result = engine.local_search("", top_k=3, max_tokens=1000)
        # 空查询应返回空结果，不崩溃
        assert "answer" in result

    def test_top_k_larger_than_entities(self, engine_with_mocks):
        engine, _, _ = engine_with_mocks
        result = engine.local_search("Entity", top_k=9999, max_tokens=1000)
        # entity_count 不超过实际数量
        assert result["entity_count"] <= len(engine.entities)

    def test_concurrent_skill_creation(self, mock_data_dir, mock_llm_client, mock_embedding_client, monkeypatch):
        """多线程同时创建 Skill 实例不应死锁"""
        monkeypatch.setenv("GRAPHRAG_ROOT", str(mock_data_dir))
        configure_graphrag_searcher(
            llm_client=mock_llm_client,
            embedding_client=mock_embedding_client,
        )

        skills = []

        def create_skill():
            s = GraphRAGSearcherSkill()
            skills.append(s)

        threads = [threading.Thread(target=create_skill) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(skills) == 5
        # 所有 skill 共享同一个引擎
        first_engine = skills[0]._engine
        for s in skills[1:]:
            assert s._engine is first_engine


# ============================================================
# Pydantic 输出测试
# ============================================================

@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic 未安装")
class TestOutputModel:
    def test_model_validation(self):
        from src.skills.custom.rag_skills.graphrag_searcher.skill import (
            GraphRAGSearcherOutput,
            SearchSource,
        )

        output = GraphRAGSearcherOutput(
            answer="这是一个回答",
            sources=[SearchSource(type="entity", name="Entity_0")],
            mode="local",
            entity_count=5,
            community_count=0,
        )
        assert output.answer == "这是一个回答"
        assert output.sources[0].type == "entity"

    def test_model_dump(self):
        from src.skills.custom.rag_skills.graphrag_searcher.skill import (
            GraphRAGSearcherOutput,
            SearchSource,
        )

        output = GraphRAGSearcherOutput(
            answer="answer",
            sources=[SearchSource(type="entity", name="E1")],
            mode="local",
            entity_count=1,
            community_count=0,
        )
        d = output.model_dump()
        assert isinstance(d, dict)
        assert d["answer"] == "answer"
        assert d["sources"][0]["name"] == "E1"

if __name__ == "__main__":    pytest.main()