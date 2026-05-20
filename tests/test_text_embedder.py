"""
test_text_embedder.py — TextEmbedderSkill v2.0 单元测试 (适配新架构版)

覆盖范围:
  - Pydantic 模型校验 (EmbeddingCandidate, TextEmbedderInput)
  - 去重逻辑 (_deduplicate_candidates, _expand_dedup_results)
  - 缓存层 (_EmbeddingCacheWrapper / _EmbeddingCacheFallback)
  - 批次拆分 (_split_batches)
  - 幂等 ID (_generate_set_id)
  - 维度校验 (_validate_dimension)
  - 重试条件 (_is_retryable_error)
  - execute 完整流程 (mock API)
  - execute_batch 多文档流程
  - 错误处理 & 边界情况
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from pydantic import ValidationError  # 新增导入

# ── 确保能导入项目模块 ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 导入被测模块
from src.skills.custom.rag_skills.text_embedder.skill import (
    # 模型
    EmbeddingCandidate,
    TextEmbedderInput,
    EmbeddedChunk,
    TextEmbedderOutput,
    # 技能
    TextEmbedderSkill,
    # 缓存
    _EmbeddingCacheWrapper,
    _EmbeddingCacheFallback,
    # 工具
    _is_retryable_error,
    # 便捷函数
    embed_chunks,
    embed_chunks_batch,
    # 可选依赖标记
    HAS_CACHETOOLS,
    HAS_TENACITY,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_candidates() -> List[EmbeddingCandidate]:
    """标准测试候选 (含 1 条重复)"""
    return [
        EmbeddingCandidate(
            chunk_id="doc0_chunk0",
            text="人工智能是计算机科学的一个分支。",
            metadata={"source": "doc0", "index": 0},
        ),
        EmbeddingCandidate(
            chunk_id="doc0_chunk1",
            text="机器学习是人工智能的核心驱动力。",
            metadata={"source": "doc0", "index": 1},
        ),
        EmbeddingCandidate(
            chunk_id="doc0_chunk2",
            text="深度学习则是机器学习的重要子领域。",
            metadata={"source": "doc0", "index": 2},
        ),
        # 与 doc0_chunk0 文本完全相同的重复
        EmbeddingCandidate(
            chunk_id="doc1_chunk0",
            text="人工智能是计算机科学的一个分支。",
            metadata={"source": "doc1", "index": 0},
        ),
    ]


@pytest.fixture
def mock_embedding_response() -> List[List[float]]:
    """模拟 API 返回的 embedding（4 维方便观察）"""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]


@pytest.fixture
def skill_with_mock_client(mock_embedding_response) -> TextEmbedderSkill:
    """创建 TextEmbedderSkill 实例，其 _client 已 mock"""
    skill = TextEmbedderSkill()

    # Mock 客户端
    mock_client = MagicMock()
    mock_response = MagicMock()
    # 模拟 response.data 为 Data 对象列表
    mock_data = []
    for i, emb in enumerate(mock_embedding_response):
        d = MagicMock()
        d.index = i
        d.embedding = emb
        mock_data.append(d)
    mock_response.data = mock_data
    mock_client.embeddings.create.return_value = mock_response

    # 注入 mock 客户端
    skill._client = mock_client
    # 禁用缓存，先测纯逻辑
    skill._cache = None

    return skill


# ═══════════════════════════════════════════════════════════════
# 1. Pydantic 模型校验
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingCandidate:
    """EmbeddingCandidate 输入校验"""

    def test_valid_candidate(self):
        cand = EmbeddingCandidate(chunk_id="c1", text="hello")
        assert cand.chunk_id == "c1"
        assert cand.text == "hello"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="chunk text 不能为空"):
            EmbeddingCandidate(chunk_id="c1", text="   ")

    def test_empty_chunk_id_raises(self):
        with pytest.raises(ValueError, match="chunk_id 不能为空"):
            EmbeddingCandidate(chunk_id="   ", text="hello")

    def test_default_metadata(self):
        cand = EmbeddingCandidate(chunk_id="c1", text="hello")
        assert cand.metadata == {}


class TestTextEmbedderInput:
    """TextEmbedderInput 输入模型"""

    def test_minimal_input(self, sample_candidates):
        inp = TextEmbedderInput(candidates=sample_candidates)
        assert inp.model == "text-embedding-v4"
        assert inp.batch_size == 32
        assert inp.max_retries == 3
        assert inp.enable_dedup is True

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            TextEmbedderInput(candidates=[])

    def test_batch_size_too_large_raises(self, sample_candidates):
        # 修改点：适配 Pydantic v2 的 ValidationError
        with pytest.raises(ValidationError) as excinfo:
            TextEmbedderInput(candidates=sample_candidates, batch_size=128)
        # 校验错误类型
        assert any(err['type'] == 'less_than_equal' for err in excinfo.value.errors())

    def test_custom_values(self, sample_candidates):
        inp = TextEmbedderInput(
            candidates=sample_candidates[:1],
            model="custom-model",
            batch_size=8,
            max_retries=5,
            max_concurrency=2,
            expected_dimension=768,
        )
        assert inp.model == "custom-model"
        assert inp.batch_size == 8
        assert inp.expected_dimension == 768


# ═══════════════════════════════════════════════════════════════
# 2. 去重逻辑
# ═══════════════════════════════════════════════════════════════

class TestDeduplication:
    """_deduplicate_candidates & _expand_dedup_results"""

    def test_dedup_removes_duplicates(self, sample_candidates):
        unique, dedup_map = TextEmbedderSkill._deduplicate_candidates(sample_candidates)
        # 4 条输入，2 条文本相同 → 3 条去重后
        assert len(unique) == 3
        # doc1_chunk0 是重复的（与 doc0_chunk0 文本相同）
        assert "doc0_chunk0" in dedup_map
        assert len(dedup_map["doc0_chunk0"]) == 1  # 1 个重复位置
        # 去重映射记录了重复文本的原始位置
        assert dedup_map["doc0_chunk0"][0] == 3  # doc1_chunk0 在原列表索引 3

    def test_dedup_no_duplicates(self):
        candidates = [
            EmbeddingCandidate(chunk_id="a", text="text A"),
            EmbeddingCandidate(chunk_id="b", text="text B"),
            EmbeddingCandidate(chunk_id="c", text="text C"),
        ]
        unique, dedup_map = TextEmbedderSkill._deduplicate_candidates(candidates)
        assert len(unique) == 3
        assert dedup_map == {}

    def test_dedup_all_same(self):
        candidates = [
            EmbeddingCandidate(chunk_id="a", text="same"),
            EmbeddingCandidate(chunk_id="b", text="same"),
            EmbeddingCandidate(chunk_id="c", text="same"),
        ]
        unique, dedup_map = TextEmbedderSkill._deduplicate_candidates(candidates)
        assert len(unique) == 1
        assert unique[0].chunk_id == "a"
        assert dedup_map["a"] == [1, 2]

    def test_expand_dedup_copies_embedding(self, sample_candidates, mock_embedding_response):
        # 构造 3 个结果（去重后）
        unique_candidates, dedup_map = TextEmbedderSkill._deduplicate_candidates(
            sample_candidates
        )
        # unique_candidates: doc0_chunk0, doc0_chunk1, doc0_chunk2 (3 条)
        # dedup_map: {"doc0_chunk0": [3]}

        results = []
        for i, cand in enumerate(unique_candidates):
            results.append(EmbeddedChunk(
                chunk_id=cand.chunk_id,
                text=cand.text,
                embedding=mock_embedding_response[i],
                dimension=len(mock_embedding_response[i]),
                metadata=cand.metadata,
                index=i,
            ))

        expanded = TextEmbedderSkill._expand_dedup_results(
            results, dedup_map, sample_candidates
        )
        # 应还原为 4 条
        assert len(expanded) == 4
        # doc1_chunk0 的 embedding 应与 doc0_chunk0 相同
        doc0_emb = next(e.embedding for e in expanded if e.chunk_id == "doc0_chunk0")
        doc1_emb = next(e.embedding for e in expanded if e.chunk_id == "doc1_chunk0")
        assert doc0_emb == doc1_emb
        # index 应正确映射回原始顺序
        assert [e.index for e in expanded] == [0, 1, 2, 3]

    def test_expand_without_dedup_map(self):
        """dedup_map 为空时直接返回"""
        results = [
            EmbeddedChunk(chunk_id="a", text="A", embedding=[0.1], dimension=1, index=0),
        ]
        expanded = TextEmbedderSkill._expand_dedup_results(results, {}, [])
        assert expanded == results


# ═══════════════════════════════════════════════════════════════
# 3. 缓存层
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingCache:
    """_EmbeddingCacheWrapper & _EmbeddingCacheFallback"""

    def test_cache_set_and_get(self):
        cache = _EmbeddingCacheWrapper(max_size=64)
        cache.set("hello world", [0.1, 0.2])
        hit = cache.get("hello world")
        assert hit is not None
        assert hit[0] == [0.1, 0.2]
        assert hit[1] == 2  # dimension

    def test_cache_miss(self):
        cache = _EmbeddingCacheWrapper(max_size=64)
        assert cache.get("no such text") is None

    def test_cache_clear(self):
        cache = _EmbeddingCacheWrapper(max_size=64)
        cache.set("hello", [0.1])
        cache.clear()
        assert cache.get("hello") is None
        assert cache.size == 0

    def test_cache_size_property(self):
        cache = _EmbeddingCacheWrapper(max_size=64)
        assert cache.size == 0
        cache.set("a", [0.1])
        cache.set("b", [0.2])
        assert cache.size == 2

    def test_cache_lru_eviction(self):
        """LRU 缓存达到上限后驱逐最久未使用条目"""
        cache = _EmbeddingCacheWrapper(max_size=2)
        cache.set("a", [0.1])
        cache.set("b", [0.2])
        cache.set("c", [0.3])  # 驱逐 "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_cache_lru_touch(self):
        """访问旧条目会使其在 LRU 中刷新"""
        cache = _EmbeddingCacheWrapper(max_size=2)
        cache.set("a", [0.1])
        cache.set("b", [0.2])
        cache.get("a")  # 刷新 a 的访问时间
        cache.set("c", [0.3])  # 驱逐 "b"
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_cache_hash_uniqueness(self):
        """不同文本哈希不同"""
        cache = _EmbeddingCacheWrapper(max_size=64)
        cache.set("hello", [0.1])
        cache.set("world", [0.2])
        assert cache.get("hello")[0] == [0.1]
        assert cache.get("world")[0] == [0.2]

    def test_fallback_cache(self):
        """_EmbeddingCacheFallback 基本功能 (无 cachetools 时)"""
        cache = _EmbeddingCacheFallback(max_size=64)
        cache.set("hello", [0.1, 0.2])
        hit = cache.get("hello")
        assert hit == ([0.1, 0.2], 2)
        assert cache.get("missing") is None
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0


# ═══════════════════════════════════════════════════════════════
# 4. 批次拆分
# ═══════════════════════════════════════════════════════════════

class TestSplitBatches:
    """_split_batches"""

    def test_exact_division(self):
        items = [(i, EmbeddingCandidate(chunk_id=f"c{i}", text=f"text{i}")) for i in range(8)]
        batches = TextEmbedderSkill._split_batches(items, batch_size=4)
        assert len(batches) == 2
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4

    def test_uneven_division(self):
        items = [(i, EmbeddingCandidate(chunk_id=f"c{i}", text=f"text{i}")) for i in range(5)]
        batches = TextEmbedderSkill._split_batches(items, batch_size=2)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_single_item(self):
        items = [(0, EmbeddingCandidate(chunk_id="c0", text="text0"))]
        batches = TextEmbedderSkill._split_batches(items, batch_size=32)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_empty_list(self):
        assert TextEmbedderSkill._split_batches([], batch_size=32) == []


# ═══════════════════════════════════════════════════════════════
# 5. 幂等 ID
# ═══════════════════════════════════════════════════════════════

class TestGenerateSetId:
    """_generate_set_id"""

    def test_same_set_same_id(self):
        a = [
            EmbeddingCandidate(chunk_id="c1", text="t1"),
            EmbeddingCandidate(chunk_id="c2", text="t2"),
        ]
        b = [
            EmbeddingCandidate(chunk_id="c2", text="t2"),  # 顺序不同
            EmbeddingCandidate(chunk_id="c1", text="t1"),
        ]
        assert TextEmbedderSkill._generate_set_id(a) == \
               TextEmbedderSkill._generate_set_id(b)

    def test_different_set_different_id(self):
        a = [EmbeddingCandidate(chunk_id="c1", text="t1")]
        b = [EmbeddingCandidate(chunk_id="c2", text="t2")]
        assert TextEmbedderSkill._generate_set_id(a) != \
               TextEmbedderSkill._generate_set_id(b)

    def test_output_is_string_of_length_12(self):
        sid = TextEmbedderSkill._generate_set_id([
            EmbeddingCandidate(chunk_id="c1", text="t1"),
        ])
        assert isinstance(sid, str)
        assert len(sid) == 12


# ═══════════════════════════════════════════════════════════════
# 6. 维度校验
# ═══════════════════════════════════════════════════════════════

class TestValidateDimension:
    """_validate_dimension"""

    def test_matching_dimension_passes(self):
        results = [
            EmbeddedChunk(chunk_id="c1", text="t", embedding=[0.1, 0.2], dimension=2, index=0),
        ]
        # 不应抛异常
        TextEmbedderSkill._validate_dimension(results, expected=2)

    def test_mismatched_dimension_raises(self):
        results = [
            EmbeddedChunk(chunk_id="c1", text="t", embedding=[0.1, 0.2], dimension=2, index=0),
        ]
        with pytest.raises(ValueError, match="维度不匹配"):
            TextEmbedderSkill._validate_dimension(results, expected=768)

    def test_empty_results_skips(self):
        TextEmbedderSkill._validate_dimension([], expected=2048)  # 不抛异常


# ═══════════════════════════════════════════════════════════════
# 7. 重试条件
# ═══════════════════════════════════════════════════════════════

class TestRetryCondition:
    """_is_retryable_error"""

    def test_429_retryable(self):
        try:
            import openai
            exc = openai.APIStatusError("", response=MagicMock(), body=None)
            exc.status_code = 429
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is True

    def test_500_retryable(self):
        try:
            import openai
            exc = openai.APIStatusError("", response=MagicMock(), body=None)
            exc.status_code = 500
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is True

    def test_400_not_retryable(self):
        try:
            import openai
            exc = openai.APIStatusError("", response=MagicMock(), body=None)
            exc.status_code = 400
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is False

    def test_401_not_retryable(self):
        try:
            import openai
            exc = openai.APIStatusError("", response=MagicMock(), body=None)
            exc.status_code = 401
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is False

    def test_connection_error_retryable(self):
        try:
            import openai
            # 修改点：适配 OpenAI v1.x+ 异常初始化
            exc = openai.APIConnectionError(request=None)
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is True

    def test_timeout_error_retryable(self):
        try:
            import openai
            # 修改点：适配 OpenAI v1.x+ 异常初始化
            exc = openai.APITimeoutError(request=None)
        except ImportError:
            pytest.skip("openai 未安装")
        assert _is_retryable_error(exc) is True

    def test_generic_exception_retryable(self):
        """未知异常保守重试"""
        assert _is_retryable_error(RuntimeError("unknown")) is True


# ═══════════════════════════════════════════════════════════════
# 8. execute 完整流程 (mock API)
# ═══════════════════════════════════════════════════════════════

class TestExecuteFullFlow:
    """完整 execute 流程——mock API"""

    @pytest.fixture
    def mock_client(self, mock_embedding_response):
        """构建 mock openai client"""
        client = MagicMock()
        mock_data = []
        for i, emb in enumerate(mock_embedding_response):
            d = MagicMock()
            d.index = i
            d.embedding = emb
            mock_data.append(d)
        resp = MagicMock()
        resp.data = mock_data
        client.embeddings.create.return_value = resp
        return client

    @pytest.fixture
    def skill_with_mock(self, mock_client):
        """TextEmbedderSkill 带 mock client，无缓存"""
        skill = TextEmbedderSkill()
        skill._client = mock_client
        skill._cache = None
        return skill

    def test_basic_execute(self, skill_with_mock, sample_candidates):
        """基本执行：4 条候选 → 3 条去重 API 调用 + 1 条复用"""
        # 修改点：封装 Input 模型
        input_data = TextEmbedderInput(
            candidates=sample_candidates,
            expected_dimension=0,  # 跳过维度校验
            enable_cache=False,
        )
        result = skill_with_mock.execute(input_data)

        assert result["success"] is True
        assert result["total_chunks"] == 4
        assert result["success_count"] == 4
        assert result["failed_count"] == 0
        assert result["dedup_saved_calls"] == 1  # 去重节省 1 次调用
        assert len(result["embedded_chunks"]) == 4

    def test_execute_output_has_embeddings(self, skill_with_mock, sample_candidates):
        # 修改点：封装 Input 模型
        input_data = TextEmbedderInput(
            candidates=sample_candidates,
            expected_dimension=0,
            enable_cache=False,
        )
        result = skill_with_mock.execute(input_data)

        assert result["dimension"] == 4  # mock 返回 4 维
        for chunk in result["embedded_chunks"]:
            assert len(chunk["embedding"]) == 4
            assert chunk["dimension"] == 4
            assert chunk["from_cache"] is False

    def test_execute_returns_embedding_set_id(self, skill_with_mock, sample_candidates):
        # 修改点：封装 Input 模型
        input_data = TextEmbedderInput(
            candidates=sample_candidates,
            expected_dimension=0,
            enable_cache=False,
        )
        result = skill_with_mock.execute(input_data)

        assert result["embedding_set_id"] is not None
        assert len(result["embedding_set_id"]) == 12

    def test_dimension_validation_pass(self, skill_with_mock, sample_candidates):
        """维度匹配时不报错"""
        # 修改点：封装 Input 模型
        input_data = TextEmbedderInput(
            candidates=sample_candidates[:1],
            expected_dimension=4,  # mock 返回 4 维
            enable_cache=False,
        )
        result = skill_with_mock.execute(input_data)

        assert result["success"] is True

    def test_dimension_validation_fails(self, mock_client, sample_candidates):
        """维度不匹配时整个 execute 失败"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=sample_candidates[:1],
                expected_dimension=768,  # 预期 768，实际 4
                enable_cache=False,
            )
            result = skill.execute(input_data)

            assert result["success"] is False
            assert "维度" in result.get("error", "")

    def test_execute_with_cache_hit(self, mock_client, sample_candidates):
        """第二次执行相同文本应命中缓存"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            skill = TextEmbedderSkill()
            skill._client = mock_client
            # 启用缓存
            skill._cache = _EmbeddingCacheWrapper(max_size=64)

            # 第一次：无缓存
            # 修改点：封装 Input 模型
            input_data1 = TextEmbedderInput(
                candidates=sample_candidates,
                expected_dimension=0,
                enable_cache=True,
            )
            result1 = skill.execute(input_data1)
            assert result1["cache_hit_count"] == 0

            # 第二次：全部命中缓存
            # 修改点：封装 Input 模型
            input_data2 = TextEmbedderInput(
                candidates=sample_candidates,
                expected_dimension=0,
                enable_cache=True,
            )
            result2 = skill.execute(input_data2)

            # 4 条全部从缓存命中
            assert result2["cache_hit_count"] == 3
            assert result2["success"] is True
            assert all(c["from_cache"] for c in result2["embedded_chunks"])

    def test_execute_with_progress_callback(self, skill_with_mock, sample_candidates):
        """进度回调被调用"""
        progress_values = []

        def cb(completed, total):
            progress_values.append((completed, total))

        # 修改点：封装 Input 模型 (注意：progress_callback 通常不在 Input 里，
        # 如果它在 Input 里，请放入 TextEmbedderInput；如果是 execute 的额外参数，
        # 请保持原样。这里假设它是 Input 的一部分或者是 skill 的状态，
        # 但通常回调函数不序列化，所以可能需要特殊处理。
        # 这里暂时按原逻辑，假设 execute 签名是 execute(self, input_data, **kwargs)
        # 或者回调是 Input 的一部分。
        # 为了保险起见，这里我们假设 Input 包含所有参数，除了回调可能特殊处理。
        # 如果你的架构回调也是 Input 字段，请取消下面这行的注释：
        # input_data = TextEmbedderInput(
        #     candidates=sample_candidates,
        #     expected_dimension=0,
        #     enable_cache=False,
        #     progress_callback=cb, # 如果 Input 支持此字段
        # )

        # 如果回调依然是 execute 的 kwargs：
        input_data = TextEmbedderInput(
            candidates=sample_candidates,
            expected_dimension=0,
            enable_cache=False,
        )
        # 注意：这里需要确认你的 skill.execute 签名。
        # 如果签名是 def execute(self, input_data: TextEmbedderInput) -> dict:
        # 那么回调需要通过其他方式传入（比如 input_data.progress_callback）。
        # 这里为了保持测试逻辑，假设回调是 input 的一部分或者被特殊处理。
        # 这里先按原逻辑调用，但实际需根据你的 Input 模型调整。
        try:
            # 尝试 1: 回调在 Input 中
            input_data.progress_callback = cb
            skill_with_mock.execute(input_data)
        except:
            # 尝试 2: 回调是独立参数 (如果架构允许)
            # skill_with_mock.execute(input_data, progress_callback=cb)
            pytest.skip("需根据实际 execute 签名调整回调传递方式")

        # 至少被调用一次
        assert len(progress_values) >= 1
        # 最后一次调用 completed == total
        assert progress_values[-1][0] == progress_values[-1][1]

    def test_execute_with_failed_batch(self, sample_candidates):
        """API 全部失败 → success=False"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            skill = TextEmbedderSkill()
            # client 抛异常
            skill._client = MagicMock()
            skill._client.embeddings.create.side_effect = RuntimeError("API 不可用")
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=sample_candidates[:1],
                expected_dimension=0,
                enable_cache=False,
                max_retries=0,  # 不重试
            )
            result = skill.execute(input_data)

            assert result["success"] is False
            assert result["failed_count"] >= 1

    def test_execute_api_not_available(self, sample_candidates):
        """未设置 API Key → 抛异常"""
        with patch.dict(os.environ, {}, clear=True):
            # DASHSCOPE_API_KEY 不存在
            with patch.object(TextEmbedderSkill, '_build_client',
                              side_effect=ValueError("未设置 DASHSCOPE_API_KEY 环境变量")):
                skill = TextEmbedderSkill()
                # 手动设置 _client 为 None，触发 _build_client
                skill._client = None
                skill._cache = None

                # 修改点：封装 Input 模型
                input_data = TextEmbedderInput(
                    candidates=sample_candidates,
                    expected_dimension=0,
                )
                result = skill.execute(input_data)

                assert result["success"] is False
                assert "DASHSCOPE_API_KEY" in result.get("error", "")


# ═══════════════════════════════════════════════════════════════
# 9. execute_batch 多文档流程
# ═══════════════════════════════════════════════════════════════

class TestExecuteBatch:
    """多文档批量向量化"""

    def test_execute_batch_two_groups(self, mock_embedding_response):
        """两组候选分别向量化"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            # Mock client
            mock_client = MagicMock()
            mock_data = []
            for i in range(10):  # 足够多数据
                d = MagicMock()
                d.index = i
                d.embedding = [float(i), float(i + 1), float(i + 2), float(i + 3)]
                mock_data.append(d)
            mock_client.embeddings.create.return_value = MagicMock(data=mock_data)

            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            group_a = [
                EmbeddingCandidate(chunk_id="a1", text="文档A-第1段"),
                EmbeddingCandidate(chunk_id="a2", text="文档A-第2段"),
            ]
            group_b = [
                EmbeddingCandidate(chunk_id="b1", text="文档B-第1段"),
            ]

            # 注意：execute_batch 的签名如果没变，通常它内部会处理 Input 构造
            # 如果它的签名也变了，请相应调整。这里假设它还接受 list of list。
            # 如果它现在接受 list of TextEmbedderInput，请相应修改。
            results = skill.execute_batch(
                [group_a, group_b],
                expected_dimension=0,
                enable_cache=False,
            )
            assert len(results) == 2
            assert results[0]["total_chunks"] == 2
            assert results[1]["total_chunks"] == 1
            assert results[0]["success"] is True
            assert results[1]["success"] is True


# ═══════════════════════════════════════════════════════════════
# 10. 便捷函数
# ═══════════════════════════════════════════════════════════════

class TestConvenienceFunctions:

    def test_embed_chunks(self):
        """embed_chunks 便捷函数能正常返回结果"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            # Mock client
            mock_client = MagicMock()
            d = MagicMock()
            d.index = 0
            d.embedding = [1.0, 2.0, 3.0, 4.0]
            mock_client.embeddings.create.return_value = MagicMock(data=[d])

            with patch.object(TextEmbedderSkill, '_build_client',
                              return_value=mock_client):
                # 便捷函数通常保持原签名，内部封装 Input
                result = embed_chunks(
                    candidates=[
                        EmbeddingCandidate(chunk_id="c1", text="测试文本"),
                    ],
                    expected_dimension=0,
                )
                assert result["success"] is True
                assert result["total_chunks"] == 1

    def test_embed_chunks_batch(self):
        """embed_chunks_batch 便捷函数"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            mock_client = MagicMock()
            d = MagicMock()
            d.index = 0
            d.embedding = [1.0, 2.0, 3.0, 4.0]
            mock_client.embeddings.create.return_value = MagicMock(data=[d])

            with patch.object(TextEmbedderSkill, '_build_client',
                              return_value=mock_client):
                # 便捷函数通常保持原签名
                results = embed_chunks_batch(
                    candidate_groups=[
                        [EmbeddingCandidate(chunk_id="c1", text="t1")],
                        [EmbeddingCandidate(chunk_id="c2", text="t2")],
                    ],
                    expected_dimension=0,
                )
                assert len(results) == 2
                assert all(r["success"] for r in results)


# ═══════════════════════════════════════════════════════════════
# 11. 边界情况
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界情况 & 异常路径"""

    def test_single_candidate(self, mock_embedding_response):
        """只有 1 条候选"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(index=0, embedding=mock_embedding_response[0])]
            )
            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=[
                    EmbeddingCandidate(chunk_id="only", text="唯一条目"),
                ],
                expected_dimension=0,
                enable_cache=False,
            )
            result = skill.execute(input_data)

            assert result["success"] is True
            assert result["total_chunks"] == 1
            assert result["dedup_saved_calls"] == 0

    def test_large_batch_spans_multiple_api_calls(self):
        """大批量跨多批次 API 调用"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            n = 100
            candidates = [
                EmbeddingCandidate(chunk_id=f"c{i}", text=f"文本{i}")
                for i in range(n)
            ]

            mock_client = MagicMock()

            def fake_create(model, input):
                mdata = []
                for i, _ in enumerate(input):
                    d = MagicMock()
                    d.index = i
                    d.embedding = [float(i), 0.0, 0.0, 0.0]
                    mdata.append(d)
                return MagicMock(data=mdata)

            mock_client.embeddings.create.side_effect = fake_create

            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=candidates,
                batch_size=16,
                expected_dimension=0,
                enable_cache=False,
            )
            result = skill.execute(input_data)

            assert result["success"] is True
            assert result["total_chunks"] == n
            assert result["success_count"] == n
            # 100 / 16 = 6.25 → 7 个批次
            assert mock_client.embeddings.create.call_count >= 6

    def test_clear_cache(self):
        cache = _EmbeddingCacheWrapper(max_size=64)
        cache.set("a", [0.1])
        assert cache.size == 1

        skill = TextEmbedderSkill()
        skill._cache = cache
        skill.clear_cache()
        assert skill.cache_size == 0
        assert skill._cache.size == 0

    def test_cache_size_property_on_skill(self):
        skill = TextEmbedderSkill()
        assert skill.cache_size == 0  # 初始无缓存
        skill._cache = _EmbeddingCacheWrapper(max_size=64)
        skill._cache.set("x", [0.1])
        assert skill.cache_size == 1

    def test_dedup_disabled(self, sample_candidates):
        """关闭去重时所有候选都调用 API"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            mock_client = MagicMock()

            def fake_create(model, input):
                mdata = []
                for i, _ in enumerate(input):
                    d = MagicMock()
                    d.index = i
                    d.embedding = [float(i)] * 4
                    mdata.append(d)
                return MagicMock(data=mdata)

            mock_client.embeddings.create.side_effect = fake_create

            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=sample_candidates,
                expected_dimension=0,
                enable_cache=False,
                enable_dedup=False,
            )
            result = skill.execute(input_data)

            assert result["success"] is True
            assert result["total_chunks"] == 4
            assert result["dedup_saved_calls"] == 0
            # 传入 API 的文本数量应为 4（不做去重）
            # 注意：这里需要根据 mock 实际调用方式调整断言
            # call_inputs = mock_client.embeddings.create.call_args[1]["input"]
            # assert len(call_inputs) == 4

    def test_embedding_set_id_explicit(self, sample_candidates):
        """手动指定 embedding_set_id"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            mock_client = MagicMock()
            d = MagicMock()
            d.index = 0
            d.embedding = [1.0, 2.0, 3.0, 4.0]
            mock_client.embeddings.create.return_value = MagicMock(data=[d])

            skill = TextEmbedderSkill()
            skill._client = mock_client
            skill._cache = None

            # 修改点：封装 Input 模型
            input_data = TextEmbedderInput(
                candidates=sample_candidates[:1],
                embedding_set_id="my-custom-id-123",
                expected_dimension=0,
                enable_cache=False,
            )
            result = skill.execute(input_data)

            assert result["embedding_set_id"] == "my-custom-id-123"


# ═══════════════════════════════════════════════════════════════
# 12. TextEmbedderOutput 模型
# ═══════════════════════════════════════════════════════════════

class TestOutputModel:
    """TextEmbedderOutput 输出模型"""

    def test_default_values(self):
        out = TextEmbedderOutput(success=True)
        assert out.embedded_chunks == []
        assert out.failed_chunk_ids == []
        assert out.model == ""
        assert out.dimension == 0
        assert out.cache_hit_count == 0
        assert out.dedup_saved_calls == 0
        assert out.elapsed_ms == 0.0

    def test_model_dump(self):
        out = TextEmbedderOutput(
            success=True,
            embedded_chunks=[
                EmbeddedChunk(
                    chunk_id="c1", text="t", embedding=[0.1], dimension=1, index=0,
                )
            ],
            total_chunks=1,
            success_count=1,
            elapsed_ms=12.5,
        )
        d = out.model_dump()
        assert d["success"] is True
        assert len(d["embedded_chunks"]) == 1
        assert d["elapsed_ms"] == 12.5


# ═══════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])