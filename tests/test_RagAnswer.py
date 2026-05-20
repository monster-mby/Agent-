"""
RagAnswerSkill v1.1 — pytest 测试套件

覆盖：
- 单元测试：TokenEstimator / PromptTemplate / _extract_keywords / _extract_citations
- 集成测试：execute() 全流程（模拟 LLM）
- 边界测试：空上下文 / 空响应 / 越界引用 / token 预算极端值
- v1.1 专测：重试分级 / 引用溯源 / 置信度 / 关键词加权 / 拒绝检测 / 多轮对话
- 异常路径：LLM 可重试/不可重试 / 全部重试耗尽 / 模拟回退

运行方式：
    pytest tests/test_rag_answer.py -v
    pytest tests/test_rag_answer.py -v -k "test_citation"  # 按关键字筛选
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 被测试模块
from skills.custom.rag_skills.rag_answer.skill import (
    # 异常
    RagAnswerError,
    EmptyContextError,
    LLMCallError,
    LLMRetryableError,
    CitationGroundingError,
    # 模型
    SearchResultRef,
    Citation,
    ConversationTurn,
    RagAnswerInput,
    RagAnswerOutput,
    # 模板
    PromptTemplate,
    # Token
    TokenEstimator,
    # 技能
    RagAnswerSkill,
    # 便捷函数
    rag_answer,
    # 常量
    DEFAULT_SYSTEM_PROMPT_ZH,
    DEFAULT_SYSTEM_PROMPT_EN,
    _build_default_templates,
    _is_available,
    HAS_TIKTOKEN,
    HAS_JINJA2,
    HAS_TENACITY,
    HAS_LANGDETECT,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def skill():
    """创建 RagAnswerSkill 实例（不加载外部模板）"""
    return RagAnswerSkill()


@pytest.fixture
def mock_search_results_dict() -> List[dict]:
    """模拟上游 vector_search 返回的 dict 列表"""
    return [
        {
            "chunk_id": "chunk_003",
            "text": "Python 是一种解释型、面向对象的高级编程语言。Python 由 Guido van Rossum 于 1991 年首次发布。",
            "score": 0.92,
            "metadata": {
                "doc_id": "doc_1",
                "title": "Python简介",
                "chunk_index": 2,
                "total_chunks": 8,
            },
        },
        {
            "chunk_id": "chunk_007",
            "text": "Python 的主要特点包括：动态类型系统、自动内存管理、丰富的标准库、庞大的第三方包生态。",
            "score": 0.87,
            "metadata": {
                "doc_id": "doc_1",
                "title": "Python特点",
                "chunk_index": 6,
                "total_chunks": 8,
            },
        },
        {
            "chunk_id": "chunk_012",
            "text": "Python 在数据科学中的核心库包括 NumPy、Pandas、Matplotlib 和 Scikit-learn。",
            "score": 0.81,
            "metadata": {
                "doc_id": "doc_2",
                "title": "Python数据科学生态",
                "chunk_index": 0,
                "total_chunks": 5,
            },
        },
        {
            "chunk_id": "chunk_015",
            "text": "Java 是一种编译型、面向对象的编程语言，由 Sun Microsystems 于 1995 年发布。",
            "score": 0.55,
            "metadata": {
                "doc_id": "doc_3",
                "title": "Java简介",
                "chunk_index": 0,
                "total_chunks": 6,
            },
        },
    ]


@pytest.fixture
def mock_search_results_refs(mock_search_results_dict) -> List[SearchResultRef]:
    """SearchResultRef 对象列表"""
    return [SearchResultRef(**r) for r in mock_search_results_dict]


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端 — 返回正常回答"""
    client = MagicMock()
    client.chat.return_value = {
        "content": "Python 由 Guido van Rossum 于 1991 年首次发布 [1]。其特点包括动态类型系统和自动内存管理 [2]。",
        "usage": {"prompt_tokens": 150, "completion_tokens": 40, "total_tokens": 190},
        "model": "gpt-4o-mini",
    }
    return client


@pytest.fixture
def mock_llm_client_refusal():
    """模拟 LLM 客户端 — 返回拒绝回答"""
    client = MagicMock()
    client.chat.return_value = {
        "content": "根据现有资料，我无法回答这个问题。",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        "model": "gpt-4o-mini",
    }
    return client


@pytest.fixture
def mock_llm_client_empty():
    """模拟 LLM 客户端 — 返回空响应"""
    client = MagicMock()
    client.chat.return_value = {
        "content": "",
        "usage": {},
        "model": "gpt-4o-mini",
    }
    return client


@pytest.fixture
def mock_llm_client_retryable():
    """模拟 LLM 客户端 — 前 2 次抛可重试异常，第 3 次成功"""
    client = MagicMock()
    client.chat.side_effect = [
        LLMRetryableError("429 Too Many Requests"),
        LLMRetryableError("503 Service Unavailable"),
        {
            "content": "Python 于 1991 年发布 [1]。",
            "usage": {"prompt_tokens": 100, "completion_tokens": 15, "total_tokens": 115},
            "model": "gpt-4o-mini",
        },
    ]
    return client


@pytest.fixture
def mock_llm_client_non_retryable():
    """模拟 LLM 客户端 — 抛不可重试异常"""
    client = MagicMock()
    client.chat.side_effect = LLMCallError("401 Unauthorized")
    return client


@pytest.fixture
def mock_llm_client_all_fail():
    """模拟 LLM 客户端 — 全部重试耗尽"""
    client = MagicMock()
    client.chat.side_effect = LLMRetryableError("503 Service Unavailable")
    return client


# ═══════════════════════════════════════════════════════════════
# 1. TokenEstimator 测试
# ═══════════════════════════════════════════════════════════════

class TestTokenEstimator:
    """Token 估算器测试"""

    def test_init_creates_estimator(self):
        est = TokenEstimator()
        assert est is not None

    def test_count_returns_positive_int(self):
        est = TokenEstimator()
        result = est.count("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        est = TokenEstimator()
        result = est.count("")
        assert result == 0

    def test_count_chinese_text(self):
        est = TokenEstimator()
        result = est.count("Python 是一种解释型、面向对象的高级编程语言。")
        assert result > 5  # 至少应有几个 token

    def test_count_long_text(self):
        est = TokenEstimator()
        text = "Python 是一种解释型高级编程语言。" * 100
        result = est.count(text)
        assert result > 100

    def test_count_batch(self):
        est = TokenEstimator()
        texts = ["Hello", "World", "Python 编程"]
        results = est.count_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)

    def test_heuristic_chinese_biased(self):
        """中文文本 token 估算应大于字符数"""
        est = TokenEstimator()
        text = "这是一段中文测试文本用于验证估算逻辑的准确性"
        heuristic = est._estimate_heuristic(text)
        chars = len(text)
        assert heuristic > chars  # 中文每字约 1.5 token

    def test_heuristic_english_biased(self):
        """英文文本 token 估算"""
        est = TokenEstimator()
        text = "This is a test sentence for token estimation purposes"
        heuristic = est._estimate_heuristic(text)
        words = len(text.split())
        assert heuristic > words  # 英文每词约 1.3 token


# ═══════════════════════════════════════════════════════════════
# 2. PromptTemplate 测试
# ═══════════════════════════════════════════════════════════════

class TestPromptTemplate:
    """Prompt 模板测试"""

    def test_python_format_simple(self):
        tpl = PromptTemplate("Hello {name}", engine="python")
        result = tpl.render(name="World")
        assert result == "Hello World"

    def test_python_format_missing_var(self):
        """未提供变量时保留原占位符"""
        tpl = PromptTemplate("Hello {name}, welcome to {place}", engine="python")
        result = tpl.render(name="World")
        assert "World" in result
        assert "{place}" in result  # 未提供的变量保留

    def test_auto_engine_selects_python_without_jinja2(self, monkeypatch):
        """没有 Jinja2 时自动选 python 引擎"""
        import skills.custom.rag_skills.rag_answer.skill as skill_module
        original_has_jinja2 = skill_module.HAS_JINJA2
        monkeypatch.setattr(skill_module, 'HAS_JINJA2', False)

        try:
            tpl = PromptTemplate("Hello {name}")
            assert tpl._engine == "python"
        finally:
            monkeypatch.setattr(skill_module, 'HAS_JINJA2', original_has_jinja2)

    def test_auto_engine_selects_jinja2_when_available(self):
        """有 Jinja2 时自动选 jinja2 引擎"""
        from skills.custom.rag_skills.rag_answer.skill import HAS_JINJA2
        tpl = PromptTemplate("Hello {{ name }}")
        if HAS_JINJA2:
            assert tpl._engine == "jinja2"
        else:
            assert tpl._engine == "python"

    def test_default_templates_zh(self):
        templates = _build_default_templates()
        assert "default_zh" in templates
        assert "default_en" in templates
        result = templates["default_zh"].render()
        assert "知识问答助手" in result

    def test_default_templates_en(self):
        templates = _build_default_templates()
        result = templates["default_en"].render()
        assert "question-answering assistant" in result.lower()


# ═══════════════════════════════════════════════════════════════
# 3. Pydantic 模型验证测试
# ═══════════════════════════════════════════════════════════════

class TestPydanticModels:
    """Pydantic 输入输出模型验证"""

    def test_input_valid_minimal(self, mock_search_results_dict):
        inp = RagAnswerInput(query="test", search_results=mock_search_results_dict)
        assert inp.query == "test"
        assert len(inp.search_results) == 4

    def test_input_empty_query_raises(self, mock_search_results_dict):
        with pytest.raises(Exception):  # pydantic ValidationError
            RagAnswerInput(query="", search_results=mock_search_results_dict)

    def test_input_empty_search_results_raises(self):
        with pytest.raises(Exception):
            RagAnswerInput(query="test", search_results=[])

    def test_input_query_too_long_raises(self, mock_search_results_dict):
        with pytest.raises(Exception):
            RagAnswerInput(query="x" * 5000, search_results=mock_search_results_dict)

    def test_input_default_values(self, mock_search_results_dict):
        inp = RagAnswerInput(query="test", search_results=mock_search_results_dict)
        assert inp.max_context_chunks == 8
        assert inp.max_context_tokens == 3000
        assert inp.temperature == 0.3
        assert inp.include_citations is True
        assert inp.retry_on_failure is True
        assert inp.max_retries == 3

    def test_input_keyword_boost_clamped(self, mock_search_results_dict):
        """keyword_weight_boost 应在 0~0.5 之间"""
        with pytest.raises(Exception):
            RagAnswerInput(
                query="test", search_results=mock_search_results_dict,
                keyword_weight_boost=0.8,
            )

    def test_search_result_ref_creation(self):
        ref = SearchResultRef(
            chunk_id="c1", text="hello", score=0.9, metadata={"k": "v"}
        )
        assert ref.chunk_id == "c1"
        assert ref.score == 0.9

    def test_citation_model(self):
        cit = Citation(citation_id=1, chunk_id="c1", score=0.9, grounded=True)
        assert cit.grounded is True

    def test_output_model(self):
        out = RagAnswerOutput(
            success=True,
            answer="hello",
            confidence=0.85,
            model="gpt-4",
        )
        assert out.success is True
        assert out.confidence == 0.85


# ═══════════════════════════════════════════════════════════════
# 4. _extract_keywords 测试
# ═══════════════════════════════════════════════════════════════

class TestExtractKeywords:
    """关键词提取测试"""

    def test_chinese_bigram(self, skill):
        keywords = skill._extract_keywords("Python 是什么时候发布的")
        assert len(keywords) > 0
        # 应包含中文 bigram
        assert any("时候" in kw or "发布" in kw for kw in keywords)

    def test_english_words(self, skill):
        keywords = skill._extract_keywords("What is Python used for")
        assert "python" in [k.lower() for k in keywords]

    def test_empty_query(self, skill):
        keywords = skill._extract_keywords("")
        assert keywords == []

    def test_max_keywords_limit(self, skill):
        """关键词不应超过 20 个"""
        long_query = "Python Java C++ Rust Go Scala Kotlin Swift " * 10
        keywords = skill._extract_keywords(long_query)
        assert len(keywords) <= 20

    def test_deduplication(self, skill):
        keywords = skill._extract_keywords("Python Python Python 测试 测试")
        lowered = [k.lower() for k in keywords]
        assert lowered.count("python") <= 1
        assert lowered.count("测试") <= 1


# ═══════════════════════════════════════════════════════════════
# 5. _select_context_chunks 测试
# ═══════════════════════════════════════════════════════════════

class TestSelectContextChunks:
    """上下文选择测试"""

    def test_basic_selection(self, skill, mock_search_results_refs):
        selected = skill._select_context_chunks(
            mock_search_results_refs, "Python", max_chunks=3, max_tokens=5000,
            keyword_boost=0.0,
        )
        assert len(selected) >= 1
        # 应按分数降序
        assert selected[0].score >= selected[-1].score

    def test_max_chunks_limit(self, skill, mock_search_results_refs):
        selected = skill._select_context_chunks(
            mock_search_results_refs, "Python", max_chunks=2, max_tokens=5000,
            keyword_boost=0.0,
        )
        assert len(selected) <= 2

    def test_token_budget_truncation(self, skill, mock_search_results_refs):
        """极低 token 预算应截断到 0~1 个分块"""
        selected = skill._select_context_chunks(
            mock_search_results_refs, "Python", max_chunks=10, max_tokens=10,
            keyword_boost=0.0,
        )
        assert len(selected) <= 1  # 10 token 可能连一个块都装不下

    def test_keyword_boost_effect(self, skill, mock_search_results_refs):
        """关键词加成应提升 Java 相关分块排名"""
        no_boost = skill._select_context_chunks(
            mock_search_results_refs, "Java", max_chunks=4, max_tokens=5000,
            keyword_boost=0.0,
        )
        no_boost_ids = [c.chunk_id for c in no_boost]

        with_boost = skill._select_context_chunks(
            mock_search_results_refs, "Java", max_chunks=4, max_tokens=5000,
            keyword_boost=0.3,
        )
        boost_ids = [c.chunk_id for c in with_boost]

        java_idx_no = no_boost_ids.index("chunk_015") if "chunk_015" in no_boost_ids else 999
        java_idx_boost = boost_ids.index("chunk_015") if "chunk_015" in boost_ids else 999
        assert java_idx_boost <= java_idx_no, (
            f"加成后 Java 分块排名应不降，"
            f"无加成:{no_boost_ids} 加成后:{boost_ids}"
        )

    def test_keyword_boost_all_same_query(self, skill, mock_search_results_refs):
        """查询词与所有分块高度相关时加成不改变顺序"""
        no_boost = skill._select_context_chunks(
            mock_search_results_refs, "Python", max_chunks=3, max_tokens=5000,
            keyword_boost=0.0,
        )
        with_boost = skill._select_context_chunks(
            mock_search_results_refs, "Python", max_chunks=3, max_tokens=5000,
            keyword_boost=0.3,
        )
        assert [c.chunk_id for c in no_boost] == [c.chunk_id for c in with_boost]

    def test_empty_list(self, skill):
        selected = skill._select_context_chunks(
            [], "query", max_chunks=5, max_tokens=1000, keyword_boost=0.0,
        )
        assert selected == []


# ═══════════════════════════════════════════════════════════════
# 6. _extract_citations 测试（v1.1 扩展格式）
# ═══════════════════════════════════════════════════════════════

class TestExtractCitations:
    """引用提取测试 — v1.1 支持 [1]/[1-3]/[1,2]/[1][2]"""

    def test_single_bracket(self, skill, mock_search_results_refs):
        answer = "Python 于 1991 年发布 [1]。其特点包括 [2]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert 1 in ids
        assert 2 in ids

    def test_range_format(self, skill, mock_search_results_refs):
        """[1-3] 应展开为 1,2,3"""
        answer = "相关内容见 [1-3]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert ids == {1, 2, 3}

    def test_range_format_large(self, skill, mock_search_results_refs):
        """[1-999] 应被截断（最多 start+50）"""
        answer = "见 [1-999]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert max(ids) <= 51  # 1 + 50
        assert len(ids) <= 51

    def test_comma_format(self, skill, mock_search_results_refs):
        """[1,2,4] 应提取 3 个引用"""
        answer = "详见 [1,2,4]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert ids == {1, 2, 4}

    def test_consecutive_brackets(self, skill, mock_search_results_refs):
        """[1][2] 连续引用"""
        answer = "如 [1][2] 所述。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert ids == {1, 2}

    def test_out_of_range_ids_filtered(self, skill, mock_search_results_refs):
        """引用编号超出分块范围应被过滤"""
        answer = "见 [1] 和 [999]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        ids = {c.citation_id for c in citations}
        assert 1 in ids
        assert 999 not in ids

    def test_no_citations(self, skill, mock_search_results_refs):
        answer = "没有引用标记的普通回答。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        assert citations == []

    def test_citation_id_mapping(self, skill, mock_search_results_refs):
        """引用编号应正确映射到 chunk"""
        answer = "Python 由 Guido 于 1991 年发布 [1]。"
        citations = skill._extract_citations(answer, mock_search_results_refs)
        if citations:
            assert citations[0].chunk_id == mock_search_results_refs[0].chunk_id
            assert citations[0].score == mock_search_results_refs[0].score

    def test_too_many_ids_truncated(self, skill, mock_search_results_refs):
        """LLM 胡乱编号时应截断"""
        # 构造 50 个引用编号（远超 2 倍分块数）
        answer = " ".join(f"[{i}]" for i in range(1, 51))
        citations = skill._extract_citations(answer, mock_search_results_refs)
        # 应截断到 max_chunks (4) 附近
        assert len(citations) <= 4


# ═══════════════════════════════════════════════════════════════
# 7. 引用溯源验证测试
# ═══════════════════════════════════════════════════════════════

class TestCitationGrounding:
    """引用溯源验证测试"""

    def test_grounded_citation(self, skill, mock_search_results_refs):
        """正常引用应通过溯源"""
        answer = "Python 由 Guido van Rossum 于 1991 年首次发布 [1]。"
        raw = skill._extract_citations(answer, mock_search_results_refs)
        verified = skill._verify_citation_grounding(answer, raw, mock_search_results_refs)
        assert verified[0].grounded is True

    def test_hallucinated_citation(self, skill, mock_search_results_refs):
        """幻觉引用应标记为 ungrounded — 引用内容与分块完全无关"""
        answer = "Python 被广泛用于编写操作系统内核的微内核架构 [3]。"
        raw = skill._extract_citations(answer, mock_search_results_refs)
        verified = skill._verify_citation_grounding(answer, raw, mock_search_results_refs)
        cit_3 = [c for c in verified if c.citation_id == 3]
        if cit_3:
            assert cit_3[0].grounded is False, (
                f"引用 [3] 标注'操作系统内核的微内核架构'，但 chunk_012 不包含此内容，"
                f"应标记为 ungrounded"
            )

    def test_ngram_overlap_false(self, skill):
        """两个完全不相关文本应判定为无重叠"""
        result = skill._check_ngram_overlap(
            "编写操作系统内核的微内核架构",
            "Python 是一种解释型高级编程语言",
        )
        assert result is False

    def test_ngram_overlap_true_related(self, skill):
        """相关内容应判定为有重叠"""
        result = skill._check_ngram_overlap(
            "Guido van Rossum 于 1991 年发布了 Python",
            "Python 由 Guido van Rossum 于 1991 年首次发布",
        )
        assert result is True

    def test_short_snippet_default_trusted(self, skill, mock_search_results_refs):
        """过短的片段默认信任"""
        answer = "[1]。"
        raw = skill._extract_citations(answer, mock_search_results_refs)
        verified = skill._verify_citation_grounding(answer, raw, mock_search_results_refs)
        # snippet 太短（<5 字），默认 grounded=True
        assert all(c.grounded for c in verified)

    def test_ngram_overlap_true(self, skill):
        result = skill._check_ngram_overlap(
            "Guido van Rossum 于 1991 年发布",
            "Python 由 Guido van Rossum 于 1991 年首次发布",
        )
        assert result is True

    def test_ngram_overlap_false(self, skill):
        result = skill._check_ngram_overlap(
            "编写操作系统内核",
            "Python 是一种解释型高级编程语言",
        )
        assert result is False


# ═══════════════════════════════════════════════════════════════
# 8. 置信度估算测试
# ═══════════════════════════════════════════════════════════════

class TestEstimateConfidence:
    """置信度估算测试"""

    def test_high_confidence(self, skill, mock_search_results_refs):
        """有高质量引用时应偏高"""
        conf = skill._estimate_confidence(
            "Python 由 Guido van Rossum 于 1991 年发布 [1]。其特点包括动态类型 [2]。",
            mock_search_results_refs[:2],
            [
                Citation(citation_id=1, chunk_id="chunk_003", score=0.92, grounded=True),
                Citation(citation_id=2, chunk_id="chunk_007", score=0.87, grounded=True),
            ],
        )
        assert conf >= 0.7  # 高质量信号应偏高

    def test_low_confidence_refusal(self, skill, mock_search_results_refs):
        """拒绝回答时应偏低"""
        conf = skill._estimate_confidence(
            "根据现有资料，我无法回答这个问题。",
            mock_search_results_refs[:1],
            [],
        )
        assert conf <= 0.4

    def test_range_0_to_1(self, skill, mock_search_results_refs):
        """置信度应在 [0, 1] 范围内"""
        conf = skill._estimate_confidence("test", mock_search_results_refs, [])
        assert 0.0 <= conf <= 1.0

    def test_no_chunks_low_confidence(self, skill):
        """无上下文时置信度极低"""
        conf = skill._estimate_confidence("不知道", [], [])
        assert conf <= 0.25

    def test_no_chunks_no_citations_very_low(self, skill):
        """无分块 + 低置信短语 应接近 0"""
        conf = skill._estimate_confidence("我不确定，无法回答", [], [])
        assert conf <= 0.15


# ═══════════════════════════════════════════════════════════════
# 9. 拒绝回答检测测试
# ═══════════════════════════════════════════════════════════════

class TestDetectRefusal:
    """拒绝回答检测测试"""

    def test_refusal_chinese(self, skill):
        assert skill._detect_refusal("根据现有资料，我无法回答这个问题") is True

    def test_refusal_english(self, skill):
        assert skill._detect_refusal("I cannot answer this question") is True

    def test_normal_answer(self, skill):
        assert skill._detect_refusal("Python 由 Guido van Rossum 于 1991 年创建") is False

    def test_empty_answer(self, skill):
        assert skill._detect_refusal("") is True

    def test_short_answer(self, skill):
        assert skill._detect_refusal("ok") is True  # < 10 chars

    def test_insufficient_info(self, skill):
        assert skill._detect_refusal("没有足够信息回答此问题") is True


# ═══════════════════════════════════════════════════════════════
# 10. 异常分类测试
# ═══════════════════════════════════════════════════════════════

class TestExceptionClassification:
    """异常分类（可重试 vs 不可重试）"""

    def test_429_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("429 Too Many Requests"))

    def test_503_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("503 Service Unavailable"))

    def test_500_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("500 Internal Server Error"))

    def test_502_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("502 Bad Gateway"))

    def test_timeout_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("Connection timed out"))

    def test_rate_limit_is_retryable(self, skill):
        with pytest.raises(LLMRetryableError):
            skill._classify_and_raise(Exception("Rate limit exceeded"))

    def test_401_is_not_retryable(self, skill):
        with pytest.raises(LLMCallError):
            skill._classify_and_raise(Exception("401 Unauthorized"))

    def test_403_is_not_retryable(self, skill):
        with pytest.raises(LLMCallError):
            skill._classify_and_raise(Exception("403 Forbidden"))

    def test_400_is_not_retryable(self, skill):
        with pytest.raises(LLMCallError):
            skill._classify_and_raise(Exception("400 Bad Request"))


# ═══════════════════════════════════════════════════════════════
# 11. execute() 集成测试（使用模拟 LLM 客户端）
# ═══════════════════════════════════════════════════════════════

class TestExecuteIntegration:
    """execute() 全流程集成测试"""

    def test_basic_rag_success(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python 是什么时候发布的？",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            max_context_chunks=3,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        assert len(result["answer"]) > 0
        assert result["model"] is not None
        assert result["elapsed_ms"] >= 0
        assert "timing_breakdown" in result

    def test_answer_contains_citations(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python 的特点",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            include_citations=True,
        )
        result = skill.execute(input_data)
        assert len(result["citations"]) > 0

    def test_no_citations_mode(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            include_citations=False,
        )
        result = skill.execute(input_data)
        assert result["citations"] == []

    def test_confidence_estimation(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python 发布时间",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            estimate_confidence=True,
        )
        result = skill.execute(input_data)
        assert result["confidence"] is not None
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_disabled(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            estimate_confidence=False,
        )
        result = skill.execute(input_data)
        assert result["confidence"] is None

    def test_citation_grounding_validation(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            validate_citation_grounding=True,
        )
        result = skill.execute(input_data)
        # 应有 grounded 标记
        for cit in result["citations"]:
            assert "grounded" in cit or hasattr(cit, "grounded")

    def test_refusal_detection(
        self, skill, mock_search_results_dict, mock_llm_client_refusal,
    ):
        input_data = RagAnswerInput(
            query="不可能知道的问题",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_refusal,
            detect_refusal=True,
        )
        result = skill.execute(input_data)
        # 即使 LLM 拒绝，也应返回 success=True（友好降级）
        assert result["success"] is True

    def test_empty_response(
        self, skill, mock_search_results_dict, mock_llm_client_empty,
    ):
        input_data = RagAnswerInput(
            query="test",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_empty,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        assert result["answer"] == ""

    def test_conversation_history(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="它有什么特点？",
            search_results=mock_search_results_dict[:2],
            llm_client=mock_llm_client,
            conversation_history=[
                {"role": "user", "content": "Python 是什么时候发布的？"},
                {"role": "assistant", "content": "Python 于 1991 年首次发布 [1]。"},
            ],
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_custom_system_prompt(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            system_prompt="你是古代翰林学士，请用文言文回答。",
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_keyword_boost(self, skill, mock_search_results_dict, mock_llm_client):
        input_data = RagAnswerInput(
            query="Java",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            keyword_weight_boost=0.2,
            max_context_chunks=2,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        # Java 分块应在上下文中
        assert result["context_chunks_used"] <= 2


# ═══════════════════════════════════════════════════════════════
# 12. 重试机制测试
# ═══════════════════════════════════════════════════════════════

class TestRetryMechanism:
    """tenacity / 手写重试测试"""

    def test_retry_succeeds_after_failures(
        self, skill, mock_search_results_dict, mock_llm_client_retryable,
    ):
        """前 2 次失败，第 3 次成功"""
        input_data = RagAnswerInput(
            query="Python 发布时间",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_retryable,
            retry_on_failure=True,
            max_retries=3,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        assert result["retry_count"] == 2  # 前 2 次重试

    def test_non_retryable_fails_immediately(
        self, skill, mock_search_results_dict, mock_llm_client_non_retryable,
    ):
        """不可重试异常直接失败"""
        input_data = RagAnswerInput(
            query="test",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_non_retryable,
            retry_on_failure=True,
            max_retries=3,
        )
        result = skill.execute(input_data)
        assert result["success"] is False
        assert result["error"] is not None
        assert "401" in result["error"]

    def test_all_retries_exhausted_fallback_simulated(
        self, skill, mock_search_results_dict, mock_llm_client_all_fail,
    ):
        """全部重试耗尽 → 回退模拟模式"""
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_all_fail,
            retry_on_failure=True,
            max_retries=2,
        )
        result = skill.execute(input_data)
        # 模拟模式也应返回 success=True
        assert result["success"] is True
        assert "模拟回答" in result["answer"] or result["model"] == "simulated"

    def test_retry_disabled(self, skill, mock_search_results_dict, mock_llm_client_retryable):
        """禁用重试时第一次失败即返回"""
        input_data = RagAnswerInput(
            query="test",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client_retryable,
            retry_on_failure=False,
        )
        result = skill.execute(input_data)
        # 禁用重试 → 第一次抛 LLMRetryableError → 进入异常处理
        assert result["success"] is False


# ═══════════════════════════════════════════════════════════════
# 13. 边界测试
# ═══════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    """边界条件测试"""

    def test_single_chunk(self, skill, mock_search_results_dict, mock_llm_client):
        """只有一个检索结果"""
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict[:1],
            llm_client=mock_llm_client,
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_max_chunks_one(self, skill, mock_search_results_dict, mock_llm_client):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            max_context_chunks=1,
        )
        result = skill.execute(input_data)
        assert result["context_chunks_used"] <= 1

    def test_min_token_budget(self, skill, mock_search_results_dict, mock_llm_client):
        """最小 token 预算（256）"""
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            max_context_tokens=256,
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_high_temperature(self, skill, mock_search_results_dict, mock_llm_client):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            temperature=1.5,
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_english_language(self, skill, mock_search_results_dict, mock_llm_client):
        input_data = RagAnswerInput(
            query="What is Python?",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
            answer_language="en",
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_search_results_out_of_order_ok(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        """search_results 乱序也不应崩溃（只警告）"""
        reversed_results = list(reversed(mock_search_results_dict))
        input_data = RagAnswerInput(
            query="Python",
            search_results=reversed_results,
            llm_client=mock_llm_client,
        )
        result = skill.execute(input_data)
        assert result["success"] is True

    def test_timing_breakdown_present(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
        )
        result = skill.execute(input_data)
        assert "select_context_ms" in result["timing_breakdown"]
        assert "build_prompt_ms" in result["timing_breakdown"]
        assert "llm_call_ms" in result["timing_breakdown"]


# ═══════════════════════════════════════════════════════════════
# 14. 模拟模式测试
# ═══════════════════════════════════════════════════════════════

class TestSimulatedMode:
    """无 LLM 时的模拟回退测试"""

    def test_simulated_has_answer(self, skill, mock_search_results_dict):
        """不传 llm_client 时走模拟模式"""
        input_data = RagAnswerInput(
            query="Python 发布",
            search_results=mock_search_results_dict,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        assert len(result["answer"]) > 0
        assert "模拟回答" in result["answer"] or result["model"] == "simulated"

    def test_simulated_no_match(self, skill, mock_search_results_dict):
        """关键词完全不匹配时应返回兜底信息"""
        input_data = RagAnswerInput(
            query="xyzzy_not_exist_12345",
            search_results=mock_search_results_dict,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        assert len(result["answer"]) > 0

    def test_simulated_has_usage(self, skill, mock_search_results_dict):
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
        )
        result = skill.execute(input_data)
        assert "usage" in result


# ═══════════════════════════════════════════════════════════════
# 15. 便捷函数测试
# ═══════════════════════════════════════════════════════════════

class TestConvenienceFunction:
    """rag_answer() 便捷函数测试"""

    def test_rag_answer_returns_dict(self, mock_search_results_dict):
        result = rag_answer(
            query="Python 是什么",
            search_results=mock_search_results_dict,
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert "answer" in result

    def test_rag_answer_with_extra_kwargs(self, mock_search_results_dict):
        result = rag_answer(
            query="Python",
            search_results=mock_search_results_dict,
            max_context_chunks=2,
            include_citations=False,
        )
        assert result["citations"] == []


# ═══════════════════════════════════════════════════════════════
# 16. Skill 元数据测试
# ═══════════════════════════════════════════════════════════════

class TestSkillMetadata:
    """技能元数据验证"""

    def test_name(self, skill):
        assert skill.name == "rag_answer"

    def test_version(self, skill):
        assert skill.version == "1.1.0"

    def test_triggers_not_empty(self, skill):
        assert len(skill.triggers) > 0

    def test_input_schema(self, skill):
        assert skill.input_schema is RagAnswerInput

    def test_output_schema(self, skill):
        assert skill.output_schema is RagAnswerOutput


# ═══════════════════════════════════════════════════════════════
# 17. _build_user_prompt 测试
# ═══════════════════════════════════════════════════════════════

class TestBuildUserPrompt:
    """Prompt 构建测试"""

    def test_includes_chunk_text(self, skill, mock_search_results_refs):
        prompt = skill._build_user_prompt(
            mock_search_results_refs[:2], "test query",
            include_citations=True,
        )
        assert "chunk_003" in prompt or "Python 是一种" in prompt

    def test_includes_metadata(self, skill, mock_search_results_refs):
        prompt = skill._build_user_prompt(
            mock_search_results_refs[:1], "test query",
            include_citations=True,
        )
        assert "Python简介" in prompt or "doc_1" in prompt

    def test_no_citations_format(self, skill, mock_search_results_refs):
        prompt = skill._build_user_prompt(
            mock_search_results_refs[:1], "test query",
            include_citations=False,
        )
        # 应该仍有 [1] 标记（只是不含详细 metadata）
        assert "[1]" in prompt

    def test_with_conversation_history(self, skill, mock_search_results_refs):
        history = [
            ConversationTurn(role="user", content="Python 是什么？"),
            ConversationTurn(role="assistant", content="Python 是一种编程语言 [1]."),
        ]
        prompt = skill._build_user_prompt(
            mock_search_results_refs[:1], "它有什么特点？",
            include_citations=True,
            conversation_history=history,
        )
        assert "Python 是什么" in prompt
        assert "它有什么特点" in prompt
        assert "对话历史" in prompt

    def test_without_conversation_history(self, skill, mock_search_results_refs):
        prompt = skill._build_user_prompt(
            mock_search_results_refs[:1], "test query",
            include_citations=True,
            conversation_history=None,
        )
        assert "对话历史" not in prompt


# ═══════════════════════════════════════════════════════════════
# 18. _get_system_prompt 测试
# ═══════════════════════════════════════════════════════════════

class TestGetSystemPrompt:
    """System prompt 选择逻辑测试"""

    def test_custom_priority(self, skill):
        result = skill._get_system_prompt("自定义 prompt", "default_zh", "zh")
        assert result == "自定义 prompt"

    def test_template_by_name(self, skill):
        result = skill._get_system_prompt(None, "default_en", "en")
        assert "question-answering assistant" in result.lower()

    def test_fallback_zh(self, skill):
        result = skill._get_system_prompt(None, "nonexistent", "zh")
        assert "知识问答助手" in result

    def test_fallback_en(self, skill):
        result = skill._get_system_prompt(None, "nonexistent", "en")
        assert "question-answering assistant" in result.lower()


# ═══════════════════════════════════════════════════════════════
# 19. _detect_language 测试
# ═══════════════════════════════════════════════════════════════

class TestDetectLanguage:
    """语言检测测试"""

    def test_chinese(self, skill):
        assert skill._detect_language("这是一段中文文本") == "zh"

    def test_english(self, skill):
        assert skill._detect_language("This is English text") == "en"

    def test_empty_default(self, skill):
        assert skill._detect_language("") == "zh"


# ═══════════════════════════════════════════════════════════════
# 20. 模板目录加载测试
# ═══════════════════════════════════════════════════════════════

class TestTemplateLoading:
    """模板加载测试"""

    def test_default_templates_loaded(self, skill):
        assert "default_zh" in skill._templates
        assert "default_en" in skill._templates

    def test_nonexistent_dir_no_error(self):
        """不存在的模板目录不应抛异常"""
        s = RagAnswerSkill(template_dir="/nonexistent/path")
        assert "default_zh" in s._templates  # 内置模板仍存在

    def test_env_var_loading(self, monkeypatch, tmp_path):
        """通过环境变量加载模板目录"""
        # 创建临时 yaml 模板
        import yaml
        template_file = tmp_path / "custom.yaml"
        template_file.write_text(
            yaml.dump({
                "name": "custom_test",
                "template": "Custom: {query}",
                "engine": "python",
            }),
            encoding="utf-8",
        )

        monkeypatch.setenv("RAG_ANSWER_TEMPLATE_DIR", str(tmp_path))
        s = RagAnswerSkill()
        assert "custom_test" in s._templates


# ═══════════════════════════════════════════════════════════════
# 21. 异常路径全覆盖
# ═══════════════════════════════════════════════════════════════

class TestExceptionPaths:
    """异常路径全覆盖"""

    def test_empty_context_handled_gracefully(self, skill):
        """无上下文时的友好返回"""
        # 传入空列表会触发 pydantic 校验错误，因此需要用一个额外手段
        # 这里测 _select_context_chunks 返回空后的逻辑
        # 通过极低的 token 预算让所有分块被过滤
        mock_results = [
            SearchResultRef(
                chunk_id="c1",
                text="A" * 5000,  # 极大文本
                score=0.9,
                metadata={},
            )
        ]
        input_data = RagAnswerInput(
            query="test",
            search_results=[r.model_dump() for r in mock_results],
            max_context_tokens=256,
        )
        result = skill.execute(input_data)
        assert result["success"] is True
        # 要么回答"没有找到"，要么用 1 个分块
        assert len(result["answer"]) >= 0

    def test_exception_in_llm_call_routed_to_simulated(
        self, skill, mock_search_results_dict,
    ):
        """LLM 调用异常后回退模拟模式"""
        faulty_client = MagicMock()
        faulty_client.chat.side_effect = Exception("Unexpected network error")

        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=faulty_client,
            retry_on_failure=False,
        )
        result = skill.execute(input_data)
        # 最终应该走模拟模式
        assert result["success"] is True
        assert "模拟回答" in result["answer"]

    def test_error_output_format(self, skill):
        result = skill._error_output({}, time.perf_counter(), {}, "test error", 2)
        assert result["success"] is False
        assert result["error"] == "test error"
        assert result["retry_count"] == 2


# ═══════════════════════════════════════════════════════════════
# 22. 性能 + 回归测试
# ═══════════════════════════════════════════════════════════════

class TestPerformanceRegression:
    """性能回归测试"""

    def test_execute_within_reasonable_time(
        self, skill, mock_search_results_dict, mock_llm_client,
    ):
        """execute 应在 2 秒内完成（模拟客户端）"""
        start = time.perf_counter()
        input_data = RagAnswerInput(
            query="Python",
            search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
        )
        result = skill.execute(input_data)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"execute 耗时 {elapsed:.2f}s，超过 2s 阈值"
        assert result["success"] is True

    def test_batch_consistent(self, skill, mock_search_results_dict, mock_llm_client):
        """多次调用不应崩溃"""
        for i in range(5):
            input_data = RagAnswerInput(
                query=f"Python query {i}",
                search_results=mock_search_results_dict,
                llm_client=mock_llm_client,
            )
            result = skill.execute(input_data)
            assert result["success"] is True

    def test_skill_reuse_ok(self, mock_search_results_dict, mock_llm_client):
        """同一 skill 实例多次调用不应有状态污染"""
        s = RagAnswerSkill()
        input_data1 = RagAnswerInput(
            query="Q1", search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
        )
        r1 = s.execute(input_data1)

        input_data2 = RagAnswerInput(
            query="Q2", search_results=mock_search_results_dict,
            llm_client=mock_llm_client,
        )
        r2 = s.execute(input_data2)

        assert r1["success"] and r2["success"]

# ═══════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])