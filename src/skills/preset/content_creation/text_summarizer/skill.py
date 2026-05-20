"""
Text Summarizer Skill — 文本摘要（重构版 v3.0.0）

输入长文本，输出精炼摘要。
支持：
  - 长度控制（短/中/长）、语言自适应、要点式/段落式风格
  - 多种摘要策略：启发式 / TextRank / LLM（大模型生成式）
  - LLM 后端：OpenAI API（含豆包等兼容端点）/ 本地 HuggingFace 模型
  - Map-Reduce 长文本分块
  - MMR 句子去重（基于 sklearn TF-IDF 余弦相似度）
  - 特殊文本保护、Markdown 输入
  - 配置外部化

Changelog:
    v3.0.0: 大幅重构 — 用成熟库替代通用 NLP 工具代码
        - 英文分句: nltk.sent_tokenize 替代手写正则（覆盖数百种缩写）
        - 语言检测: langdetect 替代字符统计（55 语言，准确率 95%+）
        - Markdown 净化: markdown+bs4 替代 9 个正则（完整 GFM 支持）
        - TextRank 相似度: sklearn TF-IDF 余弦替代手写 Jaccard
        - MMR 去重: sklearn TF-IDF 余弦替代手写 Jaccard
        - 中文分词: jieba 替代连续 CJK 正则
    v2.1.0: 新增 LLMStrategy（大模型生成式摘要）
    v2.0.0: 全面重构
"""

from __future__ import annotations

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill

# ============================================================================
# Optional dependency imports with graceful fallback
# ============================================================================
logger = logging.getLogger(__name__)

# --- nltk (English sentence tokenizer) ---
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("nltk 未安装，英文分句将使用手写正则。pip install nltk")

# --- langdetect (language detection) ---
try:
    from langdetect import detect as lang_detect
    from langdetect.lang_detect_exception import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning("langdetect 未安装，语言检测将使用字符统计。pip install langdetect")

# --- markdown + bs4 (Markdown to plain text) ---
try:
    import markdown
    from bs4 import BeautifulSoup
    _MARKDOWN_BS4_AVAILABLE = True
except ImportError:
    _MARKDOWN_BS4_AVAILABLE = False
    logger.warning("markdown/bs4 未安装，Markdown 净化将使用正则。pip install markdown beautifulsoup4")

# --- sklearn (TF-IDF + cosine similarity) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn 未安装，相似度计算将回退到 Jaccard。pip install scikit-learn")

# --- jieba (Chinese tokenization) ---
try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，中文分词将使用连续字符匹配。pip install jieba")


# ============================================================================
# Pydantic Schemas
# ============================================================================
class TextSummarizerInput(BaseModel):
    """文本摘要技能输入参数"""
    text: str = Field(..., description="待摘要的原始文本", min_length=1)
    max_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="摘要长度：short(~80字), medium(~150字), long(~300字)"
    )
    style: Literal["bullet", "paragraph"] = Field(
        default="paragraph",
        description="输出风格：bullet=要点式, paragraph=段落式"
    )
    language: Literal["zh", "en", "auto"] = Field(
        default="auto",
        description="输出语言：zh=中文, en=英文, auto=自动检测"
    )
    algorithm: Literal["heuristic", "textrank", "llm", "auto"] = Field(
        default="auto",
        description="摘要算法：heuristic=启发式, textrank=TextRank, llm=大模型生成式, auto=自动选择"
    )
    deduplicate: bool = Field(
        default=True,
        description="是否启用 MMR 句子去重（仅对提取式算法生效）"
    )
    strip_markdown: bool = Field(
        default=False,
        description="是否预处理 Markdown 标记"
    )
    # LLM 专属参数
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM 模型名称。OpenAI: gpt-3.5-turbo/gpt-4; HuggingFace: bart-large-cnn 等"
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM 温度参数（0=确定性, 1=创造性）"
    )
    llm_max_tokens: Optional[int] = Field(
        default=None,
        description="LLM 最大生成 token 数（None=自动根据 max_length 计算）"
    )


class TextSummarizerOutput(BaseModel):
    """文本摘要输出"""
    summary: str = Field(..., description="摘要文本")
    original_length: int = Field(..., description="原文总字符数")
    summary_length: int = Field(..., description="摘要总字符数")
    compression_ratio: float = Field(..., description="压缩比")
    key_points: list[str] = Field(default_factory=list, description="提取的关键要点")
    style: str = Field(default="paragraph", description="输出风格")
    max_length_used: str = Field(default="medium", description="使用的长度配置")
    detected_language: str = Field(default="auto", description="检测到的语言")
    algorithm_used: str = Field(default="heuristic", description="实际使用的算法")
    num_sentences_selected: int = Field(default=0, description="选中的句子数（提取式）")
    llm_model_used: Optional[str] = Field(default=None, description="使用的 LLM 模型名")
    llm_latency_ms: Optional[float] = Field(default=None, description="LLM 推理耗时(ms)")


# ============================================================================
# Pre-compiled Regex — 仅保留必要的作用：
# 将必要的正则表达式在类加载时预编译（re.compile），避免每次调用时重复编译，大幅提升性能。
# v3.0.0 大幅精简了正则数量（90% 的正则被成熟库替代），仅保留核心业务逻辑必需的正则。
# ============================================================================
# v3.0: 英文分句正则已由 nltk.sent_tokenize 替代
# v3.0: 语言检测已由 langdetect 替代
# v3.0: Markdown 净化正则（_RE_MD_BOLD/_RE_MD_ITALIC 等）已由 markdown+bs4 替代
# v3.0: _RE_TOKEN 分词正则已由 jieba 替代

# --- 保留：启发式评分仍需要 ---
_RE_DIGITS = re.compile(r'\d+')
_RE_PROPER_NOUN = re.compile(r'[A-Z][a-z]{2,}')
_RE_CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')  # 语言检测回退 + 关键要点提取
_RE_MULTI_NEWLINE = re.compile(r'\n{2,}')

# --- 保留：中文/通用分句回退 ---
_RE_SPLIT_ZH = re.compile(r'[。！？]+')
_RE_SPLIT_UNIVERSAL = re.compile(r'[。！？\.!\?\n]+')

# --- 保留：特殊块保护 ---
_RE_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_RE_MD_INLINE_CODE = re.compile(r'`[^`]+`')
_RE_TABLE = re.compile(r'(\|.+\|[\r\n]){2,}')

# --- 保留：启发式评分赘语惩罚 ---
_FILLER_PATTERNS_COMPILED: list[re.Pattern] = [
    re.compile(p) for p in [
        r"^总而言之[,，]?",
        r"^换句话说[,，]?",
        r"^众所周知[,，]?",
        r"^正如我们所知[,，]?",
        r"^需要注意的是[,，]?",
        r"^值得一提的是[,，]?",
        r"^此外[,，]?",
        r"^另外[,，]?",
        r"^In\s+summary[,，]?",
        r"^In\s+conclusion[,，]?",
        r"^As\s+we\s+all\s+know[,，]?",
        r"^It\s+is\s+worth\s+noting\s+that[,，]?",
        r"^To\s+sum\s+up[,，]?",
    ]
]


# ============================================================================
# Default Configuration
# DEFAULT_CONFIG（默认配置字典）
# 作用：
# 将所有硬编码参数（目标字符数、权重、Prompt 模板、LLM 配置等）外部化，无需改代码即可调整效果；
# 支持通过外部 YAML 文件覆盖默认配置。
# ============================================================================
DEFAULT_CONFIG = {
    "summarizer": {
        #短 / 中 / 长摘要的目标字符数
        "target_chars": {"short": 80, "medium": 150, "long": 300},
        # 句子目标数量（按比例或绝对值）
        "target_counts": {
            "short": {"method": "ratio", "value": 0.2, "min": 2},
            "medium": {"method": "ratio", "value": 0.33, "min": 3},
            "long": {"method": "ratio", "value": 0.5, "min": 5},
        },
        #启发式算法的各项权重（关键词、数字、位置等）
        "sentence_scoring": {
            "keyword_weight": 3.0,
            "digit_weight": 2.0,
            "proper_noun_weight": 1.0,
            "optimal_length": {"min": 20, "max": 200, "weight": 1.5},
            "long_length_weight": 0.5,
            "position_weight_first": 2.0,
            "position_weight_last": 2.0,
            "filler_penalty": -2.0,
        },
        #TextRank 算法参数（阻尼系数、迭代次数、收敛阈值）
        "textrank": {
            "damping": 0.85,
            "max_iterations": 50,
            "convergence_threshold": 1e-6,
        },
        #MMR 算法的参数（是否启用、阻尼系数、相似度阈值）
        "mmr": {
            "enabled": True,
            "lambda": 0.7,
            "similarity_threshold": 0.75,
        },
        #分句算法参数（最小句子长度）
        "splitter": {
            "min_sentence_length": 3,
        },
        #高权重关键词
        "high_weight_keywords": {
            "zh": [
                "关键", "重要", "核心", "主要", "必须", "因此", "所以",
                "结论", "总结", "发现", "结果表明", "然而", "但是",
                "显著", "明显", "特别", "尤其",
            ],
            "en": [
                "key", "important", "critical", "essential", "therefore",
                "conclusion", "results show", "finding", "significant",
                "notably", "particularly", "however", "crucial",
            ],
        },
        #特殊块处理
        "special_block_handling": {
            "code_block": "omit",
            "inline_code": "keep",
            "table": "compress",
        },
        #LLM 参数LLM 后端、默认模型、超时重试、Map-Reduce 分块、Prompt 模板
        "llm": {
            "backend": "openai",
            "default_model": "ep-20260321194105-hkdkp",  # 豆包模型
            "timeout_seconds": 120,
            "retry_count": 2,
            "retry_delay_seconds": 2,
            "fallback_to_extractive": True,
            "map_reduce": {
                "enabled": True,
                "chunk_max_chars": 3000,
                "chunk_overlap_chars": 200,
                "max_chunks": 8,
            },
            "prompts": {
                "zh": {
                    "system": (
                        "你是一位专业的文本摘要助手。请根据用户提供的文本生成精炼的摘要。"
                        "严格遵循以下要求：\n"
                        "- 语言：中文\n"
                        "- 风格：{style}\n"
                        "- 长度：{length_desc}（约 {target_chars} 字以内）\n"
                        "- 不要添加原文中没有的信息\n"
                        "- 不要使用\"本文\"\"作者\"等元描述词汇"
                    ),
                    "user_single": "请为以下文本生成摘要：\n\n{text}",
                    "user_map": "请为以下文本片段生成一段简短摘要（约 {chunk_target} 字）：\n\n{text}",
                    "user_reduce": (
                        "以下是将一篇长文档分块后的各段摘要。"
                        "请将它们整合为一段连贯的整体摘要（约 {target_chars} 字），{style_instruction}：\n\n"
                        "{chunk_summaries}"
                    ),
                    "style_bullet": "使用要点式（bullet points）列出，每行以\"•\"开头",
                    "style_paragraph": "使用段落式叙述",
                    "length_labels": {
                        "short": "简短（约 {chars} 字）",
                        "medium": "中等（约 {chars} 字）",
                        "long": "较长（约 {chars} 字）",
                    },
                },
                "en": {
                    "system": (
                        "You are a professional text summarization assistant. "
                        "Generate a concise summary based on the user's text. "
                        "Strictly follow these requirements:\n"
                        "- Language: English\n"
                        "- Style: {style}\n"
                        "- Length: {length_desc} (within ~{target_chars} characters)\n"
                        "- Do not add information not present in the original text\n"
                        "- Avoid meta-descriptions like \"this article\" or \"the author\""
                    ),
                    "user_single": "Please summarize the following text:\n\n{text}",
                    "user_map": "Please write a brief summary for this text excerpt (about {chunk_target} chars):\n\n{text}",
                    "user_reduce": (
                        "Below are summaries of chunks from a longer document. "
                        "Please merge them into one coherent overall summary "
                        "(within {target_chars} chars), {style_instruction}:\n\n"
                        "{chunk_summaries}"
                    ),
                    "style_bullet": "using bullet points starting with \"•\"",
                    "style_paragraph": "in paragraph form",
                    "length_labels": {
                        "short": "short (~{chars} chars)",
                        "medium": "medium (~{chars} chars)",
                        "long": "long (~{chars} chars)",
                    },
                },
            },
        },
    }
}


# ============================================================================
# Abstract Strategy
# ============================================================================


# ============================================================================
#（1）SummarizationStrategy（抽象基类）作用：
#定义所有摘要策略的统一接口 summarize，确保算法可插拔，新增算法只需继承该类。
# ============================================================================
class SummarizationStrategy(ABC):
    """摘要算法抽象基类"""

    @abstractmethod
    def summarize(
        self,
        text: str,
        sentences: list[str],
        max_sentences: int,
        language: str,
        config: dict,
        **kwargs,
    ) -> list[int] | str:
        """
        返回：提取式 → list[int]（句子索引），生成式 → str（摘要文本）
        """
        ...


# ============================================================================
# Heuristic Strategy（保留 — 核心业务逻辑）（
# 2）HeuristicStrategy（启发式策略）
# 作用：
# 基于人工规则给句子打分（关键词、数字、位置、长度等），选取得分最高的句子拼接摘要。
# ============================================================================
class HeuristicStrategy(SummarizationStrategy):
    """基于规则的启发式评分策略"""

    def summarize(
        self,
        text: str,
        sentences: list[str],
        max_sentences: int,
        language: str,
        config: dict,
        **kwargs,
    ) -> list[int]:
        scoring_cfg = config["sentence_scoring"]
        kw_cfg = config["high_weight_keywords"]
        keywords = kw_cfg.get(language, kw_cfg.get("en", []))

        scored: list[tuple[int, float]] = []
        for i, s in enumerate(sentences):
            score = self._score_sentence(s, i, len(sentences), keywords, scoring_cfg)
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scored[:max_sentences]]
        top_indices.sort()
        return top_indices

    #给单个句子打分
    @staticmethod
    def _score_sentence(
        sentence: str,
        idx: int,
        total: int,
        keywords: list[str],
        cfg: dict,
    ) -> float:
        score = 0.0
        s_lower = sentence.lower()
        for kw in keywords:
            if kw in s_lower:
                score += cfg["keyword_weight"]
        if _RE_DIGITS.search(sentence):
            score += cfg["digit_weight"]
        if _RE_PROPER_NOUN.search(sentence):
            score += cfg["proper_noun_weight"]
        length = len(sentence)
        opt = cfg["optimal_length"]
        if opt["min"] <= length <= opt["max"]:
            score += opt["weight"]
        elif length > opt["max"]:
            score += cfg["long_length_weight"]
        if idx == 0:
            score += cfg["position_weight_first"]
        if idx == total - 1:
            score += cfg["position_weight_last"]
        for pat in _FILLER_PATTERNS_COMPILED:
            if pat.match(sentence):
                score += cfg["filler_penalty"]
                break
        return score


# ============================================================================
# TextRank Strategy — v3.0: 用 sklearn TF-IDF 余弦替代手写 Jaccard
#作用：基于图排序的无监督算法，把句子看作节点，相似的句子之间连边，通过迭代计算「句子重要性」，选取得分最高的句子。
# ============================================================================
class TextRankStrategy(SummarizationStrategy):
    """
    基于 TextRank 的图排序策略。

    v3.0 重构：使用 sklearn TfidfVectorizer + cosine_similarity 构建
    相似度矩阵，替代手写的 Jaccard 词集重叠。PageRank 迭代保持。
    回退：sklearn 不可用时回退到简化 Jaccard。
    """

    def summarize(
        self,
        text: str,
        sentences: list[str],
        max_sentences: int,
        language: str,
        config: dict,
        **kwargs,
    ) -> list[int]:
        n = len(sentences)
        if n <= max_sentences:
            return list(range(n))

        tr_cfg = config["textrank"]

        # 构建相似度矩阵
        if _SKLEARN_AVAILABLE:
            sim = self._build_sim_sklearn(sentences, language)
        else:
            sim = self._build_sim_jaccard(sentences, language)

        # PageRank
        scores = self._pagerank(sim, tr_cfg["damping"], tr_cfg["max_iterations"], tr_cfg["convergence_threshold"])

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:max_sentences]]
        top_indices.sort()
        return top_indices

    # ------------------------------------------------------------------
    # sklearn 余弦相似度（v3.0 新增）用 sklearn 构建 TF-IDF 余弦相似度矩阵 输出句子相似度矩阵
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sim_sklearn(sentences: list[str], language: str) -> np.ndarray:
        """用 sklearn TF-IDF + 余弦相似度构建句子相似度矩阵"""
        tokenizer = TextRankStrategy._get_tokenizer(language)
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            max_features=5000,
        )
        try:
            tfidf = vectorizer.fit_transform(sentences)
        except ValueError:
            # 所有句子对 TF-IDF 无有效特征时回退
            return np.eye(len(sentences))
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0)  # 自相似置零
        return sim

    # ------------------------------------------------------------------
    # Jaccard 回退 手写 Jaccard 相似度矩阵 输出句子相似度矩阵
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sim_jaccard(sentences: list[str], language: str) -> np.ndarray:
        """手写 Jaccard 相似度（sklearn 不可用时的回退）"""
        n = len(sentences)
        tokenizer = TextRankStrategy._get_tokenizer(language)
        tokenized = [set(tokenizer(s)) for s in sentences]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = tokenized[i], tokenized[j]
                if a and b:
                    sim[i][j] = sim[j][i] = len(a & b) / len(a | b)
        return sim

    # ------------------------------------------------------------------
    # 分词器 返回语言适配的分词函数（jieba / 正则） 输出分词列表
    # ------------------------------------------------------------------
    @staticmethod
    def _get_tokenizer(language: str):
        """返回语言适配的分词函数"""
        if language == "zh" and _JIEBA_AVAILABLE:
            return lambda text: list(jieba.cut(text))
        else:
            return lambda text: re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())

    # ------------------------------------------------------------------
    # PageRank（保持算法结构，矩阵运算优化）核心 PageRank 迭代算法
    # ------------------------------------------------------------------
    @staticmethod
    def _pagerank(
        sim: np.ndarray,
        damping: float = 0.85,
        max_iterations: int = 50,
        threshold: float = 1e-6,
    ) -> list[float]:
        n = sim.shape[0]
        if n <= 1:
            return [1.0] * n

        # 归一化出度
        out_sum = sim.sum(axis=1) + 1e-12
        transition = sim / out_sum[:, np.newaxis]

        scores = np.ones(n) / n
        for _ in range(max_iterations):
            new_scores = (1 - damping) / n + damping * transition.T @ scores
            if np.abs(new_scores - scores).max() < threshold:
                scores = new_scores
                break
            scores = new_scores
        return scores.tolist()


    """
    summarize (总控)
  ├─> _get_tokenizer (提供分词工具)根据语言和库的可用性，返回一把 “合适的分词刀
  │       └─> 被 _build_sim_sklearn / _build_sim_jaccard 调用
  │
  ├─> _build_sim_sklearn (优先) / _build_sim_jaccard (回退)
  │       └─> 把每两个句子拿过来，算一算 “它们有多像”，填进一个 n×n 的表格（相似度矩阵）里
  │
  └─> _pagerank:基于相似度矩阵（“谁和谁是朋友”），用 PageRank 算法进行多轮投票：
          └─> 接收相似度矩阵，产出分数
    """


# ============================================================================
# LLM Strategy — 抽象基类作用：
# 实现 LLM 生成式摘要的通用逻辑（Map-Reduce 分块、Prompt 组装、重试机制、token 估算），
# 子类只需实现 _call_llm 对接具体后端。
# ============================================================================
class LLMStrategy(SummarizationStrategy, ABC):
    """
    summarize (总控)
      ├─> 短文本 ──> _summarize_single (单次摘要)
      │                     ├─> _estimate_max_tokens (算 Token)
      │                     └─> _call_with_retry (重试通信)
      │                           └─> _call_llm (抽象，子类实现)
      │
      └─> 长文本 ──> _map_reduce_summarize (分块合并)
                            ├─> _split_into_chunks (拆文本)
                            ├─> _summarize_single (处理每一块) [多次]
                            └─> _summarize_single (最终合并)
                                  └─> (同上，走 Token 估算 -> 重试通信 -> LLM 调用)
      """

    # LLM 调用抽象方法
    @abstractmethod
    def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        ...

    #主入口，判断是否需要 Map-Reduce
    def summarize(
        self,
        text: str,
        sentences: list[str],
        max_sentences: int,
        language: str,
        config: dict,
        **kwargs,
    ) -> str:
        llm_cfg = config["llm"]
        target_chars = config["target_chars"].get(kwargs.get("max_length", "medium"), 150)
        style = kwargs.get("style", "paragraph")
        temperature = kwargs.get("temperature", 0.3)
        max_tokens_override = kwargs.get("max_tokens", None)

        mr_cfg = llm_cfg["map_reduce"]
        if mr_cfg["enabled"] and len(text) > mr_cfg["chunk_max_chars"]:
            logger.info("[LLMStrategy] 文本过长 (%d chars)，启用 Map-Reduce 分块", len(text))
            return self._map_reduce_summarize(
                text=text, language=language, style=style, target_chars=target_chars,
                temperature=temperature, max_tokens_override=max_tokens_override,
                config=config,
            )
        else:
            return self._summarize_single(
                text=text, language=language, style=style, target_chars=target_chars,
                temperature=temperature, max_tokens_override=max_tokens_override,
                config=config,
            )

    #单次调用 LLM 摘要（短文本）
    def _summarize_single(self, text, language, style, target_chars, temperature, max_tokens_override, config, **kwargs):
        prompts = config["llm"]["prompts"][language]
        llm_cfg = config["llm"]

        length_label = prompts["length_labels"].get(kwargs.get("max_length", "medium"), "中等")
        if isinstance(length_label, str) and "{chars}" in length_label:
            length_label = length_label.format(chars=target_chars)

        style_key = "style_bullet" if style == "bullet" else "style_paragraph"
        style_instruction = prompts.get(style_key, "")

        system_prompt = prompts["system"].format(style=style_instruction, length_desc=length_label, target_chars=target_chars)
        user_prompt = prompts["user_single"].format(text=text)
        max_tokens = max_tokens_override or self._estimate_max_tokens(target_chars, language)

        return self._call_with_retry(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens, timeout=llm_cfg["timeout_seconds"],
            retry_count=llm_cfg["retry_count"], retry_delay=llm_cfg["retry_delay_seconds"],
        )
    #Map-Reduce 长文本处理：分块→分别摘要→合并摘要
    def _map_reduce_summarize(self, text, language, style, target_chars, temperature, max_tokens_override, config):
        mr_cfg = config["llm"]["map_reduce"]
        prompts = config["llm"]["prompts"][language]
        llm_cfg = config["llm"]

        chunks = self._split_into_chunks(text, mr_cfg["chunk_max_chars"], mr_cfg["chunk_overlap_chars"], mr_cfg["max_chunks"])
        logger.info("[LLMStrategy] Map-Reduce: 分为 %d 块", len(chunks))

        chunk_target = max(30, target_chars // max(len(chunks), 1))
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            user_prompt = prompts["user_map"].format(chunk_target=chunk_target, text=chunk)
            try:
                result = self._call_with_retry(
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature, max_tokens=self._estimate_max_tokens(chunk_target, language),
                    timeout=llm_cfg["timeout_seconds"], retry_count=1, retry_delay=llm_cfg["retry_delay_seconds"],
                )
                chunk_summaries.append(result.strip())
            except Exception as e:
                logger.warning("[LLMStrategy] MAP chunk %d 失败: %s，使用原文截断", i + 1, e)
                chunk_summaries.append(chunk[:chunk_target] + "…")

        style_key = "style_bullet" if style == "bullet" else "style_paragraph"
        style_instruction = prompts.get(style_key, "")
        combined = "\n\n---\n\n".join(f"[Part {i+1}] {s}" for i, s in enumerate(chunk_summaries))
        user_prompt = prompts["user_reduce"].format(target_chars=target_chars, style_instruction=style_instruction, chunk_summaries=combined)

        return self._call_with_retry(
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens_override or self._estimate_max_tokens(target_chars, language),
            timeout=llm_cfg["timeout_seconds"], retry_count=llm_cfg["retry_count"], retry_delay=llm_cfg["retry_delay_seconds"],
        )

    #将长文本分块，尽量在句子边界断开
    @staticmethod
    def _split_into_chunks(text, chunk_size=3000, overlap=200, max_chunks=8):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len and len(chunks) < max_chunks:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                search_start = max(start, end - 200)
                search_region = text[search_start:end]
                best_break = -1
                for sep in ["\n\n", "。", ". ", ".\n", "\n", "！", "？", "! ", "? "]:
                    pos = search_region.rfind(sep)
                    if pos > best_break:
                        best_break = pos
                        if sep in ["\n\n", "。", ". "]:
                            break
                if best_break > 0:
                    end = search_start + best_break + 1
            chunks.append(text[start:end].strip())
            start = end - overlap if end < text_len else text_len
        return chunks

    #估算生成目标字符数所需的 max_tokens
    @staticmethod
    def _estimate_max_tokens(target_chars, language):
        ratio = 1.5 if language == "zh" else 0.6
        return max(50, int(target_chars * ratio) + 50)

    #带重试和指数退避的 LLM 调用
    def _call_with_retry(self, messages, temperature, max_tokens, timeout, retry_count, retry_delay):
        last_exception = None
        for attempt in range(retry_count + 1):
            try:
                return self._call_llm(messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            except Exception as e:
                last_exception = e
                if attempt < retry_count:
                    logger.warning("[LLMStrategy] 调用失败（第 %d/%d 次），重试中: %s", attempt + 1, retry_count + 1, e)
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error("[LLMStrategy] 所有重试均失败 (%d 次): %s", retry_count + 1, e)
        raise RuntimeError(f"LLM 调用失败，已重试 {retry_count} 次。最后错误: {last_exception}")




# ============================================================================
# LLM Strategy — OpenAI 兼容后端（含豆包）
# ============================================================================
class OpenAICompatibleLLMStrategy(LLMStrategy):
    """
    OpenAI 兼容 API 后端。支持豆包、DeepSeek、OpenAI 等端点。
    环境变量：DOUBAO_API_KEY / DOUBAO_BASE_URL（或 OPENAI_API_KEY / OPENAI_BASE_URL）
    """

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str | None = None, base_url: str | None = None):
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    #延迟初始化 OpenAI 客户端，读取 DOUBAO_* 环境变量
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                raise ImportError("需要安装依赖: pip install openai python-dotenv")
            import os
            self._client = OpenAI(
                api_key=self._api_key or os.getenv("DOUBAO_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=self._base_url or os.getenv("DOUBAO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            )
        return self._client

    #调用 API 生成文本，记录 token 用量
    def _call_llm(self, messages, temperature, max_tokens, timeout):
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("API 返回空内容")
        usage = response.usage
        if usage:
            logger.debug("[OpenAI] prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
                         usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        return content.strip()


# ============================================================================
# LLM Strategy — HuggingFace 本地模型后端
# ============================================================================
class HuggingFaceLLMStrategy(LLMStrategy):
    """基于 HuggingFace Transformers 的本地大模型摘要策略。"""

    PRESETS = {
        "bart-large-cnn": "facebook/bart-large-cnn",
        "pegasus-xsum": "google/pegasus-xsum",
        "flan-t5-base": "google/flan-t5-base",
        "flan-t5-large": "google/flan-t5-large",
        "bart-chinese": "fnlp/bart-base-chinese",
        "mengzi-t5-base": "Langboat/mengzi-t5-base",
    }

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str | None = None, use_fp16: bool = True):
        super().__init__()
        self._model_name = self.PRESETS.get(model_name, model_name)
        self._device = device
        self._use_fp16 = use_fp16
        self._pipeline = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError("需要安装: pip install torch transformers sentencepiece")
        if self._device is None:
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1
        else:
            device = self._device
        logger.info("[HuggingFace] 加载模型 %s | device=%s | fp16=%s", self._model_name, device, self._use_fp16)
        model_kwargs = {}
        if self._use_fp16 and device not in (-1, "mps"):
            model_kwargs["torch_dtype"] = torch.float16
        self._pipeline = pipeline("text2text-generation", model=self._model_name, device=device, model_kwargs=model_kwargs)
        return self._pipeline

    def _call_llm(self, messages, temperature, max_tokens, timeout):
        pipeline = self._load_pipeline()
        parts = [msg["content"] for msg in messages if msg["role"] in ("system", "user")]
        prompt = "\n\n".join(parts)
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0.01}
        if temperature > 0.01:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        import signal
        result_text: list[str] = []

        def _run():
            results = pipeline(prompt, **gen_kwargs)
            result_text.append(results[0]["generated_text"])

        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, lambda *args: None)
            signal.alarm(timeout)
            try:
                _run()
            except Exception:
                raise TimeoutError(f"HuggingFace 推理超时 ({timeout}s)")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            _run()

        return result_text[0].strip() if result_text else ""


# ============================================================================
# 作用：
# 根据配置创建对应的 LLM 策略实例，解耦策略创建和使用。
# 输入：配置字典、模型覆盖参数（可选）。输出：LLMStrategy 实例（OpenAICompatibleLLMStrategy 或 HuggingFaceLLMStrategy）。
# ============================================================================
def create_llm_strategy(config: dict, model_override: str | None = None) -> LLMStrategy:
    llm_cfg = config["llm"]
    backend = llm_cfg["backend"]
    model = model_override or llm_cfg["default_model"]
    if backend == "openai":
        logger.info("[LLM Factory] 创建 OpenAI 兼容策略，模型=%s", model)
        return OpenAICompatibleLLMStrategy(model=model)
    elif backend == "huggingface":
        logger.info("[LLM Factory] 创建 HuggingFace 策略，模型=%s", model)
        return HuggingFaceLLMStrategy(model_name=model)
    else:
        raise ValueError(f"不支持的 LLM backend: '{backend}'。支持: 'openai', 'huggingface'")


# ============================================================================
# MMR Deduplication — v3.0: 用 sklearn TF-IDF 余弦替代手写 Jaccard
#作用：
#用 Maximal Marginal Relevance（最大边际相关性）算法，在「相关性」和「多样性」之间权衡，去除语义重复的句子，避免摘要冗余。
# ============================================================================
def mmr_deduplicate(
    sentences: list[str],
    scored_indices: list[tuple[int, float]],
    max_sentences: int,
    lambda_: float = 0.7,
    threshold: float = 0.75,
) -> list[int]:
    """
    Maximal Marginal Relevance 去重。

    v3.0 重构：使用 sklearn TF-IDF + 余弦相似度替代手写 Jaccard 词集重叠。
    TF-IDF 给词分配不同权重（"the" 权重低，"人工智能" 权重高），
    相似度计算更精准。sklearn 不可用时回退到 Jaccard。
    """
    if len(scored_indices) <= 1:
        return sorted(i for i, _ in scored_indices)

    # 构建相似度矩阵
    if _SKLEARN_AVAILABLE:
        sim = _build_mmr_sim_sklearn(sentences)
    else:
        sim = _build_mmr_sim_jaccard(sentences)

    # 贪心 MMR 选择
    selected: list[int] = []
    remaining = list(scored_indices)
    first_idx, _ = remaining.pop(0)
    selected.append(first_idx)

    while remaining and len(selected) < max_sentences:
        best_idx = -1
        best_mmr = -float("inf")
        for idx, score in remaining:
            max_sim = max((sim[idx][s] for s in selected), default=0.0)
            if max_sim >= threshold:
                continue
            mmr_val = lambda_ * score - (1 - lambda_) * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = idx
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining = [(i, s) for i, s in remaining if i != best_idx]

    selected.sort()
    return selected


def _build_mmr_sim_sklearn(sentences: list[str]) -> np.ndarray:
    """sklearn TF-IDF + 余弦相似度"""
    vectorizer = TfidfVectorizer(max_features=5000)
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.eye(len(sentences))
    return cosine_similarity(tfidf)


def _build_mmr_sim_jaccard(sentences: list[str]) -> np.ndarray:
    """手写 Jaccard 相似度回退"""
    n = len(sentences)
    tokenized = []
    for s in sentences:
        tokens = set()
        if _JIEBA_AVAILABLE:
            tokens = set(jieba.cut(s))
        else:
            tokens = {t.lower() for t in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', s)}
        tokenized.append(tokens)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tokenized[i], tokenized[j]
            if a and b:
                sim[i][j] = sim[j][i] = len(a & b) / len(a | b)
    return sim


# ============================================================================
# Main Skill Class作用：
# 整个系统的核心入口，编排从输入到输出的完整流程，处理特殊文本、语言检测、算法选择、降级机制等
# ============================================================================
class TextSummarizerSkill(BaseSkill):
    """文本摘要技能（v3.0.0 — 通用 NLP 工具库化）"""

    name = "text_summarizer"
    description = (
        "将长文本压缩为精炼摘要。支持短/中/长三种长度、要点式和段落式两种输出风格、"
        "中英文自适应输出、多种摘要算法（启发式/TextRank/LLM大模型生成式）。"
    )
    triggers = ["摘要", "总结", "summarize", "summarization", "概括", "归纳", "太长不看", "tl;dr", "总结一下", "帮我压缩", "提取要点"]
    version = "3.0.0"
    author = "EnterpriseLearningAgent"
    changelog = (
        "v3.0.0: 用成熟库替代通用 NLP 工具 — nltk 分句、langdetect 语言检测、"
        "markdown+bs4 Markdown 净化、sklearn 相似度、jieba 分词"
    )
    input_schema = TextSummarizerInput
    output_schema = TextSummarizerOutput

    _config: dict
    _strategy_heuristic: HeuristicStrategy
    _strategy_textrank: TextRankStrategy
    _llm_strategy: LLMStrategy | None

    #初始化：加载配置、初始化提取式策略、延迟初始化 LLM 策略
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self._config = self._load_config(config_path)
        self._strategy_heuristic = HeuristicStrategy()
        self._strategy_textrank = TextRankStrategy()
        self._llm_strategy = None

    # ========================================================================
    # 主流程入口：预处理→语言检测→分句→算法选择→分支执行
    # ========================================================================
    """
    summarize (便捷入口)
    ↓
TextSummarizerSkill.execute (主流程控制器)
    ├─→ _preprocess_special_blocks (保护代码/表格)
    ├─→ _strip_markdown (可选: 清理Markdown)
    ├─→ _detect_language (语言检测)
    ├─→ _split_sentences (分句)
    ├─→ _resolve_algorithm (选择算法)
    ├─→ [分支1] _execute_llm (LLM生成式摘要)
    │       └─→ _get_llm_strategy (获取LLM策略实例)
    └─→ [分支2] 提取式摘要
            ├─→ _compute_target_counts (计算目标句数)
            ├─→ _strategy_textrank/_strategy_heuristic (执行算法)
            ├─→ mmr_deduplicate (MMR去重)
            ├─→ _assemble (组装摘要)
            ├─→ _trim_to_target (裁剪长度)
            ├─→ _restore_special_blocks (恢复代码/表格)
            └─→ _extract_key_points (提取关键点)
    └─→ _build_output (构建标准化输出)
    """
    def execute(self, input_data: TextSummarizerInput) -> TextSummarizerOutput:
        logger.info(
            "[text_summarizer] 开始摘要 | 原文长度=%d | max_length=%s | style=%s | algorithm=%s",
            len(input_data.text), input_data.max_length, input_data.style, input_data.algorithm,
        )

        # Step 0: Preprocess
        text, placeholders = self._preprocess_special_blocks(input_data.text)
        if input_data.strip_markdown:
            text = self._strip_markdown(text)

        original_length = len(input_data.text)

        # Step 1: Detect language — v3.0: langdetect 替代字符统计
        lang = self._detect_language(text) if input_data.language == "auto" else input_data.language
        logger.debug("[text_summarizer] 检测语言: %s", lang)

        # Step 2: Split sentences — v3.0: nltk.sent_tokenize 替代手写正则
        sentences = self._split_sentences(text, lang)
        logger.debug("[text_summarizer] 分句完成: %d 句", len(sentences))

        # Step 3: Determine algorithm
        algorithm = self._resolve_algorithm(input_data.algorithm, len(sentences))
        logger.debug("[text_summarizer] 使用算法: %s", algorithm)

        # ================================================================
        # LLM 分支（生成式摘要）
        # ================================================================
        if algorithm == "llm":
            return self._execute_llm(
                text=text, sentences=sentences, lang=lang, original_length=original_length,
                input_data=input_data, placeholders=placeholders,
            )

        # ================================================================
        # 提取式分支（启发式 / TextRank）
        # ================================================================
        target_counts = self._compute_target_counts(len(sentences), input_data.max_length)

        if algorithm == "textrank":
            indices = self._strategy_textrank.summarize(
                text=text, sentences=sentences, max_sentences=target_counts,
                language=lang, config=self._config["summarizer"],
            )
        else:
            indices = self._strategy_heuristic.summarize(
                text=text, sentences=sentences, max_sentences=target_counts,
                language=lang, config=self._config["summarizer"],
            )

        # MMR deduplication — v3.0: sklearn 余弦相似度
        if input_data.deduplicate and len(indices) > 1:
            mmr_cfg = self._config["summarizer"]["mmr"]
            if algorithm == "textrank":
                tr_scores = TextRankStrategy._pagerank(
                    TextRankStrategy._build_sim_sklearn(sentences, lang) if _SKLEARN_AVAILABLE
                    else TextRankStrategy._build_sim_jaccard(sentences, lang)
                )
                scored_for_mmr = [(i, tr_scores[i]) for i in indices]
            else:
                kw = self._config["summarizer"]["high_weight_keywords"].get(lang, self._config["summarizer"]["high_weight_keywords"]["en"])
                scfg = self._config["summarizer"]["sentence_scoring"]
                scored_for_mmr = [(i, HeuristicStrategy._score_sentence(sentences[i], i, len(sentences), kw, scfg)) for i in indices]
            scored_for_mmr.sort(key=lambda x: x[1], reverse=True)
            indices = mmr_deduplicate(
                sentences, scored_for_mmr, target_counts,
                lambda_=mmr_cfg["lambda"], threshold=mmr_cfg["similarity_threshold"],
            )
            logger.debug("[text_summarizer] MMR 去重后保留 %d 句", len(indices))

        # Assemble
        selected_sentences = [sentences[i] for i in indices]
        summary = self._assemble(selected_sentences, input_data.style, lang)
        summary = self._trim_to_target(summary, input_data.max_length, lang)
        summary = self._restore_special_blocks(summary, placeholders)
        key_points = self._extract_key_points(selected_sentences, lang)

        logger.info(
            "[text_summarizer] 摘要完成 | 选中 %d/%d 句 | 摘要长度=%d | 压缩比=%.2f%%",
            len(indices), len(sentences), len(summary), 100 * len(summary) / max(original_length, 1),
        )

        return self._build_output(
            summary=summary, original_length=original_length, input_data=input_data,
            lang=lang, num_selected=len(indices), algorithm=algorithm, key_points=key_points,
        )

    # ========================================================================
    # LLM 分支：调用 LLM 策略、记录耗时、失败时降级为启发式
    # ========================================================================
    def _execute_llm(self, text, sentences, lang, original_length, input_data, placeholders):
        llm_cfg = self._config["summarizer"]["llm"]
        fallback = llm_cfg.get("fallback_to_extractive", True)
        llm_latency_ms = None
        llm_model_used = None
        algorithm = "llm"

        try:
            strategy = self._get_llm_strategy(input_data.llm_model)
            llm_model_used = strategy.model_name
            t0 = time.perf_counter()
            summary = strategy.summarize(
                text=text, sentences=sentences, max_sentences=0, language=lang,
                config=self._config["summarizer"], max_length=input_data.max_length,
                style=input_data.style, temperature=input_data.llm_temperature,
                max_tokens=input_data.llm_max_tokens,
            )
            llm_latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("[text_summarizer] LLM 摘要完成 | 模型=%s | 耗时=%.0fms | 摘要长度=%d", llm_model_used, llm_latency_ms, len(summary))
        except Exception as e:
            logger.error("[text_summarizer] LLM 调用失败: %s", e)
            if fallback:
                logger.warning("[text_summarizer] 降级为启发式摘要")
                algorithm = "heuristic"
                target_counts = self._compute_target_counts(len(sentences), input_data.max_length)
                indices = self._strategy_heuristic.summarize(
                    text=text, sentences=sentences, max_sentences=target_counts,
                    language=lang, config=self._config["summarizer"],
                )
                selected_sentences = [sentences[i] for i in indices]
                summary = self._assemble(selected_sentences, input_data.style, lang)
                summary = self._trim_to_target(summary, input_data.max_length, lang)
            else:
                raise

        summary = self._restore_special_blocks(summary, placeholders)

        if input_data.style == "bullet":
            key_points = [line.lstrip("•-*· ").strip() for line in summary.split("\n") if line.strip()][:5]
        else:
            summary_sentences = self._split_sentences(summary, lang)
            key_points = self._extract_key_points(summary_sentences[:5] if summary_sentences else [summary], lang)

        return self._build_output(
            summary=summary, original_length=original_length, input_data=input_data,
            lang=lang, num_selected=0, algorithm=algorithm, key_points=key_points,
            llm_model_used=llm_model_used, llm_latency_ms=llm_latency_ms,
        )

    #获取或创建 LLM 策略实例（单例，避免重复加载）
    def _get_llm_strategy(self, model_override=None):
        if self._llm_strategy is None or model_override is not None:
            self._llm_strategy = create_llm_strategy(self._config["summarizer"], model_override=model_override)
        return self._llm_strategy

    # ========================================================================
    # （v3.0.0 重构）语言检测（优先 langdetect，回退 CJK 字符统计）
    # ========================================================================
    def _detect_language(self, text: str) -> str:
        """检测文本语言。优先使用 langdetect，不可用时回退到 CJK 字符统计。"""
        if _LANGDETECT_AVAILABLE:
            try:
                detected = lang_detect(text)
                return "zh" if detected.startswith("zh") else "en"
            except LangDetectException:
                logger.debug("[text_summarizer] langdetect 检测失败，回退到字符统计")
        # 回退
        cjk = len(_RE_CJK.findall(text))
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        total = cjk + ascii_alpha + 1
        return "zh" if cjk / total > 0.25 else "en"

    # ========================================================================
    # （v3.0.0 重构）语言自适应分句（英文优先 nltk，中文正则）
    # ========================================================================
    def _split_sentences(self, text: str, lang: str) -> list[str]:
        """
        分句。英文优先使用 nltk.sent_tokenize（覆盖数百种缩写），
        中文使用正则（nltk 不支持中文分句）。
        """
        min_len = self._config["summarizer"]["splitter"]["min_sentence_length"]

        if lang == "en" and _NLTK_AVAILABLE:
            raw = sent_tokenize(text)
        elif lang == "en":
            # 回退：通用标点分割
            raw = _RE_SPLIT_UNIVERSAL.split(text)
        else:
            text = text.replace("\n", "。")
            raw = _RE_SPLIT_ZH.split(text)

        sentences = [s.strip() for s in raw if len(s.strip()) > min_len]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences

    # ========================================================================
    # （v3.0.0 重构）移除 Markdown 标记（优先 markdown+bs4，回退正则）
    # ========================================================================
    def _strip_markdown(self, text: str) -> str:
        """
        将 Markdown 转为纯文本。
        优先使用 markdown 解析器 + bs4（完整 GFM 支持），
        不可用时回退到简单正则清理。
        """
        if _MARKDOWN_BS4_AVAILABLE:
            try:
                html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
                return BeautifulSoup(html, 'html.parser').get_text().strip()
            except Exception:
                logger.debug("[text_summarizer] markdown+bs4 解析失败，回退到正则")
        # 回退
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    # ========================================================================
    # 加载外部 YAML 配置或默认配置
    # ========================================================================
    def _load_config(self, config_path):
        if config_path and Path(config_path).exists():
            logger.info("[text_summarizer] 从 %s 加载配置", config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        logger.info("[text_summarizer] 使用默认配置")
        return DEFAULT_CONFIG

    #计算提取式算法的目标句数
    def _compute_target_counts(self, total_sentences, max_length):
        tc = self._config["summarizer"]["target_counts"][max_length]
        if tc["method"] == "ratio":
            return max(tc["min"], int(total_sentences * tc["value"]))
        return tc.get("fixed", 3)

    #解析算法选择（auto 模式自动判断）
    def _resolve_algorithm(self, algo, num_sentences):
        if algo == "auto":
            return "textrank" if num_sentences >= 10 else "heuristic"
        return algo

    # ========================================================================
    # 语言自适应组装摘要（要点式 / 段落式）
    # ========================================================================
    def _assemble(self, sentences, style, lang):
        if style == "bullet":
            return "\n".join(f"• {s.strip()}" for s in sentences)
        return " ".join(s.strip() for s in sentences) if lang == "en" else "".join(s.strip() for s in sentences)

    # ========================================================================
    # 裁剪摘要到目标字符数（尽量在句子边界）
    # ========================================================================
    def _trim_to_target(self, summary, max_length, lang):
        target = self._config["summarizer"]["target_chars"][max_length]
        if len(summary) <= target:
            return summary
        sep = ". " if lang == "en" else "。"
        truncated = summary[:target]
        last_sep = truncated.rfind(sep)
        if last_sep > target * 0.5:
            return truncated[:last_sep + len(sep.rstrip())].rstrip()
        return summary[:target].rstrip() + ("..." if lang == "en" else "…")

    # ========================================================================
    # （v3.0.0 重构）移除 Markdown 标记（优先 markdown+bs4，回退正则）
    # ========================================================================
    def _preprocess_special_blocks(self, text):
        placeholders = {}
        counter = [0]

        def _protect(pattern, tag, m):
            counter[0] += 1
            key = f"⟨{tag}_{counter[0]}⟩"
            placeholders[key] = m.group(0)
            return key

        text = _RE_MD_CODE_BLOCK.sub(lambda m: _protect(_RE_MD_CODE_BLOCK, "CODE_BLOCK", m), text)
        text = _RE_MD_INLINE_CODE.sub(lambda m: _protect(_RE_MD_INLINE_CODE, "INLINE_CODE", m), text)
        text = _RE_TABLE.sub(lambda m: _protect(_RE_TABLE, "TABLE", m), text)
        return text, placeholders

    def _restore_special_blocks(self, summary, placeholders):
        handling = self._config["summarizer"]["special_block_handling"]
        for key, content in placeholders.items():
            if "CODE_BLOCK" in key:
                summary = summary.replace(key, "[代码块已省略]" if handling["code_block"] == "omit" else content)
            elif "INLINE_CODE" in key:
                summary = summary.replace(key, content if handling["inline_code"] == "keep" else "")
            elif "TABLE" in key:
                if handling["table"] == "compress":
                    lines = content.strip().split("\n")
                    if len(lines) >= 2:
                        cols = lines[0].count("|") - 1
                        summary = summary.replace(key, f"[表格: {len(lines)-1}行 × {cols}列]")
                    else:
                        summary = summary.replace(key, "[表格]")
                else:
                    summary = summary.replace(key, content)
        return summary

    # ========================================================================
    # 从句子中提取关键要点
    # ========================================================================
    def _extract_key_points(self, sentences, lang):
        max_chars = 80 if lang == "zh" else 120
        ellipsis = "…" if lang == "zh" else "..."
        points = []
        for s in sentences[:5]:
            short = s[:max_chars].strip()
            if len(s) > max_chars:
                short += ellipsis
            points.append(short)
        return points

    # ========================================================================
    # 构建标准化输出对象
    # ========================================================================
    def _build_output(self, summary, original_length, input_data, lang, num_selected, algorithm, key_points=None, llm_model_used=None, llm_latency_ms=None):
        if key_points is None:
            key_points = []
        return TextSummarizerOutput(
            summary=summary,
            original_length=original_length,
            summary_length=len(summary),
            compression_ratio=round(len(summary) / max(original_length, 1), 4),
            key_points=key_points,
            style=input_data.style,
            max_length_used=input_data.max_length,
            detected_language=lang,
            algorithm_used=algorithm,
            num_sentences_selected=num_selected,
            llm_model_used=llm_model_used,
            llm_latency_ms=round(llm_latency_ms, 1) if llm_latency_ms else None,
        )


# ============================================================================
# Convenience function
# ============================================================================
def summarize(
    text: str,
    max_length: Literal["short", "medium", "long"] = "medium",
    style: Literal["bullet", "paragraph"] = "paragraph",
    language: Literal["zh", "en", "auto"] = "auto",
    algorithm: Literal["heuristic", "textrank", "llm", "auto"] = "auto",
    deduplicate: bool = True,
    strip_markdown: bool = False,
    llm_model: str | None = None,
    llm_temperature: float = 0.3,
) -> TextSummarizerOutput:
    """便捷函数：一行调用摘要"""
    skill = TextSummarizerSkill()
    return skill.execute(TextSummarizerInput(
        text=text, max_length=max_length, style=style, language=language,
        algorithm=algorithm, deduplicate=deduplicate, strip_markdown=strip_markdown,
        llm_model=llm_model, llm_temperature=llm_temperature,
    ))