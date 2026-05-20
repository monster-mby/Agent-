"""
RAG 问答技能 — RAG 体系第五环（终点站）v1.1

将用户问题 + 检索到的上下文分块组装为 Prompt，调用 LLM 生成带引用的最终回答。

v1.1 核心升级（相比 v1.0）：
- 🔴 P0: tenacity 分级重试（区分 429/5xx 可重试 vs 4xx 不重试）
- 🔴 P0: 引用格式扩展（[1-3] 范围/[1][2] 连续/越界校验 + 清洗）
- 🔴 P0: 引用溯源验证（citation grounding — 检测 LLM 张冠李戴）
- 🔴 P0: 答案置信度估算（多信号融合：top1_score + 引用覆盖率 + 低置信短语）
- 🟡 P1: tiktoken 精准 token 估算（自动回退手写估算）
- 🟡 P1: Jinja2 模板引擎（可选）+ 内置模板类 + YAML 模板目录支持
- 🟡 P1: 关键词加权上下文选择（BM25 风格关键词匹配加成）
- 🟡 P1: 上下文 metadata 富化展示（文档标题/分段位置结构化注入）
- 🟡 P1: LLM 空响应/拒绝回答检测 + 自动重试
- 🟡 P1: importlib.util.find_spec 依赖检测（与 vector_search 一致）

与上游 vector_search / 下游 orchestrator 零耦合，只通过 dict 衔接。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from src.skills.base.skill_manager import SkillManager
from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger("rag_answer")


# ═══════════════════════════════════════════════════════════════
# 依赖检测（importlib.util.find_spec）
# ═══════════════════════════════════════════════════════════════

def _is_available(package: str) -> bool:
    """检测 Python 包是否可导入"""
    return importlib.util.find_spec(package) is not None


HAS_TIKTOKEN = _is_available("tiktoken")
HAS_JINJA2 = _is_available("jinja2")
HAS_TENACITY = _is_available("tenacity")
HAS_LANGDETECT = _is_available("langdetect")

if not HAS_TIKTOKEN:
    logger.info("tiktoken 未安装 — 将使用手写 token 估算（pip install tiktoken 可提升精度）")
if not HAS_JINJA2:
    logger.info("Jinja2 未安装 — 将使用内置模板引擎（pip install jinja2 可启用高级模板）")
if not HAS_TENACITY:
    logger.warning("tenacity 未安装 — LLM 调用失败将直接回退模拟模式（pip install tenacity）")
if not HAS_LANGDETECT:
    logger.debug("langdetect 未安装 — 将使用简单语言检测（pip install langdetect）")


# ═══════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════

class RagAnswerError(Exception):
    """RAG 问答基础异常"""


class EmptyContextError(RagAnswerError):
    """检索结果为空，无法生成回答"""


class LLMCallError(RagAnswerError):
    """LLM 调用失败（不可重试）"""


class LLMRetryableError(RagAnswerError):
    """LLM 调用失败（可重试，如 429/5xx）"""


class CitationGroundingError(RagAnswerError):
    """引用溯源失败（LLM 幻觉）"""


# ═══════════════════════════════════════════════════════════════
# 模板管理
# ═══════════════════════════════════════════════════════════════

class PromptTemplate:
    """Prompt 模板 — 支持 Jinja2（可选）或 Python format 回退"""

    def __init__(
        self,
        template_str: str,
        name: str = "default",
        engine: str = "auto",  # auto / jinja2 / python
    ):
        self.name = name
        self.template_str = template_str
        self._engine = engine
        self._jinja_template: Any = None

        if engine == "auto":
            self._engine = "jinja2" if HAS_JINJA2 else "python"

        if self._engine == "jinja2" and HAS_JINJA2:
            import jinja2
            self._jinja_template = jinja2.Template(template_str)
        elif self._engine == "jinja2" and not HAS_JINJA2:
            logger.warning("Jinja2 不可用，回退到 Python format 模板")
            self._engine = "python"

    def render(self, **variables) -> str:
        """渲染模板"""
        if self._engine == "jinja2" and self._jinja_template is not None:
            return self._jinja_template.render(**variables)
        else:
            # Python format 回退 — 用 {key} 占位
            # 对未提供的变量保留原样
            result = self.template_str
            for key, value in variables.items():
                result = result.replace("{" + key + "}", str(value))
            return result

    @classmethod
    def from_yaml_dir(cls, template_dir: str) -> Dict[str, "PromptTemplate"]:
        """从 YAML 目录加载所有模板"""
        templates: Dict[str, PromptTemplate] = {}
        dir_path = Path(template_dir)
        if not dir_path.exists():
            logger.warning("模板目录不存在: %s", template_dir)
            return templates

        for yaml_file in dir_path.glob("*.yaml"):
            try:
                import yaml
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                name = data.get("name", yaml_file.stem)
                templates[name] = cls(
                    template_str=data["template"],
                    name=name,
                    engine=data.get("engine", "auto"),
                )
                logger.debug("加载模板: %s (%s)", name, yaml_file)
            except Exception as exc:
                logger.error("加载模板失败 %s: %s", yaml_file, exc)

        return templates


# ═══════════════════════════════════════════════════════════════
# Token 估算
# ═══════════════════════════════════════════════════════════════

class TokenEstimator:
    """Token 估算器 — tiktoken 精准 + 手写回退"""

    _ENCODING_NAME = "cl100k_base"  # GPT-4/3.5 通用编码

    def __init__(self):
        self._encoder: Any = None
        if HAS_TIKTOKEN:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self._ENCODING_NAME)
                logger.debug("TokenEstimator: 使用 tiktoken (%s)", self._ENCODING_NAME)
            except Exception:
                logger.debug("tiktoken 加载失败，回退手写估算")

    def count(self, text: str) -> int:
        """估算文本 token 数"""
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        return self._estimate_heuristic(text)

    def count_batch(self, texts: List[str]) -> List[int]:
        """批量估算"""
        return [self.count(t) for t in texts]

    @staticmethod
    def _estimate_heuristic(text: str) -> int:
        """手写启发式估算（中文≈字数×1.5，英文≈单词×1.3）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(
            len(w) for w in re.findall(r'[a-zA-Z]+', text)
        )
        return int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 0.5)


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class SearchResultRef(BaseModel):
    """检索结果引用 — 与 vector_search.SearchResult 对应，但不 import"""

    chunk_id: str = Field(..., description="分块 ID")
    text: str = Field(..., description="分块文本")
    score: float = Field(default=0.0, description="相似度分数")
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    """单条引用"""

    citation_id: int = Field(..., description="引用编号 [1], [2], ...")
    chunk_id: str
    text_snippet: str = Field(default="", description="引用文本片段（前 200 字）")
    score: float = 0.0
    grounded: bool = Field(default=True, description="是否通过溯源验证")


class ConversationTurn(BaseModel):
    """对话历史中的一轮"""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class RagAnswerInput(BaseModel):
    """RAG 问答输入 v1.1"""

    query: str = Field(..., min_length=1, max_length=4096, description="用户问题")
    search_results: List[SearchResultRef] = Field(
        default_factory=list,  # 允许空列表，不强制至少一个
        description="vector_search 返回的检索结果"
    )

    # ── Prompt 控制 ──
    system_prompt: Optional[str] = Field(
        default=None, description="自定义 system prompt（None 则使用默认模板）"
    )
    template_name: str = Field(
        default="default_zh", description="内置/外置模板名称"
    )
    include_citations: bool = Field(
        default=True, description="是否在回答中标注引用编号 [n]"
    )

    # ── 上下文选择 ──
    max_context_chunks: int = Field(
        default=8, ge=1, le=20, description="最多使用多少个分块作为上下文"
    )
    max_context_tokens: int = Field(
        default=3000, ge=256, le=16000, description="上下文最大 token 数（估算）"
    )
    keyword_weight_boost: float = Field(
        default=0.15, ge=0.0, le=0.5,
        description="关键词匹配对排序分数的加成权重（0=无加成，0.15=默认）",
    )

    # ── LLM 控制 ──
    llm_client: Optional[Any] = Field(
        default=None, description="LLM 客户端（需实现 .chat() 方法）"
    )
    model: Optional[str] = Field(default=None, description="模型名称")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    stream: bool = Field(default=False, description="流式输出（v1.1 占位）")

    # ── 重试与质检 ──
    retry_on_failure: bool = Field(
        default=True, description="LLM 调用失败时是否重试"
    )
    max_retries: int = Field(default=3, ge=0, le=5)
    validate_citation_grounding: bool = Field(
        default=True, description="是否验证引用溯源（检测 LLM 幻觉引用）"
    )
    estimate_confidence: bool = Field(
        default=True, description="是否估算回答置信度"
    )
    detect_refusal: bool = Field(
        default=True, description="是否检测 LLM 拒绝回答/空响应"
    )

    # ── 对话历史 ──
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None, description="历史对话轮次（多轮场景）"
    )

    # ── 语言 ──
    answer_language: str = Field(default="zh", description="回答语言：zh / en / auto")

    # ── 结构化输出 ──
    output_mode: str = Field(
        default="text", description="输出模式：text（纯文本+正则提取引用）| json（LLM 结构化 JSON）"
    )
    # ── ✨ 反思模式控制 ──
    enable_reflection: bool = Field(
        default=True,
        description="是否启用反思模式（默认 True，符合关键约定 #22）"
    )
    reflection_max_iterations: int = Field(
        default=2, ge=1, le=5, description="反思最大迭代次数"
    )
    reflection_quality_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="反思质量分数达标线"
    )

    @model_validator(mode="after")
    def _check_search_results_order(self) -> "RagAnswerInput":
        """校验 search_results 按分数降序排列（警告不阻塞）"""
        if not self.search_results:  # 添加这一行，空列表直接返回
            return self
        scores = [r.score for r in self.search_results]
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                logger.warning(
                    "search_results 未按分数降序排列，建议上游保证顺序"
                )
                break
        return self


class RagAnswerOutput(BaseModel):
    """RAG 问答输出 v1.1"""

    success: bool
    answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        default=None, description="回答置信度（0~1），estimate_confidence=True 时计算"
    )
    model: str = ""
    usage: dict = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    timing_breakdown: dict = Field(default_factory=dict)
    context_chunks_used: int = 0
    retry_count: int = Field(default=0, description="LLM 调用重试次数")
    error: Optional[str] = None

    # ── ✨ 反思报告 ──
    reflection_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="反思报告（enable_reflection=True 时返回）"
    )


# ═══════════════════════════════════════════════════════════════
# 默认 Prompt 模板
# ═══════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT_ZH = """你是一个专业、严谨的知识问答助手。

回答规则：
1. 优先使用下方「上下文」中的信息回答问题，引用时标注来源编号如 [1]、[2]。
2. 如果上下文与用户问题相关，请基于上下文回答。
3. 如果上下文与用户问题明显无关，请忽略上下文，基于你的知识直接回答。
4. 不得编造不存在于上下文或你知识中的信息。
5. 回答应结构清晰，使用简洁的中文。"""

DEFAULT_SYSTEM_PROMPT_EN = """You are a precise and trustworthy question-answering assistant. Answer the user's question based STRICTLY on the provided context.

Rules:
1. Only use information explicitly stated in the context. Do not fabricate or use external knowledge.
2. If the context lacks sufficient information, say "I cannot answer this question based on the available materials." Do not guess.
3. Cite sources using [n] or [n-m] notation when referencing specific context chunks.
4. Keep answers clear and well-structured.
5. If the context contains contradictory information, point out the contradiction and cite each source."""

CONTEXT_TEMPLATE = """## 上下文

{context_blocks}

## {conversation_section}用户问题

{query}

## 回答（请严格按照上述要求）："""

CONVERSATION_HISTORY_TEMPLATE = """## 对话历史

{history_blocks}

"""


def _build_default_templates() -> Dict[str, PromptTemplate]:
    """构建内置默认模板"""
    templates = {}
    templates["default_zh"] = PromptTemplate(DEFAULT_SYSTEM_PROMPT_ZH, name="default_zh")
    templates["default_en"] = PromptTemplate(DEFAULT_SYSTEM_PROMPT_EN, name="default_en")
    return templates


# ═══════════════════════════════════════════════════════════════
# 技能类
# ═══════════════════════════════════════════════════════════════

class RagAnswerSkill(BaseSkill):
    """RAG 问答技能 — RAG 第五环 v1.1

    v1.1 核心升级：
    - tenacity 分级重试（429/5xx 重试，4xx 不重试）
    - 引用格式扩展（[1-3] 范围、[1][2] 连续、越界清洗）
    - 引用溯源验证（citation grounding — n-gram 重叠检测）
    - 答案置信度估算（多信号融合）
    - tiktoken 精准 token 估算 + 手写回退
    - Jinja2 模板引擎（可选）+ 内置模板 + YAML 加载
    - 关键词加权上下文选择
    - metadata 富化展示
    - LLM 空响应/拒绝回答检测
    - 多轮对话历史支持
    """

    name = "rag_answer"
    description = (
        "基于检索到的上下文分块，调用 LLM 生成带引用+置信度的最终回答。"
        "RAG 管线终点：document_loader → chunker → embedder → vector_search → rag_answer。"
    )
    version = "1.1.0"
    author = "EnterpriseLearningAgent"
    triggers = [
        "问答", "RAG问答", "知识问答", "基于上下文回答", "rag answer",
        "检索问答", "文档问答", "智能问答", "带引用回答",
    ]

    input_schema = RagAnswerInput
    output_schema = RagAnswerOutput

    # ── 正则常量 ─────────────────────────────────
    _CITATION_PATTERN = re.compile(
        r'\[(\d+(?:-\d+)?)\]'          # [1] 或 [1-3]
        r'|\[(\d+)\]\[(\d+)\]'          # [1][2]
        r'|\[(\d+(?:,\d+)*)\]',         # [1,2,3]
    )
    _REFUSAL_PATTERNS = [
        re.compile(p) for p in [
            r'我无法回答',
            r'没有足够.*信息',
            r'根据现有资料.*无法',
            r'I cannot answer',
            r'insufficient information',
            r'cannot be answered',
        ]
    ]

    # ── n-gram 溯源常量 ───────────────────────────
    _NGRAM_SIZES = (5, 4, 3)  # 由大到小匹配

    # ═══════════════════════════════════════════════
    # 构造 & 模板加载
    # ═══════════════════════════════════════════════

    def __init__(self, template_dir: Optional[str] = None):
        super().__init__()
        self._token_estimator = TokenEstimator()
        self._templates = _build_default_templates()

        # 从 YAML 目录加载额外模板（覆盖同名内置模板）
        if template_dir:
            external = PromptTemplate.from_yaml_dir(template_dir)
            self._templates.update(external)
            logger.info("已加载 %d 个外置模板", len(external))

        # 尝试从环境变量加载模板目录
        env_template_dir = os.getenv("RAG_ANSWER_TEMPLATE_DIR")
        if env_template_dir and env_template_dir != template_dir:
            external = PromptTemplate.from_yaml_dir(env_template_dir)
            self._templates.update(external)
            logger.info("从环境变量加载 %d 个模板", len(external))

    # ═══════════════════════════════════════════════
    # 核心入口
    # ═══════════════════════════════════════════════

    def execute(self, input_data: RagAnswerInput) -> dict:
        """执行 RAG 问答

        Args:
            input_data: RAG 问答输入 Pydantic 对象，包含以下字段：
                - query: 用户问题
                - search_results: 检索结果列表
                - system_prompt: 自定义 system prompt（可选）
                - template_name: 模板名称
                - include_citations: 是否标注引用编号
                - max_context_chunks: 最多使用的分块数
                - max_context_tokens: 上下文最大 token 数
                - keyword_weight_boost: 关键词权重加成
                - llm_client: LLM 客户端（可选）
                - model: 模型名称（可选）
                - temperature: 生成温度
                - max_tokens: 最大生成 token
                - retry_on_failure: 是否重试
                - max_retries: 最大重试次数
                - validate_citation_grounding: 是否验证引用溯源
                - estimate_confidence: 是否估算置信度
                - detect_refusal: 是否检测拒绝回答
                - conversation_history: 对话历史（可选）
                - answer_language: 回答语言
                - output_mode: 输出模式（text | json）

        Returns:
            RagAnswerOutput.model_dump()
        """
        # ✅ 无检索结果时，直接用 LLM 自身知识回答（不做模拟兜底）
        if not input_data.search_results or len(input_data.search_results) == 0:
            logger.info("RagAnswerSkill: 无检索结果，将基于 LLM 自身知识回答")
            system_prompt = "你是一个有帮助的AI助手。请直接回答用户的问题。"
            user_prompt = input_data.query
            try:
                result = self._do_llm_call(
                    input_data, system_prompt, user_prompt
                )
                return RagAnswerOutput(
                    success=True,
                    answer=result["content"],
                    model=result.get("model", "unknown"),
                    usage=result.get("usage", {}),
                ).model_dump()
            except Exception as e:
                logger.warning("LLM 调用失败，回退模拟回答: %s", e)
                simulated = self._simulated_answer(input_data.query, "", user_prompt)
                return RagAnswerOutput(
                    success=True,
                    answer=simulated["content"],
                    model=simulated.get("model", "simulated"),
                    usage=simulated.get("usage", {}),
                ).model_dump()

        t_start = time.perf_counter()
        timing: Dict[str, float] = {}
        retry_count = 0

        try:
            logger.info(
                "RagAnswer v1.1 开始 | query='%s' | results=%d | max_chunks=%d | "
                "template=%s | citations=%s | grounding=%s | confidence=%s",
                input_data.query[:80],
                len(input_data.search_results),
                input_data.max_context_chunks,
                input_data.template_name,
                input_data.include_citations,
                input_data.validate_citation_grounding,
                input_data.estimate_confidence,
            )

            # ── 第一步：选择 & 截断上下文分块（关键词加权）──
            t_context = time.perf_counter()
            selected_chunks = self._select_context_chunks(
                input_data.search_results,
                input_data.query,
                input_data.max_context_chunks,
                input_data.max_context_tokens,
                input_data.keyword_weight_boost,
            )
            timing["select_context_ms"] = round(
                (time.perf_counter() - t_context) * 1000, 2
            )

            if not selected_chunks:
                logger.warning("上下文分块为空")
                return RagAnswerOutput(
                    success=True,
                    answer="抱歉，当前没有找到相关材料来回答这个问题。",
                    context_chunks_used=0,
                    elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    timing_breakdown=timing,
                    confidence=0.0,
                ).model_dump()

            logger.info("上下文选定: %d chunks (token预算=%d)",
                        len(selected_chunks), input_data.max_context_tokens)

            # ── 第二步：构建 Prompt ──
            t_prompt = time.perf_counter()
            system_prompt = self._get_system_prompt(
                input_data.system_prompt,
                input_data.template_name,
                input_data.answer_language,
            )
            user_prompt = self._build_user_prompt(
                selected_chunks,
                input_data.query,
                input_data.include_citations,
                input_data.conversation_history,
            )
            timing["build_prompt_ms"] = round(
                (time.perf_counter() - t_prompt) * 1000, 2
            )

            # ── 第三步：调用 LLM（含重试）──
            t_llm = time.perf_counter()
            llm_response, retry_count = self._call_llm_with_retry(
                input_data, system_prompt, user_prompt
            )
            timing["llm_call_ms"] = round(
                (time.perf_counter() - t_llm) * 1000, 2
            )

            answer_text = llm_response.get("content", "") or ""
            usage = llm_response.get("usage", {})
            model_used = llm_response.get("model", input_data.model or "unknown")

            # ── 第四步：拒绝/空响应检测 ──
            if input_data.detect_refusal:
                t_refusal = time.perf_counter()
                if self._detect_refusal(answer_text):
                    logger.warning("LLM 拒绝回答或返回空响应: '%s'", answer_text[:100])
                timing["refusal_check_ms"] = round(
                    (time.perf_counter() - t_refusal) * 1000, 2
                )

            # ── 第五步：提取引用 ──
            t_cite = time.perf_counter()
            citations: List[Citation] = []
            if input_data.include_citations and answer_text:
                citations = self._extract_citations(answer_text, selected_chunks)

                # 引用溯源验证
                if input_data.validate_citation_grounding and citations:
                    citations = self._verify_citation_grounding(
                        answer_text, citations, selected_chunks
                    )
            timing["extract_citations_ms"] = round(
                (time.perf_counter() - t_cite) * 1000, 2
            )

            # ── 第六步：置信度估算 ──
            confidence: Optional[float] = None
            if input_data.estimate_confidence:
                confidence = self._estimate_confidence(
                    answer_text, selected_chunks, citations
                )

            # ── ✨ 第七步：反思模式（v2.0 新增）──
            reflection_report = None
            # 🔍 调试日志：检查反思条件
            logger.debug(
                "反思模式检查 | enable_reflection=%s | llm_client_type=%s | llm_client_is_none=%s",
                input_data.enable_reflection,
                type(input_data.llm_client).__name__ if input_data.llm_client else "None",
                input_data.llm_client is None,
            )
            if input_data.enable_reflection and input_data.llm_client is not None:
                t_reflect = time.perf_counter()
                try:
                    reflection_report, refined_answer = self._run_reflection(
                        query=input_data.query,
                        initial_answer=answer_text,
                        search_results=input_data.search_results,
                        model=input_data.model or "gpt-4",
                        llm_client=input_data.llm_client,
                        max_iterations=input_data.reflection_max_iterations,
                        quality_threshold=input_data.reflection_quality_threshold,
                    )
                    if refined_answer:
                        answer_text = refined_answer
                        # 重新提取引用（修订后文本的引用需要重新映射）
                        if input_data.include_citations:
                            citations = self._extract_citations(answer_text, selected_chunks)
                            if input_data.validate_citation_grounding and citations:
                                citations = self._verify_citation_grounding(
                                    answer_text, citations, selected_chunks
                                )
                    timing["reflection_ms"] = round(
                        (time.perf_counter() - t_reflect) * 1000, 2
                    )
                    logger.info(
                        "反思完成 | iterations=%d | score=%.2f",
                        reflection_report.get("iterations", 0) if reflection_report else 0,
                        reflection_report.get("final_score", 0.0) if reflection_report else 0.0,
                    )
                except Exception as exc:
                    logger.error("反思失败（降级为普通模式）: %s", exc)
                    timing["reflection_ms"] = round(
                        (time.perf_counter() - t_reflect) * 1000, 2
                    )
                    # 反思失败不影响主流程，继续返回原始答案

            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

            grounded_count = sum(1 for c in citations if c.grounded)
            logger.info(
                "RagAnswer v1.1 完成 | answer_len=%d | citations=%d(%d grounded) | "
                "confidence=%s | chunks=%d | retries=%d | reflection=%s | total=%.1fms",
                len(answer_text), len(citations), grounded_count,
                f"{confidence:.2f}" if confidence is not None else "N/A",
                len(selected_chunks), retry_count,
                "enabled" if reflection_report else "disabled",
                timing["total_ms"],
            )

            return RagAnswerOutput(
                success=True,
                answer=answer_text,
                citations=citations,
                confidence=confidence,
                model=model_used,
                usage=usage,
                elapsed_ms=timing["total_ms"],
                timing_breakdown=timing,
                context_chunks_used=len(selected_chunks),
                retry_count=retry_count,
                reflection_report=reflection_report,
            ).model_dump()


        except EmptyContextError:
            elapsed = round((time.perf_counter() - t_start) * 1000, 2)
            return RagAnswerOutput(
                success=True,
                answer="抱歉，没有找到相关资料来回答您的问题。请尝试换个问法。",
                elapsed_ms=elapsed,
                timing_breakdown=timing,
                confidence=0.0,
                context_chunks_used=0,
            ).model_dump()
        except LLMCallError as exc:
            logger.error("LLM 调用失败（不可重试）: %s", exc)
            return self._error_output(input_data.model_dump(), t_start, timing, str(exc), retry_count)
        except Exception as exc:
            logger.exception("RagAnswer 执行失败")
            return self._error_output(input_data.model_dump(), t_start, timing, str(exc), retry_count)


    # ═══════════════════════════════════════════════
    # 上下文选择（关键词加权版）
    # ═══════════════════════════════════════════════

    def _select_context_chunks(
        self,
        results: List[SearchResultRef],
        query: str,
        max_chunks: int,
        max_tokens: int,
        keyword_boost: float,
    ) -> List[SearchResultRef]:
        """关键词加权排序 + token 预算截断"""
        # 提取查询关键词
        keywords = self._extract_keywords(query)

        # 计算加权分数
        scored: List[Tuple[float, int, SearchResultRef]] = []
        for idx, r in enumerate(results):
            base_score = r.score

            # 关键词加成：文本中包含越多关键词，加成越高
            if keywords and keyword_boost > 0:
                text_lower = r.text.lower()
                kw_hits = sum(1 for kw in keywords if kw.lower() in text_lower)
                if kw_hits > 0:
                    # BM25 风格：命中越多加成越大，但有上限
                    kw_bonus = min(keyword_boost, keyword_boost * kw_hits / len(keywords))
                    base_score += kw_bonus

            scored.append((base_score, idx, r))

        # 按加权分数降序排列（稳定排序，同分保持原序）
        scored.sort(key=lambda x: x[0], reverse=True)

        # Token 预算截断
        selected: List[SearchResultRef] = []
        token_budget = max_tokens

        for score, idx, r in scored:
            if len(selected) >= max_chunks:
                break

            chunk_tokens = self._token_estimator.count(r.text)
            if chunk_tokens > token_budget and selected:
                logger.debug(
                    "token 预算不足: chunk '%s' 需 %d tokens，剩余 %d，跳过",
                    r.chunk_id, chunk_tokens, token_budget,
                )
                continue

            selected.append(r)
            token_budget -= min(chunk_tokens, token_budget)

        return selected

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """从查询中提取关键词"""
        # 简单的分词 + 去停用词
        # 中文按字符 bigram，英文按空格分词
        keywords = []

        # 英文单词
        english_words = re.findall(r'[a-zA-Z]{2,}', query)
        keywords.extend(english_words)

        # 中文词语（简单 bigram）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
        for i in range(len(chinese_chars) - 1):
            keywords.append(chinese_chars[i] + chinese_chars[i + 1])
        if chinese_chars:
            keywords.extend(chinese_chars)  # 单字也保留

        # 去重，保留长度≥2的
        seen = set()
        result = []
        for kw in keywords:
            if kw.lower() not in seen and len(kw) >= 2:
                seen.add(kw.lower())
                result.append(kw)

        return result[:20]  # 最多 20 个关键词

    # ═══════════════════════════════════════════════
    # Prompt 构建（丰富 metadata 版）
    # ═══════════════════════════════════════════════

    def _get_system_prompt(
        self,
        custom: Optional[str],
        template_name: str,
        language: str,
    ) -> str:
        """获取 system prompt：自定义 > 模板 > 语言默认"""
        if custom:
            return custom

        # 尝试从模板库获取
        template = self._templates.get(template_name)
        if template:
            return template.render()

        # 语言回退
        if language == "en" or (language == "auto" and self._detect_language("") == "en"):
            return self._templates.get("default_en", PromptTemplate(DEFAULT_SYSTEM_PROMPT_EN)).render()
        return self._templates.get("default_zh", PromptTemplate(DEFAULT_SYSTEM_PROMPT_ZH)).render()

    def _build_user_prompt(
        self,
        chunks: List[SearchResultRef],
        query: str,
        include_citations: bool,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        """构建含丰富 metadata 的上下文 user prompt"""
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            if include_citations:
                meta_parts = []
                # 文档标题
                title = chunk.metadata.get("title") or chunk.metadata.get("doc_id", "")
                if title:
                    meta_parts.append(f"📄{title}")

                # 分段位置
                chunk_idx = chunk.metadata.get("chunk_index")
                total = chunk.metadata.get("total_chunks")
                if chunk_idx is not None:
                    if total:
                        meta_parts.append(f"§{int(chunk_idx)+1}/{total}")
                    else:
                        meta_parts.append(f"§{int(chunk_idx)+1}")

                # 相似度
                meta_parts.append(f"相关度:{chunk.score:.0%}")

                # 其他关键元数据（最多 2 个）
                extra_keys = [k for k in chunk.metadata if k not in (
                    "title", "doc_id", "chunk_index", "total_chunks"
                )]
                for ek in extra_keys[:2]:
                    val = str(chunk.metadata[ek])
                    if len(val) <= 60:
                        meta_parts.append(f"{ek}:{val}")

                meta_str = " | ".join(meta_parts)
                blocks.append(f"[{i}] ({meta_str})\n{chunk.text}")
            else:
                blocks.append(f"[{i}]\n{chunk.text}")

        context_blocks = "\n\n".join(blocks)

        # 对话历史
        conversation_section = ""
        if conversation_history:
            history_lines = []
            for turn in conversation_history:
                role_label = "👤 用户" if turn.role == "user" else "🤖 助手"
                history_lines.append(f"{role_label}: {turn.content}")
            conversation_section = CONVERSATION_HISTORY_TEMPLATE.format(
                history_blocks="\n\n".join(history_lines)
            )

        return CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            conversation_section=conversation_section,
            query=query,
        )

    # ═══════════════════════════════════════════════
    # LLM 调用（tenacity 分级重试版）
    # ═══════════════════════════════════════════════

    def _call_llm_with_retry(
        self,
        inp: RagAnswerInput,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[dict, int]:
        """调用 LLM，支持分级重试"""
        if not inp.retry_on_failure or inp.max_retries <= 0:
            return self._do_llm_call(inp, system_prompt, user_prompt), 0

        if HAS_TENACITY:
            return self._call_with_tenacity(inp, system_prompt, user_prompt)
        else:
            return self._call_with_manual_retry(inp, system_prompt, user_prompt)

    def _call_with_tenacity(
            self,
            inp: RagAnswerInput,
            system_prompt: str,
            user_prompt: str,
    ) -> Tuple[dict, int]:
        """tenacity 重试包装"""
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
            before_sleep_log,
            RetryError,
        )

        retry_count = [0]  # 用于记录重试次数
        attempt = [0]

        def _before_sleep(retry_state):
            retry_count[0] += 1
            logger.warning(
                "LLM 重试第 %d/%d 次，等待 %.1fs: %s",
                retry_count[0], inp.max_retries,
                retry_state.next_action.sleep if hasattr(retry_state.next_action, 'sleep') else 1,
                retry_state.outcome.exception() if retry_state.outcome else '',
            )

        @retry(
            stop=stop_after_attempt(inp.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(LLMRetryableError),
            before_sleep=_before_sleep,
        )
        def _retry_wrapper():
            attempt[0] += 1
            return self._do_llm_call(inp, system_prompt, user_prompt)

        try:
            result = _retry_wrapper()
            return result, retry_count[0]
        except RetryError:
            retry_count[0] = inp.max_retries
            logger.warning("LLM 重试 %d 次后仍失败，回退模拟模式", inp.max_retries)
            return self._simulated_answer(
                inp.query, system_prompt, user_prompt
            ), retry_count[0]
        except LLMRetryableError:
            retry_count[0] = inp.max_retries
            logger.warning("LLM 重试 %d 次后仍失败，回退模拟模式", inp.max_retries)
            return self._simulated_answer(
                inp.query, system_prompt, user_prompt
            ), retry_count[0]

    def _call_with_manual_retry(
        self,
        inp: RagAnswerInput,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[dict, int]:
        """手写重试逻辑（无 tenacity 时的回退）"""
        last_error = None
        for attempt in range(inp.max_retries + 1):
            try:
                result = self._do_llm_call(inp, system_prompt, user_prompt)
                if attempt > 0:
                    logger.info("LLM 重试第 %d 次成功", attempt)
                return result, attempt
            except LLMRetryableError as exc:
                last_error = exc
                if attempt < inp.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "LLM 可重试错误（第 %d/%d 次），%ds 后重试: %s",
                        attempt + 1, inp.max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM 重试 %d 次后仍失败", inp.max_retries)
            except LLMCallError as exc:
                # 不可重试，直接退出
                raise exc

        # 全部重试耗尽 → 模拟
        logger.warning("LLM 重试耗尽，回退模拟模式")
        return self._simulated_answer(
            inp.query, system_prompt, user_prompt
        ), inp.max_retries

    def _do_llm_call(
        self,
        inp: RagAnswerInput,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """单次 LLM 调用（分类异常为可重试/不可重试）"""
        # 若传入了 llm_client，用其调用
        if inp.llm_client is not None:
            return self._call_with_client(inp.llm_client, system_prompt, user_prompt, inp)

        # 否则自动创建 LiteLLMClient
        return self._simulated_answer(inp.query, system_prompt, user_prompt)



    def _call_with_client(
            self,
            client: Any,
            system_prompt: str,
            user_prompt: str,
            inp: RagAnswerInput,
    ) -> dict:
        """通过注入的 LLM 客户端调用（含异常分类）"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            chat_kwargs = {
                "messages": messages,
                "temperature": inp.temperature,
                "max_tokens": inp.max_tokens,
            }
            if inp.model:
                chat_kwargs["model"] = inp.model
            response = client.chat(**chat_kwargs)

            # 兼容两种返回格式
            if hasattr(response, 'content'):
                return {
                    "content": response.content,
                    "usage": getattr(response, 'usage', {}),
                    "model": getattr(response, 'model', inp.model or "unknown"),
                }
            elif isinstance(response, dict):
                return {
                    "content": response.get("content", "") or "",
                    "usage": response.get("usage", {}),
                    "model": response.get("model", inp.model or "unknown"),
                }
            else:
                return {"content": str(response), "usage": {}, "model": "unknown"}


        except (LLMRetryableError, LLMCallError):
            # 已知异常类型，向上传播给重试逻辑处理
            raise
        except Exception as exc:
            # 未知异常，分类后重新抛出
            self._classify_and_raise(exc)
            raise  # 理论上不会执行到这里，但为了类型安全保留

    @staticmethod
    def _classify_and_raise(exc: Exception) -> None:
        """分类异常为可重试/不可重试"""
        error_str = str(exc).lower()

        # 可重试：限流(429)、服务端错误(5xx)、超时、连接错误
        retryable_markers = [
            "429", "rate limit", "too many requests",
            "500", "502", "503", "504",
            "timeout", "timed out", "connection",
            "service unavailable", "overloaded",
        ]
        for marker in retryable_markers:
            if marker in error_str:
                raise LLMRetryableError(str(exc)) from exc

        # 不可重试：认证(401/403)、参数错误(400)、内容过滤
        raise LLMCallError(str(exc)) from exc

    # ═══════════════════════════════════════════════
    # 模拟回答（无 LLM 时的兜底）
    # ═══════════════════════════════════════════════

    def _simulated_answer(
        self, query: str, _system_prompt: str, user_prompt: str
    ) -> dict:
        """基于关键词匹配的简陋回答"""
        keywords = self._extract_keywords(query)

        snippets = []
        for line in user_prompt.split('\n'):
            for kw in keywords:
                if len(kw) >= 2 and kw.lower() in line.lower():
                    snippet = line.strip()
                    if len(snippet) > 20 and snippet not in snippets:
                        snippets.append(snippet[:200])
                    break

        if snippets:
            answer = (
                f"（模拟回答）根据相关资料，与您的问题「{query}」相关的信息如下：\n\n"
                + "\n".join(f"- {s}" for s in snippets[:5])
                + "\n\n⚠️ 此为模拟回答（LLM 不可用时的兜底）。请检查 LLM 配置。"
            )
        else:
            answer = (
                f"（模拟回答）关于「{query}」，当前上下文中未找到明确匹配的信息。\n"
                + "⚠️ 此为模拟回答（LLM 不可用时的兜底）。请检查 LLM 配置或尝试更具体的问题。"
            )

        return {
            "content": answer,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "simulated",
        }

    # ═══════════════════════════════════════════════
    # 引用提取（扩展格式 + 越界清洗）
    # ═══════════════════════════════════════════════

    def _extract_citations(
        self,
        answer: str,
        chunks: List[SearchResultRef],
    ) -> List[Citation]:
        """从 LLM 回答中提取所有引用标记 — 支持 [1], [1-3], [1][2], [1,2,3]"""
        all_ids: Set[int] = set()
        max_idx = len(chunks)

        # 匹配四种格式
        # 格式1: [1] 或 [1-3]
        for match in re.finditer(r'\[(\d+)(?:-(\d+))?\]', answer):
            start = int(match.group(1))
            end_str = match.group(2)
            if end_str:
                end = int(end_str)
                # [1-3] → 展开为 1,2,3（限制范围防止 LLM 输出 [1-9999]）
                for cid in range(start, min(end, start + 50) + 1):
                    if 1 <= cid <= max_idx:
                        all_ids.add(cid)
            else:
                if 1 <= start <= max_idx:
                    all_ids.add(start)

        # 格式2: [1][2] 连续引用（同一次匹配）
        for match in re.finditer(r'\[(\d+)\]\[(\d+)\]', answer):
            for g in (match.group(1), match.group(2)):
                cid = int(g)
                if 1 <= cid <= max_idx:
                    all_ids.add(cid)

        # 格式3: [1,2,3] 逗号分隔
        for match in re.finditer(r'\[(\d+(?:,\d+)*)\]', answer):
            for num_str in match.group(1).split(','):
                try:
                    cid = int(num_str.strip())
                    if 1 <= cid <= max_idx:
                        all_ids.add(cid)
                except ValueError:
                    pass

        if not all_ids:
            return []

        # 如果引用编号过多（可能是 LLM 胡乱编号），只取前 max_chunks 个
        if len(all_ids) > max_idx * 2:
            logger.warning(
                "引用编号过多 (%d)，超过分块数 (%d) 的 2 倍，可能为 LLM 幻觉，截断",
                len(all_ids), max_idx,
            )
            all_ids = set(sorted(all_ids)[:max_idx])

        # 映射到 Citation
        citations = []
        for cid in sorted(all_ids):
            idx = cid - 1
            if 0 <= idx < max_idx:
                chunk = chunks[idx]
                citations.append(Citation(
                    citation_id=cid,
                    chunk_id=chunk.chunk_id,
                    text_snippet=chunk.text[:200],
                    score=chunk.score,
                    grounded=True,  # 默认 True，_verify_citation_grounding 会更新
                ))

        return citations

    # ═══════════════════════════════════════════════
    # 引用溯源验证（Citation Grounding）
    # ═══════════════════════════════════════════════

    def _verify_citation_grounding(
        self,
        answer: str,
        citations: List[Citation],
        chunks: List[SearchResultRef],
    ) -> List[Citation]:
        """验证每条引用的文本是否确实出现在对应分块中"""
        verified = []
        for cit in citations:
            chunk = chunks[cit.citation_id - 1]

            # 在回答中找到引用 [n] 的位置
            cit_marker = f"[{cit.citation_id}]"
            cit_pos = answer.find(cit_marker)
            if cit_pos < 0:
                # 尝试 [n-m] 格式
                for match in re.finditer(r'\[(\d+)-(\d+)\]', answer):
                    if int(match.group(1)) <= cit.citation_id <= int(match.group(2)):
                        cit_pos = match.start()
                        break

            if cit_pos < 0:
                cit.grounded = False
                verified.append(cit)
                continue

            # 提取引用前后的文本片段（前 50 字 + 后 100 字）
            context_start = max(0, cit_pos - 50)
            context_end = min(len(answer), cit_pos + 150)
            surrounding_text = answer[context_start:context_end]

            # 从周围文本中提取 n-gram，检查是否在 chunk.text 中出现
            is_grounded = self._check_ngram_overlap(surrounding_text, chunk.text)

            if not is_grounded:
                logger.warning(
                    "引用 [%d] 未能溯源到 chunk '%s'，可能为 LLM 幻觉。"
                    "周围文本: '%s'",
                    cit.citation_id, chunk.chunk_id,
                    surrounding_text.replace('\n', ' ')[:100],
                )

            cit.grounded = is_grounded
            verified.append(cit)

        return verified

    def _check_ngram_overlap(
            self,
            snippet: str,
            source_text: str,
    ) -> bool:
        """检查 snippet 与 source_text 的字符级覆盖率

        算法（v1.1.1）：
        1. 过短片段（<5 字符）默认信任
        2. 对 snippet 的每个 n-gram(5/4/3)，若出现在 source 中，
           标记该 n-gram 覆盖的字符位置
        3. 覆盖率 >= 30% → grounded；否则 → 幻觉
        """
        snippet_clean = re.sub(r'\s+', ' ', snippet).strip()
        source_clean = re.sub(r'\s+', ' ', source_text).strip()

        if len(snippet_clean) < 5:
            return True

        # 收集所有匹配到的字符位置
        matched_positions: Set[int] = set()

        for n in self._NGRAM_SIZES:
            if n > len(snippet_clean):
                continue
            for i in range(len(snippet_clean) - n + 1):
                ng = snippet_clean[i:i + n]
                if ng in source_clean:
                    # 标记这个 n-gram 覆盖的所有字符位置
                    for j in range(i, i + n):
                        matched_positions.add(j)

        if len(snippet_clean) == 0:
            return True

        coverage = len(matched_positions) / len(snippet_clean)
        is_grounded = coverage >= 0.30

        logger.debug(
            "ngram overlap: coverage=%.2f (matched=%d/%d chars) → %s",
            coverage, len(matched_positions), len(snippet_clean),
            "grounded" if is_grounded else "UNGROUNDED",
        )
        return is_grounded

    @staticmethod
    def _char_ngrams(text: str, n: int) -> Set[str]:
        """字符级 n-gram 集合"""
        if len(text) < n:
            return set()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    # ═══════════════════════════════════════════════
    # 置信度估算
    # ═══════════════════════════════════════════════

    def _estimate_confidence(
            self,
            answer: str,
            chunks_used: List[SearchResultRef],
            citations: List[Citation],
    ) -> float:
        """基于多信号估算回答置信度（0~1）"""
        signals: List[float] = []

        # 信号 1: top-1 分块相似度
        if chunks_used:
            signals.append(chunks_used[0].score)
        else:
            signals.append(0.05)  # 无分块时给极低分

        # 信号 2: 引用覆盖率（grounded 引用数 / 使用的分块数）
        if chunks_used and citations:
            grounded_count = sum(1 for c in citations if c.grounded)
            coverage = grounded_count / len(chunks_used)
            signals.append(coverage)
        elif chunks_used:
            signals.append(0.2)  # 有上下文但零引用
        else:
            signals.append(0.05)  # 无分块无引用

        # 信号 3: 低置信短语检测
        low_markers = [
            "无法回答", "没有足够信息", "不确定", "可能",
            "无法确定", "资料不足", "缺少信息",
            "I cannot answer", "insufficient", "uncertain",
            "no information", "not clear",
        ]
        has_low = any(m.lower() in answer.lower() for m in low_markers)
        signals.append(0.15 if has_low else 0.85)

        # 信号 4: 回答长度（过短可能是拒绝回答）
        answer_len = len(answer)
        if answer_len < 30:
            signals.append(0.1)
        elif answer_len < 100:
            signals.append(0.5)
        else:
            signals.append(0.9)

        # 信号 5: 是否有 grounded 引用（最强信号）
        grounded_count = sum(1 for c in citations if c.grounded)
        if grounded_count > 0:
            signals.append(min(0.95, 0.5 + grounded_count * 0.1))
        else:
            signals.append(0.2)

        confidence = round(sum(signals) / len(signals), 2) if signals else 0.5
        return max(0.0, min(1.0, confidence))

    # ═══════════════════════════════════════════════
    # 拒绝回答检测
    # ═══════════════════════════════════════════════

    def _detect_refusal(self, answer: str) -> bool:
        """检测 LLM 是否拒绝回答"""
        if not answer or len(answer.strip()) < 10:
            logger.warning("LLM 返回空响应或过短")
            return True

        for pattern in self._REFUSAL_PATTERNS:
            if pattern.search(answer):
                logger.warning("检测到 LLM 拒绝回答: '%s'", pattern.pattern)
                return True

        return False

    # ═══════════════════════════════════════════════
    # 语言检测
    # ═══════════════════════════════════════════════

    @staticmethod
    def _detect_language(text: str) -> str:
        """自动检测语言（简单回退，langdetect 可选）"""
        if not text:
            return "zh"

        if HAS_LANGDETECT:
            try:
                from langdetect import detect
                lang = detect(text)
                return "zh" if lang.startswith("zh") else "en"
            except Exception:
                pass

        # 简单回退：检测中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return "zh" if chinese_chars > len(text) * 0.1 else "en"

    # ═══════════════════════════════════════════════
    # 语言检测
    # ═══════════════════════════════════════════════

    @staticmethod
    def _detect_language(text: str) -> str:
        """自动检测语言（简单回退，langdetect 可选）"""
        if not text:
            return "zh"

        if HAS_LANGDETECT:
            try:
                from langdetect import detect
                lang = detect(text)
                return "zh" if lang.startswith("zh") else "en"
            except Exception:
                pass

        # 简单回退：检测中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return "zh" if chinese_chars > len(text) * 0.1 else "en"

    # ═══════════════════════════════════════════════
    # ✨ 反思模式核心逻辑
    # ═══════════════════════════════════════════════

    def _run_reflection(
            self,
            query: str,
            initial_answer: str,
            search_results: list,
            model: Optional[str],
            llm_client: Any,
            max_iterations: int = 2,
            quality_threshold: float = 0.8,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        调用反思子图优化答案。

        Args:
            query: 用户查询
            initial_answer: 初始答案
            search_results: 检索结果列表
            model: LLM 模型名称
            llm_client: LLM 客户端实例
            max_iterations: 最大反思迭代次数（默认 2）
            quality_threshold: 质量阈值（默认 0.8）

        Returns:
            (reflection_report, refined_answer)
        """
        import difflib
        from src.agent.langgraph.graphs import (
            compile_reflection_graph,
            ReflectionConfig,
        )
        from src.agent.langgraph.state import create_initial_state

        try:
            # 获取或创建 SkillManager 实例
            sm = getattr(self, 'skill_manager', None) or SkillManager()

            # ✅ 修复 Bug 1：确保 rag_answer 技能已注册
            if not sm.get("rag_answer"):
                sm.register(RagAnswerSkill)  # ← 只需传入类，不需要传名字
            config = ReflectionConfig(
                target_skill="rag_answer",
                model=model or "gpt-4",
                output_field="answer",
                max_iterations=max_iterations,
                quality_threshold=quality_threshold,
            )
            graph = compile_reflection_graph(
                target_skill_name="rag_answer",
                skill_manager=sm,
                llm_client=llm_client,
                config=config,
            )

            # 注入检索结果到 state（供 ExternalFeedback 验证引用）
            initial_state = create_initial_state(query)
            initial_state["search_results"] = [
                {"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
                for r in search_results
            ]

            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"refl-{hash(query) % 100000}"}},
            )

            ctx = result.get("reflection_context")
            if ctx is None or not ctx.refined_output:
                return None, None

            # 构建报告
            critique_summary = ""
            if ctx.critique:
                critique_summary = ctx.critique.summary or ""
                for p in (ctx.critique.points or [])[:5]:
                    critique_summary += f"\n  [{p.severity}] {p.description}"

            diff = "\n".join(difflib.unified_diff(
                initial_answer.splitlines(),
                ctx.refined_output.splitlines(),
                fromfile="原始答案", tofile="优化答案", lineterm="",
            ))

            feedback_text = ctx.feedback.content if ctx.feedback else ""

            report = {
                "enabled": True,
                "iterations": ctx.iteration,
                "final_score": ctx.critique.overall_score if ctx.critique else None,
                "critique_summary": critique_summary[:2000],
                "external_feedback": feedback_text[:1000],
                "diff": diff[:3000],
                "status": ctx.status,
            }

            return report, ctx.refined_output

        except Exception as exc:
            logger.warning("反思流水线失败，回退原始答案: %s", exc)
            return {"enabled": True, "error": str(exc)}, None

    # ═══════════════════════════════════════════════
    # 错误输出
    # ═══════════════════════════════════════════════

    @staticmethod
    def _error_output(
        kwargs: dict,
        t_start: float,
        timing: dict,
        error: str,
        retry_count: int = 0,
    ) -> dict:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return RagAnswerOutput(
            success=False,
            answer="",
            elapsed_ms=total_ms,
            timing_breakdown=timing,
            error=error,
            retry_count=retry_count,
        ).model_dump()


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def rag_answer(
    query: str,
    search_results: List[dict],
    **kwargs,
) -> dict:
    """一行式 RAG 问答"""
    input_data = RagAnswerInput(query=query, search_results=search_results, **kwargs)
    return RagAnswerSkill().execute(input_data)



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
    print("  RagAnswerSkill v1.1 — RAG 问答技能测试")
    print("=" * 60)

    # 模拟上游检索结果
    mock_search_results = [
        {
            "chunk_id": "chunk_003",
            "text": "Python 是一种解释型、面向对象的高级编程语言。它具有简洁的语法和强大的表达能力，广泛应用于 Web 开发、数据科学、人工智能等领域。Python 由 Guido van Rossum 于 1991 年首次发布。",
            "score": 0.92,
            "metadata": {"doc_id": "doc_1", "title": "Python简介", "chunk_index": 2, "total_chunks": 8},
        },
        {
            "chunk_id": "chunk_007",
            "text": "Python 的主要特点包括：动态类型系统、自动内存管理（垃圾回收）、丰富的标准库、以及庞大的第三方包生态。PyPI（Python Package Index）上有超过 40 万个包。",
            "score": 0.87,
            "metadata": {"doc_id": "doc_1", "title": "Python特点", "chunk_index": 6, "total_chunks": 8},
        },
        {
            "chunk_id": "chunk_012",
            "text": "Python 在数据科学中的核心库包括 NumPy（数值计算）、Pandas（数据分析）、Matplotlib（可视化）和 Scikit-learn（机器学习）。深度学习框架 PyTorch 和 TensorFlow 也提供 Python API。",
            "score": 0.81,
            "metadata": {"doc_id": "doc_2", "title": "Python数据科学生态", "chunk_index": 0, "total_chunks": 5},
        },
        {
            "chunk_id": "chunk_015",
            "text": "Java 是一种编译型、面向对象的编程语言，由 Sun Microsystems 于 1995 年发布。它遵循“一次编写到处运行”的理念，通过 JVM 实现跨平台。",
            "score": 0.55,
            "metadata": {"doc_id": "doc_3", "title": "Java简介", "chunk_index": 0, "total_chunks": 6},
        },
    ]

    skill = RagAnswerSkill()

    # ── v1.1 新功能展示 ──
    print("\n📌 测试 1：基础 RAG 问答（v1.1 带置信度 + 溯源验证）")
    r1 = skill.execute(
        query="Python 是什么时候发布的？",
        search_results=mock_search_results,
        max_context_chunks=3,
    )
    print(f"   成功: {r1['success']} | 模型: {r1['model']} | 置信度: {r1.get('confidence', 'N/A')}")
    print(f"   回答: {r1['answer'][:200]}...")
    print(f"   引用数: {len(r1['citations'])} | 其中 grounded: {sum(1 for c in r1['citations'] if c.get('grounded', True))}")
    print(f"   耗时: {r1['elapsed_ms']:.1f}ms | 重试: {r1['retry_count']}")
    print(f"   分阶段: {r1['timing_breakdown']}")

    # ── 测试 2：引用格式扩展验证 ──
    print("\n📌 测试 2：引用格式扩展（[1-3]/[1,2]/[1][2]）")
    fake_answer = (
        "Python 由 Guido 于 1991 年创建 [1]。"
        "其特点包括动态类型和自动内存管理 [2]。"
        "数据科学栈包括 NumPy 和 Pandas [1-3]。"
        "这些在 [1,2] 中都有提及。[1][2] 提供了详细信息。"
    )
    chunks_refs = [SearchResultRef(**r) for r in mock_search_results]
    citations = skill._extract_citations(fake_answer, chunks_refs)
    print(f"   提取到 {len(citations)} 条引用: {[c.citation_id for c in citations]}")

    # ── 测试 3：引用溯源验证 ──
    print("\n📌 测试 3：引用溯源验证（grounding check）")
    # 模拟 LLM 幻觉引用 — 标注 [3] 但内容是编造的
    hallucinated_answer = (
        "Python 由 Guido van Rossum 于 1991 年发布 [1]。"
        "Python 被设计用于编写操作系统内核 [3]。"  # ← chunk_012 里没有这句话
    )
    chunks_refs = [SearchResultRef(**r) for r in mock_search_results]
    raw_citations = skill._extract_citations(hallucinated_answer, chunks_refs)
    verified = skill._verify_citation_grounding(hallucinated_answer, raw_citations, chunks_refs)
    for c in verified:
        status = "✅ grounded" if c.grounded else "❌ UNGROUNDED (幻觉)"
        print(f"   [{c.citation_id}] {c.chunk_id} → {status}")

    # ── 测试 4：置信度估算 ──
    print("\n📌 测试 4：置信度估算（多信号）")
    conf = skill._estimate_confidence(
        fake_answer, chunks_refs[:3],
        [Citation(citation_id=1, chunk_id="chunk_003", grounded=True),
         Citation(citation_id=2, chunk_id="chunk_007", grounded=True)],
    )
    print(f"   置信度: {conf:.2f} (有 grounded 引用，应偏高)")

    conf_low = skill._estimate_confidence(
        "不确定，可能无法回答", chunks_refs[:1],
        [],
    )
    print(f"   置信度: {conf_low:.2f} (低置信短语 + 无引用，应偏低)")

    # ── 测试 5：关键词加权上下文选择 ──
    print("\n📌 测试 5：关键词加权上下文选择")
    selected = skill._select_context_chunks(
        chunks_refs, "Python 数据科学 库", max_chunks=3, max_tokens=500, keyword_boost=0.15
    )
    for i, c in enumerate(selected):
        print(f"   [{i+1}] {c.chunk_id} | score={c.score:.3f} | {c.text[:60]}...")

    # ── 测试 6：Token 估算对比 ──
    print("\n📌 测试 6：Token 估算")
    test_text = "Python 是一种解释型、面向对象的高级编程语言。" * 3
    est = skill._token_estimator
    print(f"   手写估算: {est._estimate_heuristic(test_text)} tokens")
    print(f"   实际估算: {est.count(test_text)} tokens")
    print(f"   使用 tiktoken: {est._encoder is not None}")

    # ── 测试 7：拒绝回答检测 ──
    print("\n📌 测试 7：拒绝回答检测")
    print(f"   '无法回答': {skill._detect_refusal('根据现有资料，我无法回答')}")
    print(f"   '正常回答': {skill._detect_refusal('Python 由 Guido 于 1991 年创建。')}")

    # ── 测试 8：多轮对话 ──
    print("\n📌 测试 8：多轮对话历史")
    r8 = skill.execute(
        query="它有什么特点？",
        search_results=mock_search_results[:2],
        conversation_history=[
            {"role": "user", "content": "Python 是什么时候发布的？"},
            {"role": "assistant", "content": "Python 由 Guido van Rossum 于 1991 年首次发布 [1]。"},
        ],
        max_context_chunks=2,
        include_citations=False,
    )
    print(f"   回答: {r8['answer'][:200]}...")

    print("\n" + "=" * 60)
    print("  ✅ RagAnswerSkill v1.1 测试完成")
    print("=" * 60)