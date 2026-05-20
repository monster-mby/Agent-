"""
文档分块技能 — RAG 体系第二环（v3 生产版）

将长文本按语义边界切分为可嵌入的片段。
支持四种策略：recursive（推荐）、paragraph、sentence、fixed。
中英文自适应分句，Token 级切分，语义边界对齐重叠，上下文窗口携带。

核心特性：
- 三层分句引擎降级（sentence-splitter → spaCy → 正则）
- 可选后端适配器（LangChain / LLaMAIndex）
- 全文一次 Token 编码，O(n) 指针追踪
- 语义感知小块合并 + 上下文窗口 + 去重标记
- 流式大文档保护 + 批处理模式 + 缓存幂等

依赖（按需安装）：
    pip install nltk tiktoken langid
    pip install sentence-splitter             # 推荐，轻量多语言分句
    pip install spacy && python -m spacy download en_core_web_sm  # 可选，英文精准分句
    pip install langchain                     # 可选，作为切分后端
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from statistics import mean, median
from typing import (
    Any, ClassVar, Dict, Iterator, List, Literal, Optional, Pattern,
    Sequence, TypedDict, Union,
)

from pydantic import BaseModel, Field, field_validator, model_validator

from src.skills.base.base_skill import BaseSkill


logger = logging.getLogger("document_chunker")


# ═══════════════════════════════════════════════════════════════
# TypedDict — 中间数据结构（P2-11 类型精确化）
# ═══════════════════════════════════════════════════════════════

class _SegmentMeta(TypedDict, total=False):
    """切分片段的中间元数据"""
    text: str
    char_count: int
    byte_count: int
    token_count: int
    start_pos: int
    end_pos: int
    boundary_type: str
    is_complete_sentence: bool


class _ChunkStats(TypedDict):
    """分块大小分布统计"""
    min: int
    max: int
    avg: float
    median: float
    p95: float
    total: int


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class Chunk(BaseModel):
    """单个分块，包含完整元数据以便下游召回调权"""

    index: int = Field(..., description="分块序号（从 0 开始）")
    text: str = Field(..., description="分块文本内容")
    char_count: int = Field(..., description="字符数")
    byte_count: int = Field(..., description="UTF-8 字节数")
    token_count: int = Field(default=0, description="Token 数（若分词器可用）")
    start_pos: int = Field(..., description="原文中的起始字符位置")
    end_pos: int = Field(..., description="原文中的结束字符位置")
    chunk_type: Literal["paragraph", "sentence", "fixed", "recursive"] = Field(
        default="sentence", description="分块策略来源"
    )
    boundary_type: Literal[
        "paragraph_break", "sentence_end", "char_limit",
        "token_limit", "recursive_fallback"
    ] = Field(default="sentence_end", description="触发分块的边界类型")
    overlap_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="与前一块的重叠比例，供检索去重/降权"
    )
    is_near_duplicate_of: Optional[int] = Field(
        default=None,
        description="若 overlap_ratio > 阈值，指向目标 chunk index，供下游去重"
    )
    language: str = Field(default="unknown", description="检测到的语言")
    context_prefix: Optional[str] = Field(
        default=None, description="前一个 Chunk 的末句，作为上下文前缀"
    )
    context_suffix: Optional[str] = Field(
        default=None, description="后一个 Chunk 的首句，作为上下文后缀"
    )

    @field_validator("context_prefix", "context_suffix")
    @classmethod
    def _truncate_context(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 200:
            return v[-200:] if "prefix" in cls.__name__ else v[:200]
        return v

    @model_validator(mode="after")
    def _validate_positions(self) -> "Chunk":
        if self.start_pos >= self.end_pos:
            raise ValueError(
                f"start_pos({self.start_pos}) 必须 < end_pos({self.end_pos})"
            )
        if abs(len(self.text) - (self.end_pos - self.start_pos)) > 10:
            raise ValueError(
                f"text 长度({len(self.text)}) 与位置区间 "
                f"[{self.start_pos}, {self.end_pos}) 不一致"
            )
        return self


class ContextExtract(BaseModel):
    """从 Chunk 提取的上下文窗口"""
    last_sentence: Optional[str] = None
    first_sentence: Optional[str] = None


class DocumentChunkerInput(BaseModel):
    """输入模型"""

    # ── 文本输入：支持字符串或文件路径（P0-12 流式保护）──
    text: Optional[str] = Field(
        default=None,
        description="待分块的原始文本（与 file_path 二选一）"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="待分块的文件路径（自动流式读取大文件）"
    )

    strategy: Literal["recursive", "paragraph", "sentence", "fixed"] = Field(
        default="recursive",
        description="分块策略：recursive=递归语义切分(推荐)"
    )
    chunk_size: int = Field(
        default=500, ge=50, le=5000,
        description="目标字符数（recursive/fixed 生效）"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500,
        description="相邻分块重叠字符数"
    )
    min_chunk_size: int = Field(
        default=20, ge=1,
        description="最小字符数，低于此值合并到相邻块"
    )
    use_tokenizer: bool = Field(
        default=True,
        description="是否启用 Token 级计数（需 tiktoken）"
    )
    encoding_name: str = Field(
        default="cl100k_base",
        description="tiktoken encoding 名称"
    )

    # ── P1-9: 文档类型预设分隔符 ──
    document_type: Literal["text", "markdown", "code", "html", "auto"] = Field(
        default="auto",
        description="文档类型，影响 recursive 策略的分隔符选择"
    )
    separators: Optional[List[str]] = Field(
        default=None,
        description="自定义 recursive 分隔符优先级（覆盖 document_type 预设）"
    )

    # ── P0-13: 文本规范化 ──
    normalize_text: bool = Field(
        default=True,
        description="是否做 Unicode 规范化 + 全角转半角 + 零宽字符去除"
    )

    # ── P1-14: 上下文窗口 ──
    include_context_window: bool = Field(
        default=True,
        description="是否为每个 Chunk 携带相邻块的上下文前后缀"
    )

    # ── 去重阈值 ──
    dedup_overlap_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="overlap_ratio 超过此值标记 is_near_duplicate_of"
    )

    # ── 大文档阈值 ──
    max_memory_mb: int = Field(
        default=50, ge=1,
        description="文本超过此 MB 数时启用流式分页处理"
    )

    # ── 可选后端注入 ──
    backend: Literal["native", "langchain", "llama_index"] = Field(
        default="native",
        description="分块后端：native=内置实现，langchain/llama_index=外部库"
    )

    @model_validator(mode="after")
    def _check_text_source(self) -> "DocumentChunkerInput":
        if self.text is None and self.file_path is None:
            raise ValueError("必须提供 text 或 file_path")
        if self.text is not None and self.file_path is not None:
            raise ValueError("text 和 file_path 不能同时提供，请二选一")
        return self


class DocumentChunkerOutput(BaseModel):
    """输出模型"""
    success: bool = Field(..., description="是否成功")
    chunks: List[Chunk] = Field(default_factory=list, description="分块列表")
    total_chunks: int = Field(default=0, description="分块总数")
    strategy: str = Field(default="", description="实际使用的策略")
    source_char_count: int = Field(default=0, description="原文总字符数")
    source_token_count: int = Field(default=0, description="原文总 Token 数")
    chunk_size_stats: _ChunkStats = Field(
        default_factory=lambda: _ChunkStats(min=0, max=0, avg=0.0, median=0.0, p95=0.0, total=0),
        description="分块大小分布统计"
    )
    chunk_set_id: Optional[str] = Field(
        default=None, description="幂等性 ID（文本+参数 MD5）"
    )
    elapsed_ms: float = Field(default=0.0, description="耗时（毫秒）")
    error: Optional[str] = Field(default=None, description="错误信息")


# ═══════════════════════════════════════════════════════════════
# 文档类型 → 分隔符预设（P1-9）
# ═══════════════════════════════════════════════════════════════

_DOCUMENT_TYPE_SEPARATORS: Dict[str, List[str]] = {
    "text": [
        "\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", " ", ""
    ],
    "markdown": [
        "\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", ". ", " ", ""
    ],
    "code": [
        "\ndef ", "\nclass ", "\nasync def ", "\n# ", "\n// ", "\n\n",
        "\n", "; ", " ", ""
    ],
    "html": [
        "\n</section>", "\n</article>", "\n</div>", "\n</p>", "\n<br>",
        "\n\n", "\n", "。", ". ", " ", ""
    ],
}


# ═══════════════════════════════════════════════════════════════
# 技能类 v3
# ═══════════════════════════════════════════════════════════════

class DocumentChunkerSkill(BaseSkill):
    """文档分块技能 — v3 生产版

    策略一览：
    - recursive  : 按分隔符优先级递归切分，语义最友好（推荐默认）
    - paragraph  : 按连续空行切分（兼容 \\r\\n）
    - sentence   : 三层分句引擎降级
    - fixed      : 字符级/Token 级固定窗口 + 语义边界对齐重叠

    后端：
    - native     : 自研实现（零额外依赖）
    - langchain  : LangChain TextSplitter 系列
    - llama_index: LLaMAIndex NodeParser 系列
    """
    # ── 元数据（类变量，BaseSkill 初始化时需要）──
    name = "document_chunker"
    description = (
        "将长文本按语义边界切分为可嵌入片段，支持 recursive/paragraph/"
        "sentence/fixed 四种策略，三层分句引擎降级，Token 级计数，"
        "语义边界对齐重叠，上下文窗口携带，批处理与缓存幂等"
    )
    version = "3.0.0"
    author = "EnterpriseLearningAgent"
    triggers = ["分块", "切分文档", "文本分块", "chunk", "分割文本", "文档切片"]

    input_schema = DocumentChunkerInput
    output_schema = DocumentChunkerOutput

    # ── 预编译正则（类级常量，复用） ─────────────────────
    _PARAGRAPH_SPLITTER: ClassVar[Pattern] = re.compile(r"\r?\n\s*\r?\n")
    _NEWLINE_NORMALIZER: ClassVar[Pattern] = re.compile(r"\r\n?")
    _SENTENCE_FALLBACK: ClassVar[Pattern] = re.compile(
        r"(?<=[。！？.!?])(?=[)）】\"」』\s]*(?:[^\s]|$))"
    )
    _ZERO_WIDTH_CHARS: ClassVar[Pattern] = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]"
    )
    _FULLWIDTH_MAP: ClassVar[Dict[int, int]] = {
        # 全角数字→半角
        0xFF10: 0x0030, 0xFF11: 0x0031, 0xFF12: 0x0032,
        0xFF13: 0x0033, 0xFF14: 0x0034, 0xFF15: 0x0035,
        0xFF16: 0x0036, 0xFF17: 0x0037, 0xFF18: 0x0038,
        0xFF19: 0x0039,
        # 全角大写字母→半角
        0xFF21: 0x0041, 0xFF22: 0x0042, 0xFF23: 0x0043,
        0xFF24: 0x0044, 0xFF25: 0x0045, 0xFF26: 0x0046,
        0xFF27: 0x0047, 0xFF28: 0x0048, 0xFF29: 0x0049,
        0xFF2A: 0x004A, 0xFF2B: 0x004B, 0xFF2C: 0x004C,
        0xFF2D: 0x004D, 0xFF2E: 0x004E, 0xFF2F: 0x004F,
        0xFF30: 0x0050, 0xFF31: 0x0051, 0xFF32: 0x0052,
        0xFF33: 0x0053, 0xFF34: 0x0054, 0xFF35: 0x0055,
        0xFF36: 0x0056, 0xFF37: 0x0057, 0xFF38: 0x0058,
        0xFF39: 0x0059, 0xFF3A: 0x005A,
        # 全角小写字母→半角
        0xFF41: 0x0061, 0xFF42: 0x0062, 0xFF43: 0x0063,
        0xFF44: 0x0064, 0xFF45: 0x0065, 0xFF46: 0x0066,
        0xFF47: 0x0067, 0xFF48: 0x0068, 0xFF49: 0x0069,
        0xFF4A: 0x006A, 0xFF4B: 0x006B, 0xFF4C: 0x006C,
        0xFF4D: 0x006D, 0xFF4E: 0x006E, 0xFF4F: 0x006F,
        0xFF50: 0x0070, 0xFF51: 0x0071, 0xFF52: 0x0072,
        0xFF53: 0x0073, 0xFF54: 0x0074, 0xFF55: 0x0075,
        0xFF56: 0x0076, 0xFF57: 0x0077, 0xFF58: 0x0078,
        0xFF59: 0x0079, 0xFF5A: 0x007A,
    }

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._token_ids_cache = None
        self._sent_splitter = False
        self._lang_detector = None
        # 懒加载单例
        self._tokenizer: Any = None
        self._lang_detector: Any = None
        self._sent_splitter: Any = None
        self._token_ids_cache: Optional[List[int]] = None  # P0-8: 全文 Token 缓存

    # ── Schema ───────────────────────────────────────────



    # ═══════════════════════════════════════════════════════
    # 核心入口
    # ═══════════════════════════════════════════════════════

    def execute(self, input_data: DocumentChunkerInput) -> dict:
        t_start = time.perf_counter()

        try:
            # ── P2-10: 关键步骤日志 ──────────────────────
            logger.info(
                "DocumentChunker 开始 | strategy=%s | doc_type=%s | "
                "backend=%s | chunk_size=%d | overlap=%d",
                input_data.strategy, input_data.document_type,
                input_data.backend, input_data.chunk_size, input_data.chunk_overlap,
            )

            # ── P0-12: 读取文本（支持流式大文件） ─────────
            text = self._load_text(input_data)

            # ── P0-13: 文本规范化 ────────────────────────
            if input_data.normalize_text:
                text = self._normalize_text(text)

            # P0-12: 极短文本兜底
            if len(text) <= input_data.min_chunk_size:
                logger.info("文本过短，返回单块 | len=%d", len(text))
                return self._build_output(
                    [self._make_single_chunk(text, input_data)],
                    input_data, t_start, text,
                )

            # ── P0-8: 全文一次 Token 编码（缓存） ─────────
            total_tokens = self._precompute_tokens(text, input_data)

            # ── P1-1/3: 可选后端适配器 ──────────────────
            if input_data.backend != "native":
                chunks = self._execute_via_backend(text, input_data, total_tokens)
            else:
                chunks = self._execute_native(text, input_data, total_tokens)

            # ── P1-7: 语义感知小块合并 ──────────────────
            chunks = self._merge_small_chunks_semantic(
                chunks, input_data.min_chunk_size, text
            )
            merged_count = len(chunks)

            # ── P1-14: 附加上下文窗口 ────────────────────
            if input_data.include_context_window:
                self._attach_context_windows(chunks)

            # ── P2-15: 去重标记 ─────────────────────────
            self._mark_near_duplicates(chunks, input_data.dedup_overlap_threshold)

            # 重新编号
            for i, c in enumerate(chunks):
                c.index = i

            logger.info(
                "分块完成 | total=%d | merged_to=%d | token_count=%d | elapsed=%.1fms",
                len(chunks) + (merged_count - len(chunks)), len(chunks),
                total_tokens, (time.perf_counter() - t_start) * 1000,
            )

            return self._build_output(chunks, input_data, t_start, text)

        except Exception as exc:
            logger.exception("DocumentChunker 执行失败")
            return DocumentChunkerOutput(
                success=False,
                error=str(exc),
                strategy=input_data.strategy,
                source_char_count=len(input_data.text or ""),
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).model_dump()

    # ═══════════════════════════════════════════════════════
    # P2-17: 批处理模式
    # ═══════════════════════════════════════════════════════

    def execute_batch(
            self,
            texts: List[str],
            strategy: str = "recursive",
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            **common_kwargs,
    ) -> List[dict]:
        """批量处理多篇文档，共用分词器和语言检测器实例"""
        logger.info("批处理开始 | count=%d", len(texts))
        t_start = time.perf_counter()
        results: List[dict] = []

        for i, text in enumerate(texts):
            input_data = DocumentChunkerInput(
                text=text,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **common_kwargs,
            )
            result = self.execute(input_data)
            results.append(result)

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info("批处理完成 | count=%d | total_elapsed=%.1fms | avg=%.1fms",
                    len(texts), elapsed, elapsed / max(len(texts), 1))
        return results

    # ═══════════════════════════════════════════════════════
    # P0-12: 文本加载（字符串 / 文件路径 / 流式大文件）
    # ═══════════════════════════════════════════════════════

    def _load_text(self, inp: DocumentChunkerInput) -> str:
        if inp.text is not None:
            text_size_mb = len(inp.text.encode("utf-8")) / (1024 * 1024)
            if text_size_mb > inp.max_memory_mb:
                logger.warning(
                    "文本过大 (%.1fMB)，但已完全加载到内存。建议使用 file_path 以流式处理",
                    text_size_mb,
                )
            return inp.text

        # file_path 分支：流式读取
        import os
        file_size_mb = os.path.getsize(inp.file_path) / (1024 * 1024)
        if file_size_mb > inp.max_memory_mb:
            logger.info("大文件检测 | size=%.1fMB | 启用流式分页读取", file_size_mb)
            return self._stream_read(inp.file_path, inp.max_memory_mb * 1024 * 1024)

        with open(inp.file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _stream_read(file_path: str, max_bytes: int) -> str:
        """流式读取：一次最多读 max_bytes 字节"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(max_bytes)

    # ═══════════════════════════════════════════════════════
    # P0-13: 文本规范化
    # ═══════════════════════════════════════════════════════

    def _normalize_text(self, text: str) -> str:
        """Unicode NFC 规范化 + 全角转半角 + 零宽字符去除 + 换行归一化"""
        # P0-6: 换行归一化
        text = self._NEWLINE_NORMALIZER.sub("\n", text)
        # Unicode 规范化
        text = unicodedata.normalize("NFKC", text)
        # 全角→半角
        text = text.translate(self._FULLWIDTH_MAP)
        # 零宽字符去除
        text = self._ZERO_WIDTH_CHARS.sub("", text)
        return text

    # ═══════════════════════════════════════════════════════
    # P0-8: 全文一次 Token 编码
    # ═══════════════════════════════════════════════════════

    def _precompute_tokens(self, text: str, inp: DocumentChunkerInput) -> int:
        """全文编码一次，缓存 token_ids 供后续切片使用"""
        self._token_ids_cache = None
        if not inp.use_tokenizer:
            return 0
        enc = self._get_tokenizer(inp.encoding_name)
        if not enc or enc is False:
            return 0
        try:
            self._token_ids_cache = enc.encode(text)
            return len(self._token_ids_cache)
        except Exception as exc:
            logger.warning("Token 编码失败: %s", exc)
            return 0

    def _token_count_from_cache(self, start: int, end: int) -> int:
        """从缓存 token_ids 中估算某段的 token 数"""
        if self._token_ids_cache is None:
            return 0
        # 粗略估算：按字符比例映射 Token 数
        # （更精确的做法需要 token→char mapping，这里用比例近似）
        if end <= start:
            return 0
        return max(1, int(len(self._token_ids_cache) * (end - start) / max(end, 1)))

    # ═══════════════════════════════════════════════════════
    # 后端适配器
    # ═══════════════════════════════════════════════════════

    def _execute_via_backend(
        self, text: str, inp: DocumentChunkerInput, _total_tokens: int
    ) -> List[Chunk]:
        """通过外部后端分块"""
        if inp.backend == "langchain":
            return self._chunk_via_langchain(text, inp)
        elif inp.backend == "llama_index":
            return self._chunk_via_llama_index(text, inp)
        return self._execute_native(text, inp, _total_tokens)

    def _chunk_via_langchain(self, text: str, inp: DocumentChunkerInput) -> List[Chunk]:
        """LangChain TextSplitter 作为后端"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            logger.warning("langchain 未安装，降级到 native")
            return self._execute_native(text, inp, 0)

        separators = inp.separators or _DOCUMENT_TYPE_SEPARATORS.get(
            inp.document_type, _DOCUMENT_TYPE_SEPARATORS["text"]
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=inp.chunk_size,
            chunk_overlap=inp.chunk_overlap,
            separators=separators,
        )
        docs = splitter.create_documents([text])

        chunks: List[Chunk] = []
        search_from = 0
        for i, doc in enumerate(docs):
            content = doc.page_content
            start = text.find(content, search_from)
            if start == -1:
                start = search_from
            end = start + len(content)
            search_from = end + 1

            chunks.append(Chunk(
                index=i,
                text=content,
                char_count=len(content),
                byte_count=len(content.encode("utf-8")),
                token_count=self._token_count_from_cache(start, end),
                start_pos=start,
                end_pos=end,
                chunk_type="recursive",
                boundary_type="recursive_fallback",
                overlap_ratio=0.0,
                language=self._detect_language(content),
            ))

        logger.info("LangChain 后端完成 | chunks=%d", len(chunks))
        return chunks

    def _chunk_via_llama_index(self, text: str, inp: DocumentChunkerInput) -> List[Chunk]:
        """LLaMAIndex NodeParser 作为后端"""
        try:
            from llama_index.core.node_parser import SimpleNodeParser
        except ImportError:
            logger.warning("llama_index 未安装，降级到 native")
            return self._execute_native(text, inp, 0)

        parser = SimpleNodeParser.from_defaults(
            chunk_size=inp.chunk_size,
            chunk_overlap=inp.chunk_overlap,
        )
        nodes = parser.get_nodes_from_documents([text])  # type: ignore[arg-type]
        # 简化适配：extract text from node_impls
        # 实际 LLaMAIndex 的 get_nodes_from_documents 需要 Document 对象，
        # 这里做最简映射
        chunks: List[Chunk] = []
        search_from = 0
        for i, node in enumerate(nodes):
            content = node.text if hasattr(node, "text") else str(node)
            start = text.find(content, search_from)
            if start == -1:
                start = search_from
            end = start + len(content)
            search_from = end + 1

            chunks.append(Chunk(
                index=i,
                text=content,
                char_count=len(content),
                byte_count=len(content.encode("utf-8")),
                token_count=self._token_count_from_cache(start, end),
                start_pos=start,
                end_pos=end,
                chunk_type="recursive",
                boundary_type="recursive_fallback",
                overlap_ratio=0.0,
                language=self._detect_language(content),
            ))

        logger.info("LLaMAIndex 后端完成 | chunks=%d", len(chunks))
        return chunks

    # ═══════════════════════════════════════════════════════
    # Native 原生策略路由
    # ═══════════════════════════════════════════════════════

    def _execute_native(
        self, text: str, inp: DocumentChunkerInput, total_tokens: int
    ) -> List[Chunk]:
        if inp.strategy == "paragraph":
            return self._chunk_by_paragraph(text, inp)
        elif inp.strategy == "sentence":
            return self._chunk_by_sentence(text, inp)
        elif inp.strategy == "fixed":
            return self._chunk_fixed(text, inp, total_tokens)
        else:
            return self._chunk_recursive(text, inp, total_tokens)

    # ═══════════════════════════════════════════════════════
    # recursive 递归语义切分
    # ═══════════════════════════════════════════════════════

    def _chunk_recursive(
        self, text: str, inp: DocumentChunkerInput, total_tokens: int
    ) -> List[Chunk]:
        separators = inp.separators or self._get_separators_for_type(inp.document_type)
        return self._recursive_split(text, separators, inp.chunk_size,
                                     inp.chunk_overlap, inp)

    @staticmethod
    def _get_separators_for_type(doc_type: str) -> List[str]:
        if doc_type == "auto":
            return _DOCUMENT_TYPE_SEPARATORS["text"]
        return _DOCUMENT_TYPE_SEPARATORS.get(doc_type,
                                             _DOCUMENT_TYPE_SEPARATORS["text"])

    def _recursive_split(
        self,
        text: str,
        separators: list,
        chunk_size: int,
        chunk_overlap: int,
        inp: DocumentChunkerInput,
    ) -> List[Chunk]:
        if len(text) <= chunk_size:
            if not text.strip():
                return []
            return [self._make_chunk(text, inp, "recursive_fallback")]

        sep = separators[0] if separators else ""
        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        merged: list[str] = []
        buffer = ""
        for part in splits:
            candidate = buffer + (sep if buffer else "") + part
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                if len(part) > chunk_size and len(separators) > 1:
                    sub_chunks = self._recursive_split(
                        part, separators[1:], chunk_size, chunk_overlap, inp
                    )
                    merged.extend(c.text for c in sub_chunks)
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            merged.append(buffer)

        return self._build_chunks_with_overlap(merged, sep, chunk_overlap, inp, text)

    def _build_chunks_with_overlap(
        self,
        segments: List[str],
        separator: str,
        chunk_overlap: int,
        inp: DocumentChunkerInput,
        full_text: str = "",
    ) -> List[Chunk]:
        """P0-5: O(n) 指针追踪替代 find() 回溯"""
        chunks: List[Chunk] = []
        # 用 full_text 做全局指针推进
        pos = 0

        for i, seg in enumerate(segments):
            if not seg:
                pos += len(separator)
                continue

            start = pos
            end = pos + len(seg)

            overlap_ratio = 0.0
            if i > 0 and chunk_overlap > 0:
                # P1-6: overlap 对齐到完整句子边界
                aligned = self._align_to_sentence_boundary(
                    seg, chunk_overlap
                )
                overlap_ratio = aligned / max(len(seg), 1)

            boundary = (
                "paragraph_break" if separator in ("\n\n", "\r\n\r\n")
                else "sentence_end" if separator in ("。", "！", "？", ".", "!", "?")
                else "char_limit"
            )

            chunks.append(Chunk(
                index=i,
                text=seg,
                char_count=len(seg),
                byte_count=len(seg.encode("utf-8")),
                token_count=self._token_count_from_cache(start, end),
                start_pos=start,
                end_pos=end,
                chunk_type="recursive",
                boundary_type=boundary,  # type: ignore[arg-type]
                overlap_ratio=round(overlap_ratio, 3),
                language=self._detect_language(seg),
            ))

            pos = end + len(separator)

        return chunks

    # ═══════════════════════════════════════════════════════
    # paragraph 段落分块
    # ═══════════════════════════════════════════════════════

    def _chunk_by_paragraph(self, text: str, inp: DocumentChunkerInput) -> List[Chunk]:
        paragraphs = self._PARAGRAPH_SPLITTER.split(text)
        return self._segments_to_chunks(paragraphs, text, inp,
                                        "paragraph", "paragraph_break")

    # ═══════════════════════════════════════════════════════
    # P1-2: sentence 分句 — 三层降级引擎
    # ═══════════════════════════════════════════════════════

    def _chunk_by_sentence(self, text: str, inp: DocumentChunkerInput) -> List[Chunk]:
        sentences = self._split_sentences(text)
        return self._segments_to_chunks(sentences, text, inp,
                                        "sentence", "sentence_end")

    def _split_sentences(self, text: str) -> List[str]:
        """
        三层分句降级：
        1. sentence-splitter（轻量多语言，推荐）
        2. spaCy（英文精准分句）
        3. jieba 辅助 + 正则兜底（中文降级）
        """
        # Layer 1: sentence-splitter
        sentences = self._try_sentence_splitter(text)
        if sentences:
            return sentences

        # Layer 2: spaCy（英文）
        lang = self._detect_language(text)
        if lang == "en":
            sentences = self._try_spacy(text)
            if sentences:
                return sentences

        # Layer 3: jieba 辅助 + 正则兜底
        return self._fallback_sentence_split(text)

    def _try_sentence_splitter(self, text: str) -> Optional[List[str]]:
        """sentence-splitter 库分句"""
        if self._sent_splitter is False:
            return None
        if self._sent_splitter is not None:
            try:
                return self._sent_splitter.split(text=text)
            except Exception:
                return None
        try:
            from sentence_splitter import SentenceSplitter
            self._sent_splitter = SentenceSplitter(language="en")
            return self._sent_splitter.split(text=text)
        except ImportError:
            self._sent_splitter = False
            return None
        except Exception:
            self._sent_splitter = False
            return None

    @staticmethod
    def _try_spacy(text: str) -> Optional[List[str]]:
        """spaCy 分句"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return [sent.text for sent in nlp(text).sents]
        except (ImportError, OSError):
            return None

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """jieba 辅助 + 正则兜底"""
        lang = self._detect_language(text)
        if lang == "zh":
            try:
                import jieba
                sentences: List[str] = []
                buffer = ""
                for char in text:
                    buffer += char
                    if char in "。！？\n":
                        stripped = buffer.strip()
                        if stripped:
                            sentences.append(stripped)
                        buffer = ""
                if buffer.strip():
                    sentences.append(buffer.strip())
                return sentences or [text]
            except ImportError:
                pass

        # 正则兜底
        parts = self._SENTENCE_FALLBACK.split(text)
        return [p.strip() for p in parts if p.strip()]

    # ═══════════════════════════════════════════════════════
    # P1-4: 语言检测 — langid 优先
    # ═══════════════════════════════════════════════════════

    def _detect_language(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"

        # 快速 Unicode 范围
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
        total = len(text.replace(" ", "").replace("\n", ""))
        if total > 0 and cjk / total > 0.3:
            return "zh"

        # langid（准确率高于 langdetect，短文本友好）
        if self._lang_detector is None:
            self._lang_detector = self._init_lang_detector()
        if self._lang_detector and self._lang_detector is not False:
            try:
                return self._lang_detector.classify(text)[0]
            except Exception:
                pass

        # 纯 ASCII 兜底
        if all(ord(ch) < 128 for ch in text if ch.isalpha()):
            return "en"
        return "unknown"

    @staticmethod
    def _init_lang_detector():
        """初始化 langid"""
        try:
            import langid
            langid.set_languages(["en", "zh", "ja", "ko", "fr", "de", "es"])
            return langid
        except ImportError:
            return False

    # ═══════════════════════════════════════════════════════
    # fixed 固定窗口分块
    # ═══════════════════════════════════════════════════════

    def _chunk_fixed(
        self, text: str, inp: DocumentChunkerInput, total_tokens: int
    ) -> List[Chunk]:
        chunk_size = inp.chunk_size
        chunk_overlap = inp.chunk_overlap
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap({chunk_overlap}) 必须 < chunk_size({chunk_size})")

        step = chunk_size - chunk_overlap

        # Token 级优先
        enc = self._get_tokenizer(inp.encoding_name) if inp.use_tokenizer else None
        if enc and enc is not False and self._token_ids_cache and total_tokens > chunk_size // 2:
            return self._chunk_by_tokens_with_cache(text, enc, chunk_size, step, inp)

        return self._chunk_by_chars(text, chunk_size, step, inp)

    def _chunk_by_tokens_with_cache(
        self, text: str, enc, chunk_size: int, step: int, inp: DocumentChunkerInput
    ) -> List[Chunk]:
        """Token 级切分（利用预编码缓存）"""
        token_ids = self._token_ids_cache
        if not token_ids:
            return self._chunk_by_chars(text, chunk_size, step, inp)

        chunks: List[Chunk] = []
        pos = 0  # 字符级指针
        tok_offset = 0

        while tok_offset < len(token_ids):
            end_tok = min(tok_offset + chunk_size, len(token_ids))
            segment_ids = token_ids[tok_offset:end_tok]
            segment_text = enc.decode(segment_ids)

            start_pos = pos
            end_pos = pos + len(segment_text)

            overlap_ratio = 0.0
            if chunks and step < chunk_size:
                aligned = self._align_to_sentence_boundary(segment_text, chunk_size - step)
                overlap_ratio = aligned / max(len(segment_text), 1)

            chunks.append(Chunk(
                index=len(chunks),
                text=segment_text,
                char_count=len(segment_text),
                byte_count=len(segment_text.encode("utf-8")),
                token_count=len(segment_ids),
                start_pos=start_pos,
                end_pos=end_pos,
                chunk_type="fixed",
                boundary_type="token_limit",
                overlap_ratio=round(overlap_ratio, 3),
                language=self._detect_language(segment_text),
            ))

            if end_tok >= len(token_ids):
                break
            tok_offset += step
            pos = end_pos

        return chunks

    def _chunk_by_chars(
        self, text: str, chunk_size: int, step: int, inp: DocumentChunkerInput
    ) -> List[Chunk]:
        """字符级固定窗口（降级路径）"""
        chunks: List[Chunk] = []
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            segment = text[pos:end]

            overlap_ratio = 0.0
            if chunks and step < chunk_size:
                aligned = self._align_to_sentence_boundary(segment, chunk_size - step)
                overlap_ratio = aligned / max(len(segment), 1)

            chunks.append(Chunk(
                index=len(chunks),
                text=segment,
                char_count=len(segment),
                byte_count=len(segment.encode("utf-8")),
                token_count=self._token_count_from_cache(pos, end),
                start_pos=pos,
                end_pos=end,
                chunk_type="fixed",
                boundary_type="char_limit",
                overlap_ratio=round(overlap_ratio, 3),
                language=self._detect_language(segment),
            ))

            if end >= len(text):
                break
            pos += step

        return chunks

    # ═══════════════════════════════════════════════════════
    # P1-6: overlap 对齐到完整句子边界
    # ═══════════════════════════════════════════════════════

    def _align_to_sentence_boundary(self, text: str, desired_overlap: int) -> int:
        """
        在 desired_overlap 附近寻找最近的完整句子边界。
        优先用分句引擎，降级用标点+换行搜索。
        """
        if desired_overlap <= 0 or desired_overlap >= len(text):
            return 0

        # 尝试用分句引擎找精确边界
        sentences = self._try_sentence_splitter(text)
        if sentences:
            cumulative = 0
            for sent in sentences:
                cumulative += len(sent)
                if cumulative >= desired_overlap:
                    return min(cumulative, len(text))
            return desired_overlap

        # 降级：标点 + 换行搜索（±30% 窗口）
        search_window = max(10, int(desired_overlap * 0.3))
        sentence_ends = set("。！？.!?\n")

        for offset in range(0, search_window):
            pos = desired_overlap + offset
            if pos < len(text) and text[pos] in sentence_ends:
                return pos + 1  # 包含标点本身
            pos = desired_overlap - offset
            if pos > 0 and text[pos] in sentence_ends:
                return pos + 1

        return desired_overlap

    # ═══════════════════════════════════════════════════════
    # P1-7: 语义感知小块合并
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _merge_small_chunks_semantic(
        chunks: List[Chunk], min_size: int, full_text: str = ""
    ) -> List[Chunk]:
        """
        遍历分块，将低于 min_size 的小块合并到语义上更相关的相邻块。
        判断依据：小块与前后块的边界类型是否一致（同段落/同句子切出）。
        """
        if len(chunks) < 2:
            return chunks

        result: List[Chunk] = []

        for chunk in chunks:
            if chunk.char_count >= min_size:
                result.append(chunk)
                continue

            # 小块 → 判断合并方向
            if not result:
                # 第一个块就是小块，暂存等下一块
                if chunk.text.strip():
                    result.append(chunk)
                continue

            prev = result[-1]

            # P1-7: 语义归属判断
            # 如果前一块的 boundary_type 相同（同源切分），向前合并更自然
            # 如果 boundary_type 不同（跨语义单元），向后合并（给下一个块处理）
            merge_forward = (
                prev.boundary_type == chunk.boundary_type
                or prev.chunk_type == chunk.chunk_type
            )

            if merge_forward:
                merged_text = prev.text + "\n" + chunk.text
                result[-1] = Chunk(
                    index=prev.index,
                    text=merged_text,
                    char_count=len(merged_text),
                    byte_count=len(merged_text.encode("utf-8")),
                    token_count=prev.token_count + chunk.token_count,
                    start_pos=prev.start_pos,
                    end_pos=chunk.end_pos,
                    chunk_type=prev.chunk_type,
                    boundary_type="recursive_fallback",
                    overlap_ratio=prev.overlap_ratio,
                    language=prev.language,
                )
            elif chunk.text.strip():
                # 不向前合并，保留等待向后合并
                result.append(chunk)

        # 尾部小块二次处理：若最后一块太小且前面有大块
        if len(result) >= 2 and result[-1].char_count < min_size:
            last = result.pop()
            prev = result[-1]
            merged_text = prev.text + "\n" + last.text
            result[-1] = Chunk(
                index=prev.index,
                text=merged_text,
                char_count=len(merged_text),
                byte_count=len(merged_text.encode("utf-8")),
                token_count=prev.token_count + last.token_count,
                start_pos=prev.start_pos,
                end_pos=last.end_pos,
                chunk_type=prev.chunk_type,
                boundary_type="recursive_fallback",
                overlap_ratio=prev.overlap_ratio,
                language=prev.language,
            )

        return result

    # ═══════════════════════════════════════════════════════
    # P1-14: 上下文窗口附载
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _attach_context_windows(chunks: List[Chunk]) -> None:
        """为每个 Chunk 注入前后块的上下文首/末句"""
        if len(chunks) < 2:
            return

        for i, chunk in enumerate(chunks):
            if i > 0:
                prev = chunks[i - 1]
                chunk.context_prefix = DocumentChunkerSkill._extract_last_sentence(
                    prev.text
                )
            if i < len(chunks) - 1:
                nxt = chunks[i + 1]
                chunk.context_suffix = DocumentChunkerSkill._extract_first_sentence(
                    nxt.text
                )

    @staticmethod
    def _extract_last_sentence(text: str) -> Optional[str]:
        """提取文本最后一句话（最多 200 字符）"""
        if not text:
            return None
        # 从尾部找最近的句尾标点
        for i in range(len(text) - 1, max(len(text) - 300, -1), -1):
            if text[i] in "。！？.!?\n":
                return text[i + 1:].strip()[:200] or None
        return text[-200:].strip() or None

    @staticmethod
    def _extract_first_sentence(text: str) -> Optional[str]:
        """提取文本第一句话（最多 200 字符）"""
        if not text:
            return None
        for i, ch in enumerate(text[:300]):
            if ch in "。！？.!?\n":
                return text[:i + 1].strip()[:200] or None
        return text[:200].strip() or None

    # ═══════════════════════════════════════════════════════
    # P2-15: 去重标记
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _mark_near_duplicates(chunks: List[Chunk], threshold: float = 0.5) -> None:
        """标记高重叠率的相邻块"""
        for i in range(1, len(chunks)):
            if chunks[i].overlap_ratio >= threshold:
                chunks[i].is_near_duplicate_of = chunks[i - 1].index

    # ═══════════════════════════════════════════════════════
    # P0-5: 指针推进构建 Chunk（段落/句子策略）
    # ═══════════════════════════════════════════════════════

    def _segments_to_chunks(
        self,
        segments: List[str],
        full_text: str,
        inp: DocumentChunkerInput,
        chunk_type: Literal["paragraph", "sentence"],
        boundary_type: Literal["paragraph_break", "sentence_end"],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        search_from = 0

        for seg in segments:
            stripped = seg.strip()
            if not stripped:
                search_from += len(seg) + 2
                continue

            start = full_text.find(stripped, search_from)
            if start == -1:
                start = search_from
            end = start + len(stripped)
            search_from = end

            chunks.append(Chunk(
                index=len(chunks),
                text=stripped,
                char_count=len(stripped),
                byte_count=len(stripped.encode("utf-8")),
                token_count=self._token_count_from_cache(start, end),
                start_pos=start,
                end_pos=end,
                chunk_type=chunk_type,
                boundary_type=boundary_type,
                overlap_ratio=0.0,
                language=self._detect_language(stripped),
            ))

        return chunks

    # ═══════════════════════════════════════════════════════
    # Tokenizer 工具
    # ═══════════════════════════════════════════════════════

    def _get_tokenizer(self, encoding_name: str):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(encoding_name)
            logger.info("tiktoken 分词器已加载 | encoding=%s", encoding_name)
        except Exception as exc:
            logger.warning("tiktoken 加载失败: %s", exc)
            self._tokenizer = False
        return self._tokenizer

    # ═══════════════════════════════════════════════════════
    # 输出构建 & 统计
    # ═══════════════════════════════════════════════════════

    def _build_output(
        self,
        chunks: List[Chunk],
        inp: DocumentChunkerInput,
        t_start: float,
        text: str,
    ) -> dict:
        # P2-16: 分块大小分布统计
        sizes = [c.char_count for c in chunks] if chunks else [0]
        sorted_sizes = sorted(sizes)
        p95_idx = int(len(sorted_sizes) * 0.95)
        stats: _ChunkStats = _ChunkStats(
            min=sorted_sizes[0],
            max=sorted_sizes[-1],
            avg=round(mean(sizes), 1),
            median=round(median(sizes), 1),
            p95=float(sorted_sizes[min(p95_idx, len(sorted_sizes) - 1)]),
            total=len(chunks),
        )

        # P2-18: 幂等 chunk_set_id
        raw = (text[:2000] + inp.strategy + inp.document_type +
               str(inp.chunk_size) + str(inp.chunk_overlap))
        chunk_set_id = hashlib.md5(raw.encode()).hexdigest()[:12]

        return DocumentChunkerOutput(
            success=True,
            chunks=chunks,
            total_chunks=len(chunks),
            strategy=inp.strategy,
            source_char_count=len(text),
            source_token_count=(
                len(self._token_ids_cache) if self._token_ids_cache else 0
            ),
            chunk_size_stats=stats,
            chunk_set_id=chunk_set_id,
            elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
        ).model_dump()

    # ═══════════════════════════════════════════════════════
    # 快捷工厂方法
    # ═══════════════════════════════════════════════════════

    def _make_single_chunk(self, text: str, inp: DocumentChunkerInput) -> Chunk:
        return Chunk(
            index=0,
            text=text,
            char_count=len(text),
            byte_count=len(text.encode("utf-8")),
            token_count=self._token_count_from_cache(0, len(text)),
            start_pos=0,
            end_pos=len(text),
            chunk_type=inp.strategy,  # type: ignore[arg-type]
            boundary_type="recursive_fallback",
            overlap_ratio=0.0,
            language=self._detect_language(text),
        )

    def _make_chunk(
        self,
        text: str,
        inp: DocumentChunkerInput,
        boundary_type: str = "recursive_fallback",
    ) -> Chunk:
        return Chunk(
            index=0,
            text=text,
            char_count=len(text),
            byte_count=len(text.encode("utf-8")),
            token_count=self._token_count_from_cache(0, len(text)),
            start_pos=0,
            end_pos=len(text),
            chunk_type="recursive",
            boundary_type=boundary_type,  # type: ignore[arg-type]
            overlap_ratio=0.0,
            language=self._detect_language(text),
        )


# ═══════════════════════════════════════════════════════════════
# 便捷函数（供外部直接调用）
# ═══════════════════════════════════════════════════════════════

def chunk_document(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs,
) -> dict:
    """一行式分块便捷接口"""
    skill = DocumentChunkerSkill()
    input_data = DocumentChunkerInput(
        text=text,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    return skill.execute(input_data)


def chunk_documents_batch(
    texts: List[str],
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs,
) -> List[dict]:
    """批量分块便捷接口"""
    skill = DocumentChunkerSkill()
    return skill.execute_batch(
        texts=texts,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )