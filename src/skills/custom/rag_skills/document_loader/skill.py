"""
文档加载技能 — RAG 体系第一环
支持 txt / md / json / csv / py / js / html / xml / yaml / log
自动检测编码 + 文件类型，返回结构化元数据

┌─────────────────────────────────────────────────────────────┐
│  1. 常量层                                                    │
│     - _SUPPORTED_EXTENSIONS（扩展名→文件类型映射）            │
│     - _ENCODING_GUESS_ORDER（编码探测顺序）                  │
│     - _MAX_CHARS_DEFAULT（默认最大字符数）                   │
├─────────────────────────────────────────────────────────────┤
│  2. 数据模型层（Pydantic）                                    │
│     - DocumentLoaderInput（输入校验）                         │
│     - DocumentLoaderOutput（结构化输出）                       │
├─────────────────────────────────────────────────────────────┤
│  3. 主技能类（DocumentLoaderSkill）                           │
│     - execute（主入口，异常捕获）                              │
│     - _execute_impl（核心逻辑，7步加载流程）                  │
│     - 辅助方法：路径/类型/编码/读取/预处理                    │
└─────────────────────────────────────────────────────────────┘

DocumentLoaderSkill.execute()
  [主入口，异常捕获] 捕获所有异常，保证返回结构化结果
  └→ _execute_impl(input_data)
       [核心7步加载流程] 路径校验→类型判定→编码探测→读取→去BOM→空字节检测→元数据组装
       ├→ _validate_and_resolve_path(file_path_str)
       │   [路径校验] 解析路径，校验存在性/是否为文件，返回Path或错误
       ├→ _resolve_file_type(requested, path)
       │   [文件类型判定] auto则从后缀推断，否则用用户指定值
       ├→ _resolve_encoding(requested, path)
       │   [编码探测] auto则按顺序尝试编码，否则用用户指定值
       ├→ _read_file(path, encoding, max_chars)
       │   [读取文件] 预检查大小→读取文本→解码错误回退→后检查大小
       ├→ _strip_bom(content)
       │   [去除BOM] 去除Windows记事本常带的UTF-8 BOM头
       └→ 空字节检测+元数据组装
           [安全检查+结果组装] 检测空字节（防二进制文件），计算字符数/行数/文件大小
"""

import logging
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 作用：扩展名（小写）→文件类型的映射，支持 17 种格式
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt":  "plaintext",
    ".md":   "markdown",
    ".json": "json",
    ".csv":  "csv",
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".java": "java",
    ".go":   "go",
    ".cpp":  "cpp",
    ".c":    "c",
    ".html": "html",
    ".htm":  "html",
    ".xml":  "xml",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".log":  "log",
    ".ini":  "ini",
    ".cfg":  "ini",
    ".toml": "toml",
}

# 作用：编码探测的顺序，按真实世界流行度（中文 + 英文场景）排列。
_ENCODING_GUESS_ORDER = ["utf-8", "gbk", "gb2312", "latin-1", "cp1252"]
#作用：默认最大字符数（500,000），避免加载过大文件导致 OOM
_MAX_CHARS_DEFAULT = 500_000


# ---------------------------------------------------------------------------
# 作用：输入校验模型，保证输入格式合法。
# ---------------------------------------------------------------------------

class DocumentLoaderInput(BaseModel):
    file_path: str = Field(
        ...,
        description="文档的绝对路径或相对路径",
        examples=["data/sample.txt", "/home/user/docs/report.md"],
    )
    file_type: str = Field(
        default="auto",
        description="强制指定文件类型，auto 表示从后缀自动推断",
        examples=["auto", "txt", "md", "json"],
    )
    encoding: str = Field(
        default="auto",
        description="文件编码，auto 表示自动探测（依次尝试 utf-8/gbk/latin-1）",
        examples=["auto", "utf-8", "gbk"],
    )
    max_chars: int = Field(
        default=_MAX_CHARS_DEFAULT,
        description="单次读取最大字符数，超出抛出异常",
        ge=1,
        le=5_000_000,
    )

# ---------------------------------------------------------------------------
# 作用：结构化输出模型，返回文本内容及元数据。。
# ---------------------------------------------------------------------------

class DocumentLoaderOutput(BaseModel):
    content: str = Field(default="", description="文档完整文本内容（已剥离 BOM）")
    char_count: int = Field(default=0, description="字符总数")
    line_count: int = Field(default=0, description="行数")
    file_type: str = Field(default="", description="检测到的文件类型")
    encoding_used: str = Field(default="", description="实际使用的编码")
    file_size_bytes: int = Field(default=0, description="文件大小（字节）")
    extension: str = Field(default="", description="文件后缀（小写）")
    error: str = Field(default="", description="若失败，存放错误信息")


# ---------------------------------------------------------------------------
# 这是技能的核心，负责协调整个文档加载流程。
# ---------------------------------------------------------------------------

class DocumentLoaderSkill(BaseSkill):
    """加载本地文档，自动检测类型 + 编码，供下游分块/嵌入使用"""

    name: ClassVar[str] = "document_loader"
    description: ClassVar[str] = (
        "加载本地文档（txt/md/json/csv/py/js/html/xml/yaml/log 等），"
        "自动检测文件类型与编码，返回全文及元数据"
    )
    triggers: ClassVar[list[str]] = [
        "加载文档", "读取文件", "load document", "read file",
        "打开文件", "导入文档", "文档加载",
    ]
    version: ClassVar[str] = "0.1.0"
    author: ClassVar[str] = "EnterpriseLearningAgent"
    changelog: ClassVar[str] = "v0.1.0 — 初始版本：支持 17 种文件类型 + 编码自动探测"

    # ------------------------------------------------------------------
    # Schemas
    # ------------------------------------------------------------------

    @property
    def input_schema(self) -> type:
        return DocumentLoaderInput

    @property
    def output_schema(self) -> type:
        return DocumentLoaderOutput

    # ------------------------------------------------------------------
    # 作用：主入口，捕获所有异常，保证返回结构化结果（即使失败）。
    # ------------------------------------------------------------------

    def execute(self, input_data: DocumentLoaderInput) -> dict:
        """
        加载文档并返回结构化结果。
        步骤：路径校验 → 文件类型判定 → 编码探测 → 读取 → 元数据组装
        """
        try:
            return self._execute_impl(input_data).model_dump() # <--- 在这里加上 .model_dump()
        except MemoryError:
            logger.exception("内存不足，无法加载文件")
            return DocumentLoaderOutput(error="内存不足，文件过大无法加载").model_dump()
        except RecursionError:
            logger.exception("路径解析递归溢出（可能存在符号链接循环）")
            return DocumentLoaderOutput(error="文件路径解析失败：可能存在符号链接循环").model_dump()
        except Exception as exc:
            logger.exception("文档加载时发生未预期异常: %s", exc)
            return DocumentLoaderOutput(error=f"文档加载失败：{exc}").model_dump()

    #作用：核心加载流程，按序执行 7 个步骤。
    def _execute_impl(self, input_data: DocumentLoaderInput) -> DocumentLoaderOutput:
        file_path = self._validate_and_resolve_path(input_data.file_path)
        if isinstance(file_path, DocumentLoaderOutput):  # path validation failed
            return file_path

        logger.info("开始加载文档: %s (type=%s, encoding=%s)",
                     file_path, input_data.file_type, input_data.encoding)

        # --- step 1: 路径校验：调用_validate_and_resolve_path()，返回Path或错误Output ---
        suffix = file_path.suffix.lower()
        logger.debug("文件后缀: %s", suffix)
        if suffix not in _SUPPORTED_EXTENSIONS:
            logger.warning("不支持的文件类型: %s", suffix)
            return DocumentLoaderOutput(
                error=f"不支持的文件类型 '{suffix}'，"
                      f"当前支持：{', '.join(_SUPPORTED_EXTENSIONS)}"
            )

        # --- step 2: 文件类型判定：调用_resolve_file_type()，从后缀推断或使用用户指定值 ---
        file_type = self._resolve_file_type(input_data.file_type, file_path)
        logger.debug("文件类型判定: %s", file_type)

        # --- step 3: 调用_resolve_encoding()，按顺序尝试编码或使用用户指定值 ---
        encoding = self._resolve_encoding(input_data.encoding, file_path)
        logger.debug("编码判定: %s", encoding)

        # --- step 4: 调用_read_file()，读取全文，处理大小限制和解码错误 ---
        content, encoding_used, error = self._read_file(file_path, encoding, input_data.max_chars)
        if error:
            logger.error("读取文件失败: %s — %s", file_path, error)
            return DocumentLoaderOutput(error=error)
        logger.info("读取成功: %d 字符, 编码=%s", len(content), encoding_used)

        # --- step 5: 去除 BOM：调用_strip_bom()，去除 UTF-8 BOM 头 ---
        content = self._strip_bom(content)

        # --- step 6: 空字节检测：若内容含\x00，返回错误（可能是二进制文件） ---
        if "\x00" in content:
            logger.warning("文件中包含空字节，可能是二进制文件: %s", file_path)
            return DocumentLoaderOutput(
                error=f"文件包含空字节，可能是二进制文件而非文本文件：{file_path}"
            )

        # --- step 7: 元数据组装：计算字符数、行数、文件大小，返回完整Output ---
        file_size = file_path.stat().st_size
        char_count = len(content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        logger.debug("元数据: chars=%d, lines=%d, size=%d bytes", char_count, line_count, file_size)

        return DocumentLoaderOutput(
            content=content,
            char_count=char_count,
            line_count=line_count,
            file_type=file_type,
            encoding_used=encoding_used,
            file_size_bytes=file_size,
            extension=suffix,
        )

    # ------------------------------------------------------------------
    # 作用：校验并解析路径，返回Path对象或错误Output
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_resolve_path(file_path_str: str) -> Path | DocumentLoaderOutput:
        """校验并解析路径，返回 Path 或错误 Output"""
        try:
            file_path = Path(file_path_str).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            logger.error("路径解析失败: %s — %s", file_path_str, exc)
            return DocumentLoaderOutput(error=f"路径解析失败：{exc}")

        if not file_path.exists():
            logger.warning("文件不存在: %s", file_path)
            return DocumentLoaderOutput(error=f"文件不存在：{file_path}")

        if file_path.is_dir():
            logger.warning("路径是目录而非文件: %s", file_path)
            return DocumentLoaderOutput(error=f"路径是目录而非文件：{file_path}")

        if not file_path.is_file():
            logger.warning("路径不是常规文件: %s", file_path)
            return DocumentLoaderOutput(error=f"路径不是常规文件：{file_path}")

        return file_path

    # ------------------------------------------------------------------
    # 作用：解析文件类型，auto则从后缀推断，否则直接用用户指定值
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_file_type(requested: str, path: Path) -> str:
        """解析文件类型：auto → 从后缀推断，否则直接用给定值"""
        if requested != "auto":
            logger.debug("使用用户指定 file_type: %s", requested)
            return requested
        detected = _SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown")
        logger.debug("自动推断 file_type: %s → %s", path.suffix, detected)
        return detected

    # ------------------------------------------------------------------
    # 作用：解析编码，auto则按顺序尝试探测，否则直接用用户指定值
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_encoding(requested: str, path: Path) -> str:
        """解析编码：auto → 依次尝试 utf-8 / gbk / latin-1"""
        if requested != "auto":
            logger.debug("使用用户指定 encoding: %s", requested)
            return requested

        raw_bytes = path.read_bytes()
        for enc in _ENCODING_GUESS_ORDER:
            try:
                raw_bytes.decode(enc)
                logger.debug("编码探测成功: %s", enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        logger.warning("所有编码探测失败，回退到 utf-8")
        return "utf-8"  # last-resort fallback

    # ------------------------------------------------------------------
    # 作用：读取文件全文，处理大小限制、解码错误和系统错误。
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path, encoding: str, max_chars: int) -> tuple[str, str, str]:
        """
        读取文件全文。
        Returns: (content, encoding_actually_used, error)
        """
        file_size = path.stat().st_size
        logger.debug("文件大小: %d bytes, 上限: %d chars", file_size, max_chars)

        # Rough size guard before reading — avoids OOM on obviously huge files
        # 4 bytes per char is the worst case (all emoji / CJK extension B+)
        if file_size > max_chars * 4:
            return "", encoding, f"文件过大（{file_size} bytes），超过上限 {max_chars} 字符"

        try:
            content = path.read_text(encoding=encoding)
            logger.debug("读取成功: %d 字符, 编码=%s", len(content), encoding)
        except FileNotFoundError:
            return "", encoding, f"文件未找到：{path}"
        except PermissionError:
            return "", encoding, f"没有读取权限：{path}"
        except IsADirectoryError:
            return "", encoding, f"路径是目录而非文件：{path}"
        except UnicodeDecodeError:
            # Fallback: try latin-1 which accepts any byte sequence
            try:
                content = path.read_text(encoding="latin-1")
                encoding = "latin-1 (fallback)"
                logger.warning("utf-8 解码失败，回退到 latin-1")
            except Exception as exc:
                return "", encoding, f"解码失败（已尝试 {encoding} 及 latin-1 回退）：{exc}"
        except OSError as exc:
            return "", encoding, f"读取文件时系统错误：{exc}"
        except Exception as exc:
            return "", encoding, f"读取文件时出错：{exc}"

        # Post-read char limit check (accurate, not estimated)
        if len(content) > max_chars:
            return "", encoding, f"文件内容 {len(content)} 字符，超过上限 {max_chars}"

        return content, encoding, ""

    # ------------------------------------------------------------------
    # Content sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_bom(content: str) -> str:
        """去除 UTF-8 BOM 头（Windows 记事本常带）"""
        if content.startswith("\ufeff"):
            logger.debug("检测到 UTF-8 BOM，已剥离")
            return content[1:]
        return content