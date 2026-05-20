"""
Code Explainer Skill — 代码解释（优化版 v2.0.0）
一、 整体功能概览
    这个模块的核心目的是 **“把晦涩的代码翻译成自然语言”**。它的主要能力包括：
    自动语言检测：支持 Python/JS/TS/Java/Go/C++/Rust/Kotlin/Ruby。
    智能代码分块：Python 用 AST 精准拆分，其他语言用正则，确保行号准确。
多层级解释：
    brief：只给总结。
    detailed：按函数 / 类逐块讲解。
    line_by_line：逐行注释。
深度代码分析：
    检测潜在 Bug（如 Python 的裸 except、文件资源泄露）。
    计算圈复杂度（代码复杂度指标）。
工程化容错：依赖库可选装、异常有兜底、API 向后兼容。

v2.0.0 优化改动:
  - P0-1: detail_level 参数真正生效 (brief/detailed/line_by_line)
  - P0-2: Python 用 ast 精准分块，行号准确；其他语言改进正则
  - P0-3: _infer_purpose 从互斥改为特征累积，信息不丢失
  - P1-4: _explain_how 用 ast 做结构分析，替代纯关键词 grep
  - P1-5: "✅ 未发现明显问题" 仅出现在单块级，不污染顶层汇总
  - P1-6: 硬截断加省略号判断，避免多余 "..."
  - P1-7: 删除虚假"代码结构清晰"评价，用 radon 给出真实复杂度
  - P1-8: execute() 开头 split 一次，消除三次重复 split
  - P1-9: execute() 加 try/except，优雅降级
  - P1-10: 清理 Output 冗余字段（detected_language/explanation/summary→computed_field）
  - P2-11: radon 可选依赖，分析 Python 圈复杂度
  - P2-12: ast 替代正则做裸 except / 资源泄露检测
  - P2-13: 增强语言检测特征，加入 TS/Rust/Kotlin/Ruby
  - P2-14: 装饰器/生成器/上下文管理器识别
  - P3-15: 添加 __repr__
  - P3-16: _LANG_PATTERNS 扩展至 9 种语言
  - P3-17: Literal → Enum (Language, DetailLevel)
  - P3-18: guesslang 可选依赖，提升语言检测准确率
  - +: execute 开头一次性 code.split("\n")，传入各方法
  - +: TypedDict RawCodeBlock 替代裸 tuple
  - +: 异常处理包裹主流程
  - +: 便捷函数 generate_code_explanation

  用户调用
└─> generate_code_explanation（便捷入口）
    └─> execute（类对外唯一入口，异常兜底）
        └─> _execute_inner（核心调度器，串联全流程）
            ├─> _detect_language（语言检测）
            ├─> _split_code_blocks（代码分块入口）
            │   ├─> _split_via_ast（Python精准分块）
            │   └─> _split_via_regex（通用正则分块兜底）
            ├─> _generate_overview（生成全局概述）
            ├─> 按详细程度选择解释分支
            │   ├─> _explain_blocks（逐块解释，默认模式）
            │   │   ├─> _infer_purpose（推断代码块用途）
            │   │   │   └─> _infer_purpose_via_ast（Python AST深度推断）
            │   │   ├─> _explain_how（解释代码块工作原理）
            │   │   │   ├─> _explain_how_via_ast（Python AST原理分析）
            │   │   │   └─> _explain_how_via_keywords（通用关键词分析兜底）
            │   │   └─> _identify_issues（检测代码块潜在问题）
            │   │       └─> _identify_issues_via_ast（Python AST问题检测）
            │   └─> _explain_line_by_line（逐行解释模式）
            │       └─> _infer_line_purpose（单行代码用途推断）
            ├─> _extract_code_key_points（提取核心要点）
            └─> 结果组装返回
"""

from __future__ import annotations

import ast
import logging
import re
from enum import StrEnum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field, computed_field, field_validator

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ============================================================================
# Optional Dependencies
# ============================================================================
_GUESSLANG_AVAILABLE = False
_GUESS = None
try:
    from guesslang import Guess
    _GUESS = Guess()
    _GUESSLANG_AVAILABLE = True
except ImportError:
    logger.info("guesslang 未安装，使用内置正则检测。pip install guesslang 可提高语言检测准确率。")

_RADON_AVAILABLE = False
try:
    from radon.complexity import cc_visit
    _RADON_AVAILABLE = True
except ImportError:
    logger.info("radon 未安装，无法计算圈复杂度。pip install radon 可获得更专业的复杂度分析。")


# ============================================================================
# 核心作用：把魔法值、固定配置抽成常量，方便维护、修改、复用，避免硬编码
# ============================================================================
_SKILL_VERSION = "2.0.0"
_MAX_CODE_LENGTH = 50000  # 最大代码字符数

# guesslang → 内部语言标识映射
_GUESSLANG_MAPPING: dict[str, str] = {
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "Java": "java",
    "Go": "go",
    "C++": "cpp",
    "Rust": "rust",
    "Kotlin": "kotlin",
    "Ruby": "ruby",
}


# ============================================================================
# Enums 核心作用：用枚举替代字符串字面量，实现「类型安全 + 自解释 + 防错」，是 v2.0.0 的核心优化点之一（P3-17）。
# ============================================================================
class Language(StrEnum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    CPP = "cpp"
    RUST = "rust"
    KOTLIN = "kotlin"
    RUBY = "ruby"
    AUTO = "auto"


class DetailLevel(StrEnum):
    """解释详细程度"""
    BRIEF = "brief"            # 仅概述 + 关键要点
    DETAILED = "detailed"      # 逐块解释
    LINE_BY_LINE = "line_by_line"  # 逐行解释



# ============================================================================
# TypedDict 核心作用：定义代码分块后的中间数据结构，替代之前的裸 tuple / 无结构字典，实现类型安全，
# 是 v2.0.0 的优化点之一。
# ============================================================================
class RawCodeBlock(TypedDict):
    """原始代码块中间表示"""
    name: str
    code: str
    start_line: int
    end_line: int


# ============================================================================
# 核心作用：语言检测的兜底方案，当 guesslang 不可用 / 检测失败时，用正则匹配代码中的关键字，给每种语言打分，
# 最终判定语言类型，是 v2.0.0 扩展到 9 种语言的核心支撑（P3-16）
# ============================================================================
_LANG_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^\s*def\s", r"^\s*import\s", r"^\s*from\s+\S+\s+import",
        r"print\(", r"^\s*class\s+\w+.*:", r"self\.", r"__\w+__",
        r"^\s*@\w+", r"if\s+__name__\s*==\s*['\"]__main__",
        r"^\s*async\s+def\s", r"^\s*with\s+\w+.*:",
    ],
    "javascript": [
        r"^\s*function\s", r"const\s+\w+\s*=", r"let\s+\w+\s*=",
        r"console\.log", r"=>\s*\{", r"^\s*import\s+\{",
        r"export\s+(default\s+)?", r"\.then\(", r"^\s*var\s+\w+",
        r"document\.", r"window\.", r"addEventListener",
    ],
    "typescript": [
        r":\s*(string|number|boolean|void|any)\b", r"interface\s+\w+\s*\{",
        r"type\s+\w+\s*=", r"^\s*import\s+\{.*\}\s*from",
        r"as\s+const", r"readonly\s+\w+", r"enum\s+\w+\s*\{",
        r"React\.FC", r":\s*JSX\.Element",
    ],
    "java": [
        r"public\s+(static\s+)?(void|class|int|String|boolean)",
        r"System\.out\.print", r"^\s*@Override", r"private\s+\w+\s+\w+;",
        r"new\s+\w+\(", r"^\s*package\s+\w+", r"^\s*import\s+java\.",
        r"ArrayList<", r"HashMap<",
    ],
    "go": [
        r"^\s*func\s", r"fmt\.Print", r"package\s+\w+", r"err\s*:=\s*",
        r"defer\s", r"go\s+func", r":=\s+", r"\[\]byte", r"interface\{\}",
    ],
    "cpp": [
        r"#include\s*<", r"std::", r"int\s+main\s*\(", r"cout\s*<<",
        r"^\s*template\s*<", r"cin\s*>>", r"#define\s+", r"nullptr",
        r"std::vector", r"std::string",
    ],
    "rust": [
        r"^\s*fn\s+\w+", r"let\s+mut\s+", r"^\s*impl\s+\w+",
        r"^\s*trait\s+\w+", r"^\s*struct\s+\w+", r"^\s*enum\s+\w+\s*\{",
        r"println!\(", r"vec!\[", r"Option<", r"Result<",
        r"^\s*pub\s+(fn|struct|enum|trait|mod)",
    ],
    "kotlin": [
        r"^\s*fun\s+\w+", r"^\s*val\s+\w+", r"^\s*var\s+\w+",
        r"^\s*object\s+\w+", r"^\s*data\s+class", r"^\s*sealed\s+",
        r"println\(", r"\?\?", r"\?\.", r"!!",
    ],
    "ruby": [
        r"^\s*def\s+\w+", r"^\s*class\s+\w+\s*<", r"^\s*module\s+\w+",
        r"^\s*require\s+['\"]", r"puts\s", r"\.each\s+do\b",
        r"^\s*attr_", r"#\{", r"@\w+\s*=", r"end\s*$",
    ],
}


# ============================================================================
# 这部分是模块的「输入输出契约」，定义了用户能传什么、模块会返回什么，自动完成参数校验
# ============================================================================
class CodeExplainerInput(BaseModel):
    """代码解释输入参数"""
    code: str = Field(
        ...,
        min_length=5,
        max_length=_MAX_CODE_LENGTH,
        description="待解释的代码",
    )
    language: Language = Field(
        default=Language.AUTO,
        description="编程语言：auto=自动检测",
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.DETAILED,
        description="详细程度：brief=概述, detailed=逐块解释, line_by_line=逐行解释",
    )

    # ── ✨ 反思模式控制 ──
    enable_reflection: bool = Field(
        default=True,
        description="是否启用反思模式（默认 True）"
    )
    llm_client: Optional[Any] = Field(
        default=None,
        description="LLM 客户端（反思模式需要）"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM 模型名称（反思模式使用，默认从 llm_client 提取）"
    )

    @field_validator("code")
    @classmethod
    def code_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code 不能为空白")
        return v


class CodeBlockExplanation(BaseModel):
    """核心作用：定义单个代码块的解释结构，不管是逐块还是逐行模式
    ，最终的解释都会封装成这个模型的实例，保证输出结构统一"""
    block_name: str = Field(..., description="代码块名称（函数/类/逻辑段/行号）")
    line_range: str = Field(..., description="行号范围，如 L1-L5")
    purpose: str = Field(..., description="该代码块的目的")
    how_it_works: str = Field(..., description="工作原理")
    potential_issues: list[str] = Field(
        default_factory=list,
        description="潜在问题/注意事项（单块级可能含'✅ 未发现问题'占位符）",
    )


class CodeExplainerOutput(BaseModel):
    """代码解释输出 — 字段精简，向后兼容通过 computed_field 实现"""
    language: str = Field(..., description="检测到的编程语言")
    overview: str = Field(..., description="整体概述（含复杂度评估）")
    blocks: list[CodeBlockExplanation] = Field(
        default_factory=list,
        description="逐块/逐行解释列表",
    )
    key_points: list[str] = Field(default_factory=list, description="关键要点")
    potential_issues: list[str] = Field(
        default_factory=list,
        description="所有真正的潜在问题汇总（不含占位符）",
    )
    # ── ✨ 反思报告 ──
    reflection_report: Optional[dict] = Field(
        default=None,
        description="反思报告（enable_reflection=True 时返回）"
    )

    # P1-10: 向后兼容字段，通过 computed_field 消除数据冗余
    @computed_field
    @property
    def detected_language(self) -> str:
        """向后兼容：与 language 相同"""
        return self.language

    @computed_field
    @property
    def explanation(self) -> str:
        """向后兼容：与 overview 相同"""
        return self.overview

    @computed_field
    @property
    def summary(self) -> str:
        """向后兼容：与 overview 相同"""
        return self.overview


# ============================================================================
# 这是模块的核心业务逻辑载体，继承自 BaseSkill，所有的分析算法都在这里
# ============================================================================
class CodeExplainerSkill(BaseSkill):
    """
    代码解释技能（v2.0.0）

    对代码进行逐行/逐块解释，生成自然语言的教学性说明。
    支持 9 种语言，三种详细程度，可选 guesslang 高精度语言检测，
    可选 radon 圈复杂度分析。
    """

    name: str = "code_explainer"
    description: str = (
        "对代码进行逐行/逐块解释，生成自然语言的教学性说明。"
        "支持 python / javascript / typescript / java / go / cpp / rust / kotlin / ruby，"
        "适合代码审查、新人培训、遗留代码理解。"
    )
    triggers: list[str] = [
        "解释代码", "代码解释", "explain code", "这段代码", "什么意思",
        "帮我看看", "分析代码", "代码分析", "code review", "逐行解释",
        "这段在做什么", "看不懂", "帮我理解",
    ]
    version: str = _SKILL_VERSION
    author: str = "EnterpriseLearningAgent"
    changelog: str = (
        "v2.0.0: 全面优化 — ast 精准分块/问题检测、detail_level 真正生效、"
        "特征累积替代互斥、clean 字段/向后兼容、guesslang/radon 可选依赖、"
        "9 种语言检测、异常处理、一次性 split、装饰器/生成器识别"
    )
    input_schema = CodeExplainerInput
    output_schema = CodeExplainerOutput

    # ========================================================================
    # Execute
    # ========================================================================
    def execute(self, input_data: CodeExplainerInput) -> CodeExplainerOutput:
        """
        P1-9: 添加异常处理包裹主流程。
        P1-8: 一次性 split，消除重复。
        """
        code = input_data.code.strip()
        language_raw = input_data.language
        detail_level = input_data.detail_level
        try:
            result = self._execute_inner(code, language_raw, detail_level)
        except Exception as exc:
            logger.exception("[code_explainer] 解释失败: %s", exc)
            return CodeExplainerOutput(
                language="unknown",
                overview=f"⚠️ 代码解释过程中发生错误：{exc}",
                blocks=[],
                key_points=["解释失败", f"错误信息: {exc}"],
                potential_issues=[],
            )

        # ── ✨ 反思模式（v2.1 新增）──
        if (input_data.enable_reflection
                and input_data.llm_client is not None
                and detail_level != DetailLevel.BRIEF):
            report, refined = self._run_code_review_reflection(
                code=code,
                initial_review=result.overview,
                language=result.language,
                llm_client=input_data.llm_client,
                model=input_data.model,  # ← 新增
            )
            if refined:
                result = result.model_copy(update={"overview": refined})
            result = result.model_copy(update={"reflection_report": report})

        return result

    #===========================================================================
    # 核心作用：整个技能的核心调度器，按顺序执行全流程，是所有子方法的关联中枢。
    # 入参：预处理后的代码字符串、用户指定的语言枚举、解释详细程度枚举
    # 出参：CodeExplainerOutput（完整的解释结果）
    # 关联关系：上游被execute调用，下游串联调用语言检测、代码分块、解释生成、要点提取所有核心子方法，控制全流程走向。
    #===========================================================================
    def _execute_inner(
        self,
        code: str,
        language_raw: Language,
        detail_level: DetailLevel,
    ) -> CodeExplainerOutput:
        # P1-8: 一次性 split
        lines = code.split("\n")

        # 语言检测
        if language_raw == Language.AUTO:
            language = self._detect_language(code)
        else:
            language = language_raw.value

        logger.info(
            "[code_explainer] 语言=%s | 详细程度=%s | 行数=%d",
            language, detail_level.value, len(lines),
        )

        # 分块
        raw_blocks: list[RawCodeBlock] = self._split_code_blocks(code, lines, language)

        # 概述（含真实复杂度评估，P1-7）
        overview = self._generate_overview(code, lines, language, raw_blocks)

        # 根据 detail_level 生成 blocks（P0-1）
        if detail_level == DetailLevel.BRIEF:
            blocks: list[CodeBlockExplanation] = []
        elif detail_level == DetailLevel.LINE_BY_LINE:
            blocks = self._explain_line_by_line(code, lines, language)
        else:
            blocks = self._explain_blocks(raw_blocks, language)

        # 关键要点
        key_points = self._extract_code_key_points(blocks, language, raw_blocks)

        # 汇总真正的问题（P1-5：排除占位符）
        all_issues = [
            issue
            for block in blocks
            for issue in block.potential_issues
            if not issue.startswith("✅")
        ]

        return CodeExplainerOutput(
            language=language,
            overview=overview,
            blocks=blocks,
            key_points=key_points,
            potential_issues=all_issues,
        )

    # ========================================================================
    # 核心作用：自动识别代码的编程语言，优先用机器学习库 guesslang，识别失败 / 未安装时用正则匹配兜底，兼顾准确率和可用性。
    # 入参：待检测的完整代码字符串
    # 出参：内部统一的语言标识（如python/javascript），识别失败返回unknown
    # 关联关系：上游被_execute_inner调用，返回的语言标识会传给后续所有分块、解释、检测方法，是核心基础参数。
    # ========================================================================
    def _detect_language(self, code: str) -> str:
        """P3-18: guesslang 优先，回退到增强的正则匹配"""
        if _GUESSLANG_AVAILABLE and _GUESS is not None:
            try:
                lang_full = _GUESS.language_name(code)
                detected = _GUESSLANG_MAPPING.get(lang_full, "unknown")
                if detected != "unknown":
                    logger.debug("[guesslang] 检测到语言: %s → %s", lang_full, detected)
                    return detected
            except Exception:
                logger.debug("[guesslang] 检测失败，回退到正则匹配", exc_info=True)

        # 回退：增强的正则匹配（P3-16: 扩展至 9 种语言）
        scores: dict[str, int] = {}
        for lang, patterns in _LANG_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, code, re.MULTILINE))
            if score > 0:
                scores[lang] = score

        if not scores:
            return "unknown"

        # 如果 typescript 得分高但 javascript 也高，优先 typescript
        if "typescript" in scores and "javascript" in scores:
            if scores["typescript"] >= scores["javascript"]:
                del scores["javascript"]

        return max(scores, key=scores.get)

    # ========================================================================
    # 核心作用：代码分块的分发器，根据语言选择分块策略，把完整代码拆分成函数 / 类 / 逻辑块，方便后续逐块解释。
    # 入参：完整代码字符串、提前按行拆分的代码列表、检测到的语言标识
    # 出参：list[RawCodeBlock]（拆分后的代码块列表）
    # 关联关系：上游被_execute_inner调用，Python 语言调用_split_via_ast精准分块，其他语言 /
    # 解析失败时调用_split_via_regex兜底，返回的分块结果传给后续概述、解释方法。
    # ========================================================================
    def _split_code_blocks(
        self, code: str, lines: list[str], language: str,
    ) -> list[RawCodeBlock]:
        """
        P0-2: Python 用 ast 精准分块，行号准确。
        其他语言用增强正则分割。
        """
        if language == "python":
            ast_blocks = self._split_via_ast(code, lines)
            if ast_blocks:
                return ast_blocks
            # SyntaxError 等回退到正则

        return self._split_via_regex(lines, language)
    # ===========================================================================
    # 核心作用：基于 Python 内置 AST 抽象语法树，精准拆分 Python 代码的函数、类、顶层代码，行号 100% 准确，不会被注释 / 字符串里的内容干扰。
    # 入参：完整 Python 代码、代码行列表
    # 出参：精准拆分后的代码块列表，语法错误时返回空列表
    # 关联关系：上游被_split_code_blocks调用，是 Python 代码的首选分块方案。
    #=============================================================================
    def _split_via_ast(self, code: str, lines: list[str]) -> list[RawCodeBlock]:
        """使用 Python ast 精准提取函数/类定义及其行号范围"""
        blocks: list[RawCodeBlock] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # 回退信号

        covered_end = 0
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                # Python 3.8+ 支持 end_lineno
                end = getattr(node, "end_lineno", len(lines))
                if end is None:
                    end = len(lines)

                block_code = "\n".join(lines[start - 1 : end])
                block_type = (
                    "class" if isinstance(node, ast.ClassDef)
                    else "async function" if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                blocks.append(RawCodeBlock(
                    name=f"{block_type} {node.name}",
                    code=block_code,
                    start_line=start,
                    end_line=end,
                ))
                covered_end = max(covered_end, end)

        # 顶层未覆盖的代码（import、全局变量等）
        if covered_end < len(lines):
            remaining = "\n".join(lines[covered_end:])
            if remaining.strip():
                blocks.insert(0, RawCodeBlock(
                    name="Module-level / Imports",
                    code=remaining,
                    start_line=covered_end + 1,
                    end_line=len(lines),
                ))

        return blocks

    # ========================================================================
    # 核心作用：通用正则分块兜底方案，支持 9 种语言，匹配函数 / 类定义关键字拆分代码，用于非 Python 语言、Python 代码语法错误的场景。
    # 入参：代码行列表、语言标识
    # 出参：拆分后的代码块列表
    # 关联关系：上游被_split_code_blocks调用，是分块逻辑的兜底方案。
    #==========================================================================
    def _split_via_regex(self, lines: list[str], language: str) -> list[RawCodeBlock]:
        """增强正则分割（非 Python 语言或 ast 回退）"""
        blocks: list[RawCodeBlock] = []
        current_name = "Main Block"
        current_start = 1
        current_lines: list[str] = []

        # 不同语言的新块检测模式
        new_block_patterns = [
            # Python
            r'^\s*(?:async\s+)?(?:def |class )(\w+)',
            # JavaScript / TypeScript
            r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)',
            r'^\s*(?:export\s+)?class\s+(\w+)',
            # Java
            r'^\s*(?:public|private|protected|static|\s)+(?:class|interface|enum)\s+(\w+)',
            r'^\s*(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\(',  # method
            # Go
            r'^\s*func\s+(\w+)',
            # Rust
            r'^\s*(?:pub\s+)?fn\s+(\w+)',
            r'^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)',
            # Kotlin
            r'^\s*(?:fun|class|object|interface)\s+(\w+)',
            # Ruby
            r'^\s*def\s+(\w+)',
            r'^\s*(?:class|module)\s+(\w+)',
        ]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # 跳过纯注释行（不作为新块起点）
            if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("--"):
                current_lines.append(line)
                continue

            is_new_block = False
            new_block_name = ""

            for pattern in new_block_patterns:
                match = re.match(pattern, stripped)
                if match:
                    is_new_block = True
                    keyword = re.match(r'^\s*(\w+)', stripped)
                    kw = keyword.group(1) if keyword else "block"
                    new_block_name = f"{kw} {match.group(1)}"
                    break

            if is_new_block and current_lines:
                block_code = "\n".join(current_lines)
                if block_code.strip():
                    blocks.append(RawCodeBlock(
                        name=current_name,
                        code=block_code,
                        start_line=current_start,
                        end_line=i - 1,
                    ))
                current_name = new_block_name
                current_start = i
                current_lines = [line]
            else:
                current_lines.append(line)

        # 最后一个块
        if current_lines:
            block_code = "\n".join(current_lines)
            if block_code.strip():
                blocks.append(RawCodeBlock(
                    name=current_name,
                    code=block_code,
                    start_line=current_start,
                    end_line=len(lines),
                ))

        return blocks if blocks else [
            RawCodeBlock(
                name="Full Code",
                code="\n".join(lines),
                start_line=1,
                end_line=len(lines),
            )
        ]

    # ========================================================================
    # 核心作用：生成代码的全局概述，包含代码行数、块数、Python 代码的圈复杂度评估，给用户整体认知。
    # 入参：完整代码、代码行列表、语言标识、拆分后的代码块列表
    # 出参：拼接好的全局概述文本
    # 关联关系：上游被_execute_inner调用，返回的概述是最终输出结果的核心字段。
    # ========================================================================
    def _generate_overview(
        self, code: str, lines: list[str], language: str, blocks: list[RawCodeBlock],
    ) -> str:
        """P1-7: 删除虚假评价，加入真实复杂度评估"""
        n_lines = len(lines)
        n_blocks = len(blocks)
        block_names = [b["name"] for b in blocks]

        parts = [
            f"这是一段 {language} 代码，共 {n_lines} 行，"
            f"包含 {n_blocks} 个逻辑块：{'、'.join(block_names[:5])}"
            f"{'...' if len(block_names) > 5 else ''}。",
        ]

        # P2-11: radon 圈复杂度分析（仅 Python）
        if language == "python" and _RADON_AVAILABLE:
            try:
                complexities = cc_visit(code)
                if complexities:
                    avg_cc = sum(b.complexity for b in complexities) / len(complexities)
                    max_cc = max(b.complexity for b in complexities)
                    if avg_cc <= 5:
                        level = "较低，代码结构相对简单"
                    elif avg_cc <= 10:
                        level = "中等，建议关注高复杂度函数"
                    else:
                        level = "偏高，建议拆分复杂函数"
                    parts.append(
                        f"平均圈复杂度 {avg_cc:.1f}（{level}），"
                        f"最高复杂度 {max_cc}。"
                    )
            except Exception:
                logger.debug("radon 分析失败", exc_info=True)

        return "".join(parts)

    # ========================================================================
    # 核心作用：默认detailed模式的核心方法，遍历每个代码块，生成完整的逐块解释。
    # 入参：拆分后的代码块列表、语言标识
    # 出参：list[CodeBlockExplanation]（每个代码块的标准化解释结果）
    # 关联关系：上游被_execute_inner在默认模式下调用，下游串联调用_infer_purpose、
    # _explain_how、_identify_issues，生成每个块的完整解释。
    # ========================================================================
    def _explain_blocks(
        self, raw_blocks: list[RawCodeBlock], language: str,
    ) -> list[CodeBlockExplanation]:
        """逐块生成解释"""
        blocks: list[CodeBlockExplanation] = []
        for rb in raw_blocks:
            block_code = rb["code"]
            purpose = self._infer_purpose(block_code, language, rb["name"])
            how_it = self._explain_how(block_code, language)
            issues = self._identify_issues(block_code, language)
            # P1-5: 单块级允许"未发现问题"，顶层汇总时过滤
            if not issues:
                issues = ["✅ 该块未发现明显问题"]
            blocks.append(CodeBlockExplanation(
                block_name=rb["name"],
                line_range=f"L{rb['start_line']}-L{rb['end_line']}",
                purpose=purpose,
                how_it_works=how_it,
                potential_issues=issues,
            ))
        return blocks

    # ========================================================================
    # 核心作用：line_by_line逐行模式的核心方法，给每一行代码生成单独的解释。
    # 入参：完整代码、代码行列表、语言标识
    # 出参：list[CodeBlockExplanation]（每一行代码的标准化解释结果）
    # 关联关系：上游被_execute_inner在逐行模式下调用，下游调用_infer_line_purpose生成每行代码的解释。
    #==========================================================================
    def _explain_line_by_line(
        self, code: str, lines: list[str], language: str,
    ) -> list[CodeBlockExplanation]:
        """P0-1: line_by_line 模式 — 每行生成解释"""
        blocks: list[CodeBlockExplanation] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                purpose = "空行，用于代码分段，提高可读性。"
            else:
                purpose = self._infer_line_purpose(stripped, language)
            blocks.append(CodeBlockExplanation(
                block_name=f"Line {i}",
                line_range=f"L{i}",
                purpose=purpose,
                how_it_works=stripped[:80] + ("..." if len(stripped) > 80 else ""),
                potential_issues=[],
            ))
        return blocks

    # ========================================================================
    # 核心作用：检测代码块的潜在问题、风险、不规范写法，包含 Python 专属检测和全语言通用检测（TODO 标记、硬编码敏感信息等）。
    # 入参：单个代码块的代码字符串、语言标识
    # 出参：检测到的问题列表，空列表表示无问题
    # 关联关系：上游被_explain_blocks调用，Python 语言会调用_identify_issues_via_ast做精准检测
    # ，返回的问题列表会作为代码块的风险提示。
    # =================
    def _infer_line_purpose(self, line: str, language: str) -> str:
        """推断单行代码的用途（line_by_line 模式）"""
        stripped = line.strip()

        if stripped.startswith("#") or stripped.startswith("//"):
            return "注释，用于解释代码意图或暂时禁用某段逻辑。"
        if re.match(r'^\s*(import |from |require\(|#include)', stripped):
            return "导入语句，引入外部模块/库。"
        if re.match(r'^\s*(def |class |function |func |fn |fun )', stripped):
            return "定义语句，声明函数/类/方法。"
        if re.match(r'^\s*(if |elif |else)', stripped):
            return "条件分支，根据布尔表达式选择执行路径。"
        if re.match(r'^\s*(for |while )', stripped):
            return "循环语句，重复执行代码块。"
        if "return " in stripped:
            return "返回语句，将计算结果传出函数。"
        if re.match(r'^\s*\w+\s*=\s*', stripped):
            return "赋值语句，将右侧表达式的结果绑定到变量。"
        if re.match(r'^\s*(print|console\.log|fmt\.Print|cout|puts|println)', stripped):
            return "输出语句，将信息打印到控制台/标准输出。"

        return "执行语句。"

    # ========================================================================
    # 核心作用：推断单行代码的用途，给逐行模式提供解释内容。
    # 入参：去空白后的单行代码字符串、语言标识
    # 出参：单行代码的用途解释文本
    # 关联关系：上游被_explain_line_by_line调用，是逐行模式的核心实现。
    # ========================================================================
    def _infer_purpose(self, code: str, language: str, block_name: str) -> str:
        """
        P0-3: 从互斥 if-elif 改为特征累积。
        所有匹配的特征都会被收集，信息不丢失。
        """
        purposes: list[str] = []

        # Python AST 深度分析
        if language == "python":
            ast_purposes = self._infer_purpose_via_ast(code)
            if ast_purposes:
                purposes.extend(ast_purposes)
        else:
            # 非 Python：关键词特征累积
            code_lower = code.lower()
            if re.search(r'\bdef\b|\bfunction\b|\bfunc\b|\bfn\b', code_lower):
                purposes.append("定义函数/方法")
            if re.search(r'\bclass\b|\bstruct\b|\binterface\b|\bobject\b', code_lower):
                purposes.append("定义类型（类/结构体/接口）")
            if re.search(r'\bfor\b|\bwhile\b|\.each\b|\.forEach\b', code_lower):
                purposes.append("包含循环迭代")
            if re.search(r'\bif\b|\belse\b|\bswitch\b|\bmatch\b', code_lower):
                purposes.append("包含条件分支")
            if re.search(r'\breturn\b', code_lower):
                purposes.append("计算并返回值")
            if re.search(r'\bimport\b|\brequire\b|#include|use\s+\w+::', code_lower):
                purposes.append("导入外部依赖")

        if not purposes:
            # 通过 block_name 推断
            if "import" in block_name.lower():
                purposes.append("导入外部依赖模块")
            elif any(kw in block_name.lower() for kw in ["main", "entry"]):
                purposes.append("程序入口点")

        if not purposes:
            return "执行一组顺序操作。"

        return "；".join(purposes) + "。"

    # ========================================================================
    # 核心作用：基于 Python AST 深度解析代码块的语法特征，精准推断用途，支持装饰器、生成器、上下文管理器等高级语法识别。
    # 入参：单个 Python 代码块的字符串
    # 出参：代码块的所有特征列表，用于拼接最终解释
    # 关联关系：上游被_infer_purpose调用，是 Python 代码用途推断的核心实现。
    #==========================================================================
    def _infer_purpose_via_ast(self, code: str) -> list[str]:
        """Python AST 深度特征提取"""
        purposes: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        node_types: set[str] = set()
        has_decorator = False

        for node in ast.walk(tree):
            node_types.add(type(node).__name__)

            # P2-14: 装饰器检测
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
                has_decorator = True

        if "FunctionDef" in node_types or "AsyncFunctionDef" in node_types:
            base = "定义函数/方法"
            if "AsyncFunctionDef" in node_types:
                base = "定义异步函数/协程"
            if has_decorator:
                base += "（含装饰器增强）"
            purposes.append(base)

        if "ClassDef" in node_types:
            purposes.append("定义类，封装数据和行为")

        if "For" in node_types or "While" in node_types:
            if "For" in node_types and "While" in node_types:
                purposes.append("包含 for 和 while 两种循环")
            elif "For" in node_types:
                purposes.append("包含 for 循环迭代")
            else:
                purposes.append("包含 while 循环")

        if "If" in node_types:
            purposes.append("包含条件分支控制")

        if "Return" in node_types:
            purposes.append("计算并返回值")

        if "Import" in node_types or "ImportFrom" in node_types:
            purposes.append("导入外部依赖模块")

        if "Try" in node_types:
            purposes.append("包含异常处理逻辑")

        if "Yield" in node_types:
            purposes.append("使用生成器惰性产出值（yield）")

        if "With" in node_types:
            purposes.append("使用上下文管理器管理资源")

        if "ListComp" in node_types or "DictComp" in node_types or "GeneratorExp" in node_types:
            purposes.append("使用推导式/生成器表达式")

        if "Global" in node_types:
            purposes.append("修改全局变量")

        if "Nonlocal" in node_types:
            purposes.append("修改外层作用域变量（闭包）")

        return purposes

    # ========================================================================
    # 核心作用：解释代码块的工作原理（「这段代码怎么实现的」），Python 用 AST 分析，其他语言用关键词匹配兜底。
    # 入参：单个代码块的代码字符串、语言标识
    # 出参：代码块的工作原理解释文本
    # 关联关系：上游被_explain_blocks调用，Python 语言调用_explain_how_via_ast，
    # 非 Python 语言调用_explain_how_via_keywords。
    # ========================================================================
    def _explain_how(self, code: str, language: str) -> str:
        """
        P1-4: Python 用 AST 结构分析，替代纯关键词 grep。
        其他语言保留关键词匹配。
        """
        lines = code.strip().split("\n")
        n = len(lines)

        parts: list[str] = []

        if language == "python":
            parts = self._explain_how_via_ast(code)
        else:
            parts = self._explain_how_via_keywords(lines)

        if not parts:
            parts.append("顺序执行语句")

        return "；".join(parts) + f"（共 {n} 行）"

    # =========================================================================
    # 核心作用：基于 Python AST 解析代码的执行逻辑，精准解释工作原理，比关键词匹配更准确。
    # 入参：单个 Python 代码块的字符串
    # 出参：代码块的执行逻辑特征列表
    # 关联关系：上游被_explain_how调用，是 Python 代码原理分析的核心实现。
    #============================================================================
    def _explain_how_via_ast(self, code: str) -> list[str]:
        """Python AST 结构分析"""
        parts: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        node_types: set[str] = set()
        call_names: set[str] = set()

        for node in ast.walk(tree):
            node_types.add(type(node).__name__)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                call_names.add(node.func.id)

        if "For" in node_types or "While" in node_types:
            parts.append("通过循环迭代处理数据")

        if "If" in node_types:
            parts.append("使用条件分支控制流程")

        if "Return" in node_types:
            parts.append("计算并返回值")

        if "print" in call_names:
            parts.append("输出信息到控制台")

        if "Try" in node_types:
            parts.append("包含异常处理逻辑")

        if "List" in node_types or "Dict" in node_types or "Set" in node_types:
            parts.append("操作数据结构（列表/字典/集合）")

        if "ListComp" in node_types or "DictComp" in node_types or "GeneratorExp" in node_types:
            parts.append("使用推导式/生成器表达式")

        if "With" in node_types:
            parts.append("使用上下文管理器管理资源")

        if "Yield" in node_types:
            parts.append("使用生成器惰性产出值")

        # 检查是否有装饰器
        has_decorator = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.decorator_list
            for n in ast.walk(tree)
        )
        if has_decorator:
            parts.append("使用装饰器增强函数功能")

        return parts

    # =========================================================================
    # 核心作用：非 Python 语言的兜底方案，用关键词匹配解释代码的工作原理。
    # 入参：单个代码块的代码行列表
    # 出参：代码块的执行逻辑特征列表
    # 关联关系：上游被_explain_how调用，是通用原理分析的兜底方案。
    #============================================================================
    @staticmethod
    def _explain_how_via_keywords(lines: list[str]) -> list[str]:
        """回退：关键词匹配（非 Python 语言）"""
        parts: list[str] = []
        if any("for " in l or "while " in l or ".each" in l or ".forEach" in l for l in lines):
            parts.append("通过循环迭代处理数据")
        if any("if " in l or "else " in l or "switch " in l for l in lines):
            parts.append("使用条件分支控制流程")
        if any("return " in l for l in lines):
            parts.append("计算并返回值")
        if any(
            kw in "\n".join(lines)
            for kw in ["print(", "console.log", "fmt.Print", "cout", "puts ", "println("]
        ):
            parts.append("输出信息到控制台")
        if any("try" in l or "except" in l or "catch" in l for l in lines):
            parts.append("包含异常处理逻辑")
        return parts

    # ========================================================================
    # Issue Detection (AST for Python, heuristic for others)
    # ========================================================================
    def _identify_issues(self, code: str, language: str) -> list[str]:
        """
        P2-12: Python 用 AST 精准检测裸 except、资源泄露等。
        其他语言沿用启发式检查。
        注意：顶层汇总时过滤 "✅" 占位符，此处返回空列表或真实问题。
        """
        issues: list[str] = []

        if language == "python":
            issues.extend(self._identify_issues_via_ast(code))

        # 通用检查
        if "TODO" in code or "FIXME" in code or "HACK" in code:
            issues.append("📝 代码中有 TODO/FIXME/HACK 标记，可能有未完成的工作")

        if "XXX" in code and language != "python":  # Python 的 XXX 可能是占位符用法
            pass

        # 硬编码敏感信息检查（所有语言）
        sensitive_keywords = ["password", "secret", "api_key", "token", "private_key"]
        for kw in sensitive_keywords:
            if kw in code.lower():
                # 粗略判断：如果同一行有赋值
                for line in code.split("\n"):
                    if kw in line.lower() and "=" in line:
                        issues.append(f"🔒 检测到可能的硬编码敏感信息 ({kw})，建议使用环境变量或密钥管理服务")
                        break

        return issues  # 空列表表示"未发现问题"（顶层会正确处理）

    # ========================================================================
    # 核心作用：基于 Python AST，精准检测 Python 代码的 3 个经典坑：裸 except、文件资源泄露、可变默认参数。
    # 入参：单个 Python 代码块的字符串
    # 出参：检测到的问题列表
    # 关联关系：上游被_identify_issues调用，是 Python 代码问题检测的核心实现。
    # ========================================================================
    def _identify_issues_via_ast(self, code: str) -> list[str]:
        """Python AST 级别问题检测（修复版）"""
        issues: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # 1. 裸 except 检测
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(
                        "⚠️ 使用了裸 except (bare except)，"
                        "建议捕获具体异常类型（如 ValueError）"
                    )
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    issues.append(
                        "💡 捕获了过于宽泛的 Exception，可考虑更具体的异常类型"
                    )

        # 2. open() 未使用 with 语句 —— 修复版
        # 第一步：收集所有 open() 调用节点
        open_calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "open"
            ):
                open_calls.append(node)

        if open_calls:
            # 第二步：构建 child → parent 映射
            parent_map: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parent_map[child] = parent

            # 第三步：逐节点沿祖先链查找 With
            for open_call in open_calls:
                current: ast.AST = open_call
                found_in_with = False
                while current in parent_map:
                    current = parent_map[current]
                    if isinstance(current, ast.With):
                        found_in_with = True
                        break
                if not found_in_with:
                    issues.append(
                        "⚠️ 检测到 open() 未使用 with 语句，"
                        "可能导致文件资源泄露"
                    )
                    break  # 只报一次，避免刷屏

        # 3. 可变默认参数（经典 Python 陷阱）
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(
                            f"⚠️ 函数 '{node.name}' 使用了可变默认参数"
                            f"（{type(default).__name__}），"
                            f"可能导致意外的状态跨调用共享"
                        )
                        break

        return issues

    # ========================================================================
    # 核心作用：提取代码的核心要点，方便用户快速抓住代码核心信息。
    # 入参：生成好的代码块解释列表、语言标识、原始代码分块列表
    # 出参：代码的关键要点列表
    # 关联关系：上游被_execute_inner调用，返回的要点列表是最终输出结果的核心字段。
    # ========================================================================
    def _extract_code_key_points(
        self,
        blocks: list[CodeBlockExplanation],
        language: str,
        raw_blocks: list[RawCodeBlock],
    ) -> list[str]:
        """P1-6: 硬截断加省略号判断"""
        points: list[str] = []
        points.append(f"语言：{language}")
        points.append(f"共 {len(raw_blocks)} 个代码块")

        for b in blocks:
            text = b.purpose
            truncated = text[:50]
            suffix = "..." if len(text) > 50 else ""
            points.append(f"• {b.block_name}: {truncated}{suffix}")

        return points

    @staticmethod
    def _run_code_review_reflection(
            code: str,
            initial_review: str,
            language: str,
            llm_client: Any,
            model: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """调用反思子图优化代码审查结果（含 flake8 lint）"""
        import difflib
        from src.agent.langgraph.graphs import (
            compile_reflection_graph,
            ReflectionConfig,
        )
        from src.agent.langgraph.state import create_initial_state

        try:
            config = ReflectionConfig(
                target_skill="code_explainer",  # ← 修复：改为正确的技能名称
                model=model or "qwen3.6-plus",
                output_field="overview",  # ← 修复：CodeExplainerOutput 的输出字段是 overview
                max_iterations=2,
                quality_threshold=0.7,
            )
            from skills import SkillManager

            sm = SkillManager()
            if not sm.get("code_explainer"):
                sm.register(CodeExplainerSkill)
            graph = compile_reflection_graph(
                target_skill_name="code_explainer",  # ← 修复：改为正确的技能名称
                skill_manager=sm,
                llm_client=llm_client,
                config=config,
            )

            # 注入原始代码到 state（供 ExternalFeedback 跑 flake8）
            initial_state = create_initial_state(code)

            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"cr-{hash(code) % 100000}"}},
            )

            ctx = result.get("reflection_context")
            if ctx is None or not ctx.refined_output:
                return None, None

            critique_summary = ""
            if ctx.critique:
                critique_summary = ctx.critique.summary or ""
                for p in (ctx.critique.points or [])[:5]:
                    critique_summary += f"\n  [{p.severity}] {p.description}"

            diff = "\n".join(difflib.unified_diff(
                initial_review.splitlines(),
                ctx.refined_output.splitlines(),
                fromfile="原始审查", tofile="优化审查", lineterm="",
            ))

            fb = ctx.feedback.content if ctx.feedback else ""

            report = {
                "enabled": True,
                "iterations": ctx.iteration,
                "final_score": ctx.critique.overall_score if ctx.critique else None,
                "critique_summary": critique_summary[:2000],
                "external_feedback": fb[:1500],
                "diff": diff[:3000],
                "status": ctx.status,
            }

            return report, ctx.refined_output


        except Exception as exc:
            import traceback
            logger.error("反思流水线失败，回退原始答案: %s", exc)
            logger.error("详细堆栈:\n%s", traceback.format_exc())
            return {"enabled": True, "error": str(exc)}, None


    # ========================================================================
    # Debug Representation
    # ========================================================================
    def __repr__(self) -> str:
        """P3-15: 调试友好"""
        langs = list(_LANG_PATTERNS.keys())
        return (
            f"CodeExplainerSkill(version={self.version!r}, "
            f"languages={langs}, guesslang={_GUESSLANG_AVAILABLE}, "
            f"radon={_RADON_AVAILABLE})"
        )

    # ============================================================================
    # 核心作用：给用户提供的极简调用入口，不需要手动实例化类、构建输入模型，一行代码即可完成代码解释。
    # 入参：待解释代码字符串、可选的语言 / 详细程度（支持枚举或字符串）
    # 出参：完整的标准化解释结果
    # 关联关系：上游被终端用户直接调用，下游实例化CodeExplainerSkill类，构建标准化输入，调用execute方法返回结果。
    # ============================================================================


# ============================================================================
# 便捷函数
# ============================================================================
def generate_code_explanation(
        code: str,
        language: Language | str = Language.AUTO,
        detail_level: DetailLevel | str = DetailLevel.DETAILED,
) -> CodeExplainerOutput:
    """
    便捷函数：一行调用解释代码。

    Args:
        code: 待解释的代码
        language: 编程语言 (auto 自动检测)
        detail_level: 详细程度 (brief/detailed/line_by_line)

    Returns:
        CodeExplainerOutput
    """
    skill = CodeExplainerSkill()
    return skill.execute(CodeExplainerInput(
        code=code,
        language=Language(language) if isinstance(language, str) else language,
        detail_level=DetailLevel(detail_level) if isinstance(detail_level, str) else detail_level,
    ))

    # ═══════════════════════════════════════════════
    # ✨ 反思模式核心逻辑
    # ═══════════════════════════════════════════════







