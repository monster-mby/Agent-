"""
Unit Test Generator Skill — 单元测试生成（优化版 v2.1.0）

这是一个企业级多语言单元测试自动生成工具，采用策略模式设计，能根据输入的函数代码自动生成覆盖正常路径、
边界情况和异常路径的测试用例。下面从功能概览、核心结构、类 / 方法详解、调用链路、代码质量五个维度进行深度解析。
一、功能概览
这个工具的核心能力是：
多语言支持：Python（pytest）、JavaScript（Jest）、Java（JUnit 5）、Go（testing）
智能解析：Python 用 AST 精准解析，其他语言用正则 + 括号计数回退
测试覆盖：自动生成三类测试用例（正常路径、边界情况、异常路径）
工程化：包含异常处理、版本管理、向后兼容等企业级特性
二、核心结构设计
代码采用分层架构+策略模式，核心结构如下：
plaintext
┌─────────────────────────────────────────────────────────────┐
│  1. 基础设施层（常量、枚举、数据模型）                          │
│     - SupportedLanguage/TestCategory (枚举)                   │
│     - UnitTestGeneratorInput/Output (Pydantic模型)            │
├─────────────────────────────────────────────────────────────┤
│  2. 策略模式层（语言特化逻辑）                                  │
│     - TestGeneratorStrategy (抽象基类)                         │
│     - Python/JavaScript/Java/GoTestStrategy (具体实现)        │
├─────────────────────────────────────────────────────────────┤
│  3. 核心业务层（主技能类）                                      │
│     - UnitTestGeneratorSkill (继承BaseSkill)                  │
├─────────────────────────────────────────────────────────────┤
│  4. 便捷接口层                                                  │
│     - generate_unit_tests (一键调用函数)                       │
└─────────────────────────────────────────────────────────────┘

v2.1.0 优化改动:
  - 策略基类提取公共映射 (SAMPLE_MAP/ZERO_MAP)
  - 异常处理细化 (区分 SyntaxError)
  - 使用 textwrap 格式化代码字符串


generate_unit_tests()
  [便捷入口函数] 封装初始化与调用流程，提供一键生成单元测试的接口
  ↓
UnitTestGeneratorSkill.execute()
  [异常处理包装层] 捕获执行异常，区分语法错误/通用错误并返回友好提示，调用核心流程
  ↓
UnitTestGeneratorSkill._execute_inner()
  [核心流程控制器] 协调语言检测、函数解析、策略获取、测试生成与结果组装全流程
  ├→ _detect_language()  [如果language=auto]
  │    [自动语言识别] 通过函数签名正则特征自动判断编程语言
  ├→ _parse_python_ast()
  │    [Python AST解析] 用抽象语法树精准提取Python函数名、参数、是否async/方法等信息
  │   └→ _extract_type_hint_from_annotation()
  │        [类型提示提取] 从AST注解节点中提取简化的参数类型字符串
  ├→ _get_strategy() → PythonTestStrategy
  │    [策略工厂] 根据语言返回对应的测试生成策略实例
  ├→ PythonTestStrategy.generate_imports()
  │    [导入语句生成] 生成pytest框架导入及被测函数/类的导入示例
  ├→ PythonTestStrategy.generate_happy_path()
  │    [正常用例生成] 基于示例参数生成验证基本功能的测试用例
  │   └→ make_sample_args()
  │        [示例参数生成] 根据参数类型提示生成对应示例值（如int→42）
  ├→ PythonTestStrategy.generate_edge_cases()
  │    [边界用例生成] 基于零值/大输入生成验证边界条件的测试用例
  │   └→ _make_zero_args()
  │        [零值参数生成] 根据参数类型提示生成对应零值/空值（如int→0）
  ├→ PythonTestStrategy.generate_exception_cases()
  │    [异常用例生成] 基于错误类型参数生成验证错误处理的测试用例
  └→ 组装UnitTestGeneratorOutput
       [结果组装] 统计用例数量、生成覆盖率说明、格式化完整测试代码并返回
"""

from __future__ import annotations

import re
import ast
import logging
import textwrap
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional, TypedDict

from pydantic import BaseModel, Field, computed_field, ConfigDict

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ============================================================================
# Module Constants
# ============================================================================
_SKILL_VERSION = "2.1.0"
_MAX_CODE_LENGTH = 10000


# ============================================================================
# 定义支持的编程语言（python/javascript/java/go/auto），用StrEnum保证类型安全。
# ============================================================================
class SupportedLanguage(StrEnum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    AUTO = "auto"

# ============================================================================
# 定义测试用例类型（happy_path/edge_case/exception），用于分类管理测试。
# ============================================================================
class TestCategory(StrEnum):
    """测试类别"""
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    EXCEPTION = "exception"


# ============================================================================
# 描述函数参数的结构（name参数名、type_hint类型提示）。
# ============================================================================
class ParameterInfo(TypedDict):
    """函数参数信息"""
    name: str
    type_hint: str

# ============================================================================
# 测试用例的中间表示（name/desc/code），用于策略层生成后组装。
# ============================================================================
class RawTestCase(TypedDict):
    """原始测试用例中间表示"""
    name: str
    desc: str
    code: str


# ============================================================================
# 预编译正则
# ============================================================================
# 语言检测正则（_RE_LANG_PYTHON等）：通过函数签名特征自动识别语言
_RE_LANG_PYTHON = re.compile(r'^\s*(?:async\s+)?def\s+\w+', re.MULTILINE)
_RE_LANG_JS = re.compile(r'(?:function\s+\w+|const\s+\w+\s*=|export\s+const\s+\w+\s*=|=>\s*\{)', re.MULTILINE)
_RE_LANG_GO = re.compile(r'^\s*func\s+(?:\(\w+\s+\*?[\w.]+\)\s+)?\w+\s*\(', re.MULTILINE)
_RE_LANG_JAVA = re.compile(r'^\s*(?:public|private|protected)\s+(?:static\s+)?(?:void|\w+)\s+\w+\s*\(', re.MULTILINE)

# 函数名提取正则（_RE_FUNC_JS等）：非 Python 语言回退提取函数名
_RE_FUNC_JS = re.compile(
    r'(?:function\s+(\w+)|(?:export\s+)?const\s+(\w+)\s*=\s*(?:\(|async\s*\()|(\w+)\s*=\s*function)'
)
_RE_FUNC_GO = re.compile(
    r'func\s+(?:\(\s*\w+\s+\*?[\w.]+\s*\)\s+)?(\w+)\s*\('
)
_RE_FUNC_JAVA = re.compile(
    r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]*>)?\s+)?(\w+)\s*\(.*\)\s*(?:\{|throws)'
)

# 参数提取正则（_RE_PARENS_BALANCED等）：处理嵌套括号的参数分割
_RE_PARENS_BALANCED = re.compile(r'\((.*)\)', re.DOTALL)  # 初始匹配，再用括号计数
_RE_PARAM_JS_DESTRUCTURE = re.compile(r'\{[^}]+\}')
_RE_PARAM_NAME = re.compile(r'^\s*(\w+)')

# --- Return 检测 ---
_RE_RETURN = re.compile(r'\breturn\b')

# --- Python 特殊 ---
_RE_SELF_CLS = re.compile(r'^\s*(self|cls)\b')


# ============================================================================
# UnitTestGeneratorInput：输入校验模型
# 输入：code（函数代码）、language（编程语言）、include_edge_cases/include_exception_cases（开关）
# 校验：code长度限制（5-10000）、language枚举校验
# ============================================================================
class UnitTestGeneratorInput(BaseModel):
    """单元测试生成输入"""
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(
        ...,
        min_length=5,
        max_length=_MAX_CODE_LENGTH,
        description="函数/方法代码",
        alias="function_code",
    )
    language: SupportedLanguage = Field(
        default=SupportedLanguage.PYTHON,
        description="编程语言：auto=自动检测",
    )
    include_edge_cases: bool = Field(
        default=True,
        description="是否包含边界情况测试",
    )
    include_exception_cases: bool = Field(
        default=True,
        description="是否包含异常路径测试",
    )

# ============================================================================
# TestCase：单个测试用例的输出模型
# 输出：name（测试名）、description（描述）、code（测试代码）、category（测试类型）
# ============================================================================
class TestCase(BaseModel):
    """单个测试用例"""
    name: str = Field(..., description="测试用例名称")
    description: str = Field(..., description="测试描述")
    code: str = Field(..., description="测试代码")
    category: TestCategory = Field(..., description="测试类别")

# ============================================================================
# UnitTestGeneratorOutput：完整输出模型
# 输出：function_name（被测函数名）、language（语言）、framework（测试框架）、imports（导入语句）、test_cases（测试用例列表）、coverage_note（覆盖率说明）、test_code（完整测试代码）、test_count（用例数量）
# 向后兼容：用computed_field让coverage_notes（旧字段）等价于coverage_note（新字段）
# ============================================================================
class UnitTestGeneratorOutput(BaseModel):
    """单元测试生成输出（v2.0.0 精简版）"""
    function_name: str = Field(..., description="被测函数名称")
    language: str = Field(..., description="编程语言")
    framework: str = Field(..., description="测试框架")
    imports: str = Field(..., description="需要导入的模块")
    test_cases: list[TestCase] = Field(default_factory=list, description="测试用例列表")
    coverage_note: str = Field(default="", description="覆盖率说明")
    test_code: str = Field(default="", description="完整测试代码字符串")
    test_count: int = Field(default=0, description="测试用例数量")

    # P1-4: 向后兼容字段（computed_field 消除冗余）
    @computed_field
    @property
    def coverage_notes(self) -> str:
        """向后兼容：与 coverage_note 相同"""
        return self.coverage_note


# ============================================================================
# 抽象基类 TestGeneratorStrategy
# 核心映射：
# SAMPLE_MAP：类型到示例值的映射（如int→42）
# ZERO_MAP：类型到零值的映射（如int→0）
# ============================================================================
class TestGeneratorStrategy(ABC):
    """测试生成策略抽象基类"""

    # 子类只需覆盖这两个映射
    SAMPLE_MAP: dict[str, str] = {}#类型到示例值的映射（如int→42）
    ZERO_MAP: dict[str, str] = {}#类型到零值的映射（如int→0）

    #返回测试框架名称（如pytest）
    @abstractmethod
    def get_framework(self) -> str:
        ...

    #返回导入语句
    @abstractmethod
    def generate_imports(self, func_name: str, is_method: bool) -> str:
        ...

    #生成正常路径测试用例
    @abstractmethod
    def generate_happy_path(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool, is_async: bool,
    ) -> list[RawTestCase]:
        ...

    #生成边界路径测试用例
    @abstractmethod
    def generate_edge_cases(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool,
    ) -> list[RawTestCase]:
        ...

    #生异常路径测试用例
    @abstractmethod
    def generate_exception_cases(
        self, func_name: str, params: list[ParameterInfo],
        is_method: bool,
    ) -> list[RawTestCase]:
        ...

    # 公共方法（子类无需重写（子类复用））
    #根据SAMPLE_MAP生成示例参数
    def make_sample_args(self, params: list[ParameterInfo]) -> str:
        return ", ".join(
            self.SAMPLE_MAP.get(p["type_hint"], self._null_literal())
            for p in params
        )

    #根据ZERO_MAP生成零参数
    def _make_zero_args(self, params: list[ParameterInfo]) -> str:
        if not params:
            return ""
        return ", ".join(
            self.ZERO_MAP.get(p["type_hint"], self._null_literal())
            for p in params
        )

    @staticmethod
    def _bool_literal() -> str:
        """本语言的布尔字面量"""
        return "true"

    @staticmethod
    def _null_literal() -> str:
        """本语言的 null 字面量"""
        return "null"

# ============================================================================
# 具体策略类（以PythonTestStrategy为例）
# ============================================================================
class PythonTestStrategy(TestGeneratorStrategy):

    #针对 Python 类型定制（如NoneType→None）
    SAMPLE_MAP = {
        "int": "42", "float": "3.14", "str": '"hello"',
        "bool": "True", "list": "[]", "dict": "{}",
        "any": "None", "NoneType": "None",
    }
    ZERO_MAP = {
        "int": "0", "float": "0.0", "str": '""',
        "bool": "False", "list": "[]", "dict": "{}",
    }

    def get_framework(self) -> str:
        return "pytest"

    @staticmethod
    def _bool_literal() -> str:
        return "True"

    @staticmethod
    def _null_literal() -> str:
        return "None"

    #生成pytest导入和被测函数导入
    def generate_imports(self, func_name: str, is_method: bool) -> str:
        if is_method:
            return (
                "import pytest\n"
                f"from your_module import TheClass  # Replace with actual class import\n\n"
                f"# Example instantiation:\n"
                f"# obj = TheClass()\n"
            )
        return f"import pytest\nfrom your_module import {func_name}  # Replace with actual module"

    #正常路径测试用例
    def generate_happy_path(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool, is_async: bool,
    ) -> list[RawTestCase]:
        sample_args = self.make_sample_args(params)
        async_kw = "async " if is_async else ""
        await_kw = "await " if is_async else ""
        call_prefix = "obj." if is_method else ""
        setup_line = "obj = TheClass()\n    " if is_method else ""
        assert_line = (
            "assert result is not None" if has_return
            else "# Verify side effects"
        )

        test_body = textwrap.dedent(f"""
            {async_kw}def test_{func_name}_basic():
                {setup_line}result = {await_kw}{call_prefix}{func_name}({sample_args})
                {assert_line}
        """).strip()

        return [RawTestCase(
            name=f"test_{func_name}_basic",
            desc=f"Test {func_name} basic functionality with typical parameters",
            code=test_body,
        )]

    #边界路径测试用例
    def generate_edge_cases(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool,
    ) -> list[RawTestCase]:
        cases: list[RawTestCase] = []
        call_prefix = "obj." if is_method else ""

        # Empty / zero-value test
        zero_args = self._make_zero_args(params)
        cases.append(RawTestCase(
            name=f"test_{func_name}_empty_input",
            desc=f"Edge case: pass empty/zero values",
            code=(
                f"def test_{func_name}_empty_input():\n"
                f"    {'obj = TheClass()' + chr(10) + '    ' if is_method else ''}"
                f"    result = {call_prefix}{func_name}({zero_args})\n"
                f"    # Add assertion based on function semantics\n"
                f"    assert True  # TODO: replace with specific assertion"
            ),
        ))

        # Large input test (only if params exist)
        if params:
            large_map = {
                "int": "1_000_000", "float": "1e6",
                "str": '"x" * 10_000', "list": "list(range(10_000))",
                "dict": "{i: i for i in range(1_000)}",
            }
            large_args = ", ".join(
                large_map.get(p["type_hint"], "None") for p in params
            )
            cases.append(RawTestCase(
                name=f"test_{func_name}_large_input",
                desc=f"Edge case: pass large volume of data",
                code=(
                    f"def test_{func_name}_large_input():\n"
                    f"    {'obj = TheClass()' + chr(10) + '    ' if is_method else ''}"
                    f"    result = {call_prefix}{func_name}({large_args})\n"
                    f"    assert result is not None"
                ),
            ))

        return cases

    #异常测试用例
    def generate_exception_cases(
        self, func_name: str, params: list[ParameterInfo],
        is_method: bool,
    ) -> list[RawTestCase]:
        call_prefix = "obj." if is_method else ""

        # P0-11: 构造正确的参数个数，但类型错误
        if params:
            # 用错误类型但正确数量的参数
            bad_args = ", ".join(
                '"not_an_int"' if p["type_hint"] in ("int", "float") else
                "42" if p["type_hint"] == "str" else
                "None"
                for p in params
            )
            call_str = f"{call_prefix}{func_name}({bad_args})"
        else:
            call_str = f"{call_prefix}{func_name}(unexpected_kwarg=123)"

        return [RawTestCase(
            name=f"test_{func_name}_invalid_type",
            desc=f"Exception test: pass wrong type to verify TypeError/ValueError",
            code=(
                f"def test_{func_name}_invalid_type():\n"
                f"    {'obj = TheClass()' + chr(10) + '    ' if is_method else ''}"
                f"    with pytest.raises((TypeError, ValueError)):\n"
                f"        {call_str}\n"
            ),
        )]


# ============================================================================
# 核心业务层 UnitTestGeneratorSkill
# 这是工具的主类，继承自BaseSkill（自定义基类，代码中未给出），负责协调整个测试生成流程。
# ============================================================================
class JavaScriptTestStrategy(TestGeneratorStrategy):

    SAMPLE_MAP = {
        "int": "42", "float": "3.14", "str": '"hello"',
        "bool": "true", "list": "[]", "dict": "{}",
        "any": "null", "number": "42", "string": '"hello"',
    }
    ZERO_MAP = {
        "int": "0", "float": "0.0", "str": '""',
        "bool": "false", "list": "[]", "dict": "{}",
        "number": "0", "string": '""',
    }

    def get_framework(self) -> str:
        return "jest"

    def generate_imports(self, func_name: str, is_method: bool) -> str:
        if is_method:
            return (
                f"const {{ TheClass }} = require('./your-module');  // Replace with actual path\n"
                f"// const obj = new TheClass();\n"
            )
        return f"const {{ {func_name} }} = require('./your-module');  // Replace with actual path"

    def generate_happy_path(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool, is_async: bool,
    ) -> list[RawTestCase]:
        sample_args = self.make_sample_args(params)
        call_prefix = "obj." if is_method else ""
        async_kw = "async " if is_async else ""
        await_kw = "await " if is_async else ""

        return [RawTestCase(
            name=f"test {func_name} basic functionality",
            desc=f"Test {func_name} basic functionality",
            code=(
                f"test('{func_name} should work correctly', {async_kw}() => {{\n"
                f"    {'const obj = new TheClass();' + chr(10) + '    ' if is_method else ''}"
                f"    const result = {await_kw}{call_prefix}{func_name}({sample_args});\n"
                f"{'    expect(result).toBeDefined();' if has_return else '    // Verify side effects'}\n"
                f"}});"
            ),
        )]

    def generate_edge_cases(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool,
    ) -> list[RawTestCase]:
        call_prefix = "obj." if is_method else ""
        zero_args = self._make_zero_args(params)

        return [RawTestCase(
            name=f"test {func_name} with empty input",
            desc=f"Edge case: pass empty/zero values",
            code=(
                f"test('{func_name} handles empty input', () => {{\n"
                f"    {'const obj = new TheClass();' + chr(10) + '    ' if is_method else ''}"
                f"    const result = {call_prefix}{func_name}({zero_args});\n"
                f"    // TODO: add assertion\n"
                f"    expect(true).toBe(true);\n"
                f"}});"
            ),
        )]

    def generate_exception_cases(
        self, func_name: str, params: list[ParameterInfo],
        is_method: bool,
    ) -> list[RawTestCase]:
        call_prefix = "obj." if is_method else ""
        return [RawTestCase(
            name=f"test {func_name} throws on invalid input",
            desc=f"Exception test: pass invalid argument",
            code=(
                f"test('{func_name} throws on invalid input', () => {{\n"
                f"    {'const obj = new TheClass();' + chr(10) + '    ' if is_method else ''}"
                f"    expect(() => {call_prefix}{func_name}(null)).toThrow();\n"
                f"}});"
            ),
        )]


class JavaTestStrategy(TestGeneratorStrategy):

    SAMPLE_MAP = {
        "int": "42", "float": "3.14f", "str": '"hello"',
        "bool": "true", "list": "List.of()", "dict": "Map.of()",
        "any": "null", "String": '"hello"', "Object": "null",
    }
    ZERO_MAP = {
        "int": "0", "float": "0.0f", "str": '""',
        "bool": "false", "list": "List.of()", "dict": "Map.of()",
        "String": '""',
    }

    def get_framework(self) -> str:
        return "JUnit 5"

    def generate_imports(self, func_name: str, is_method: bool) -> str:
        return (
            "import org.junit.jupiter.api.Test;\n"
            "import static org.junit.jupiter.api.Assertions.*;\n"
            "// Replace your.package.YourClass with actual package path"
        )

    def generate_happy_path(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool, is_async: bool,
    ) -> list[RawTestCase]:
        sample_args = self.make_sample_args(params)
        call_prefix = "obj." if is_method else ""
        return [RawTestCase(
            name=f"test{func_name.capitalize()}Basic",
            desc=f"Test {func_name} basic functionality",
            code=(
                f"@Test\n"
                f"void test{func_name.capitalize()}Basic() {{\n"
                f"    {'var obj = new YourClass();' + chr(10) + '    ' if is_method else ''}"
                f"    var result = {call_prefix}{func_name}({sample_args});\n"
                f"{'    assertNotNull(result);' if has_return else '    // Verify side effects'}\n"
                f"}}"
            ),
        )]

    def generate_edge_cases(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool,
    ) -> list[RawTestCase]:
        # P0-9: Java 补充边界测试
        call_prefix = "obj." if is_method else ""
        zero_args = self._make_zero_args(params)
        return [RawTestCase(
            name=f"test{func_name.capitalize()}EmptyInput",
            desc=f"Edge case: pass empty/zero/null values",
            code=(
                f"@Test\n"
                f"void test{func_name.capitalize()}EmptyInput() {{\n"
                f"    {'var obj = new YourClass();' + chr(10) + '    ' if is_method else ''}"
                f"    var result = {call_prefix}{func_name}({zero_args});\n"
                f"    // TODO: add assertion\n"
                f"    assertTrue(true);\n"
                f"}}"
            ),
        )]

    def generate_exception_cases(
        self, func_name: str, params: list[ParameterInfo],
        is_method: bool,
    ) -> list[RawTestCase]:
        call_prefix = "obj." if is_method else ""
        return [RawTestCase(
            name=f"test{func_name.capitalize()}ThrowsOnNull",
            desc=f"Exception test: pass null argument",
            code=(
                f"@Test\n"
                f"void test{func_name.capitalize()}ThrowsOnNull() {{\n"
                f"    {'var obj = new YourClass();' + chr(10) + '    ' if is_method else ''}"
                f"    assertThrows(Exception.class, () -> {call_prefix}{func_name}(null));\n"
                f"}}"
            ),
        )]


class GoTestStrategy(TestGeneratorStrategy):

    SAMPLE_MAP = {
        "int": "42", "float64": "3.14", "str": '"hello"',
        "bool": "true", "list": "[]int{}", "dict": "map[string]int{}",
        "any": "nil", "string": '"hello"', "interface{}": "nil",
        "error": "nil",
    }
    ZERO_MAP = {
        "int": "0", "float64": "0.0", "str": '""',
        "bool": "false", "list": "nil", "dict": "nil",
        "string": '""', "interface{}": "nil",
    }

    @staticmethod
    def _bool_literal() -> str:
        return "true"

    @staticmethod
    def _null_literal() -> str:
        return "nil"

    def get_framework(self) -> str:
        return "testing"

    def generate_imports(self, func_name: str, is_method: bool) -> str:
        return (
            'import (\n'
            '    "testing"\n'
            '    // Replace with actual package import path\n'
            ')'
        )

    def generate_happy_path(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool, is_async: bool,
    ) -> list[RawTestCase]:
        sample_args = self.make_sample_args(params)
        call_prefix = "obj." if is_method else ""
        return [RawTestCase(
            name=f"Test{_capitalize(func_name)}Basic",
            desc=f"Test {func_name} basic functionality",
            code=(
                f"func Test{_capitalize(func_name)}Basic(t *testing.T) {{\n"
                f"    {'obj := NewTheClass()' + chr(10) + '    ' if is_method else ''}"
                f"    result := {call_prefix}{func_name}({sample_args})\n"
                f"    // TODO: add assertion\n"
                f"    _ = result\n"
                f"}}"
            ),
        )]

    def generate_edge_cases(
        self, func_name: str, params: list[ParameterInfo],
        has_return: bool, is_method: bool,
    ) -> list[RawTestCase]:
        # P0-9: Go 补充边界测试
        call_prefix = "obj." if is_method else ""
        zero_args = self._make_zero_args(params)
        return [RawTestCase(
            name=f"Test{_capitalize(func_name)}EmptyInput",
            desc=f"Edge case: pass zero/nil values",
            code=(
                f"func Test{_capitalize(func_name)}EmptyInput(t *testing.T) {{\n"
                f"    {'obj := NewTheClass()' + chr(10) + '    ' if is_method else ''}"
                f"    result := {call_prefix}{func_name}({zero_args})\n"
                f"    // TODO: add assertion\n"
                f"    _ = result\n"
                f"}}"
            ),
        )]

    def generate_exception_cases(
        self, func_name: str, params: list[ParameterInfo],
        is_method: bool,
    ) -> list[RawTestCase]:
        call_prefix = "obj." if is_method else ""
        return [RawTestCase(
            name=f"Test{_capitalize(func_name)}InvalidInput",
            desc=f"Exception test: pass invalid argument",
            code=(
                f"func Test{_capitalize(func_name)}InvalidInput(t *testing.T) {{\n"
                f"    {'obj := NewTheClass()' + chr(10) + '    ' if is_method else ''}"
                f"    // result := {call_prefix}{func_name}(/* invalid input */)\n"
                f"    // TODO: verify error handling\n"
                f"}}"
            ),
        )]


def _capitalize(s: str) -> str:
    """首字母大写（处理空字符串）"""
    return s[0].upper() + s[1:] if s else s


# ============================================================================
# Strategy Factory
# ============================================================================
_STRATEGY_REGISTRY: dict[str, TestGeneratorStrategy] = {
    "python": PythonTestStrategy(),
    "javascript": JavaScriptTestStrategy(),
    "java": JavaTestStrategy(),
    "go": GoTestStrategy(),
}


def _get_strategy(language: str) -> TestGeneratorStrategy:
    strategy = _STRATEGY_REGISTRY.get(language)
    if strategy is None:
        logger.warning("Unsupported language '%s', falling back to Python", language)
        return _STRATEGY_REGISTRY["python"]
    return strategy


# ============================================================================
# 核心业务层 UnitTestGeneratorSkill
# 这是工具的主类，继承自BaseSkill（自定义基类，代码中未给出），负责协调整个测试生成流程。
# ============================================================================
class UnitTestGeneratorSkill(BaseSkill):
    """
    单元测试生成技能（v2.1.0）

    根据函数代码自动生成单元测试用例，覆盖正常路径、边界情况和异常路径。
    支持 python(pytest) / javascript(jest) / java(JUnit 5) / go(testing)。
    """

    name: str = "unit_test_generator"
    description: str = (
        "根据函数代码自动生成单元测试用例，覆盖正常路径、边界情况和异常路径。"
        "支持 python(pytest)/javascript(jest)/java(JUnit 5)/go(testing)。"
    )
    triggers: list[str] = [
        "单元测试", "测试用例", "unit test", "生成测试", "写测试",
        "test case", "pytest", "unittest", "帮我测试", "补测试",
        "测试覆盖", "test coverage",
    ]
    version: str = _SKILL_VERSION
    author: str = "EnterpriseLearningAgent"
    changelog: str = (
        "v2.1.0: 策略基类提取公共映射 (SAMPLE_MAP/ZERO_MAP)、"
        "异常处理细化 (SyntaxError)、textwrap 代码格式化"
    )
    input_schema = UnitTestGeneratorInput
    output_schema = UnitTestGeneratorOutput

    # ========================================================================
    # Execute
    # 输入：UnitTestGeneratorInput
    # 输出：UnitTestGeneratorOutput
    # 逻辑：异常捕获 wrapper，调用_execute_inner()，区分SyntaxError（代码语法错误）和其他异常，返回友好错误信息
    # ========================================================================
    def execute(self, input_data: UnitTestGeneratorInput) -> UnitTestGeneratorOutput:
        try:
            return self._execute_inner(input_data)
        except SyntaxError as exc:
            logger.warning("[unit_test_generator] Syntax error: %s", exc)
            return UnitTestGeneratorOutput(
                function_name="unknown_function",
                language=input_data.language.value,
                framework="unknown",
                imports="",
                test_cases=[],
                coverage_note=(
                    f"Syntax error in input code: line {exc.lineno}, {exc.msg}. "
                    f"Please fix syntax errors before generating tests."
                ),
                test_code=f"# SyntaxError: line {exc.lineno}, {exc.msg}",
                test_count=0,
            )
        except Exception as exc:
            logger.exception("[unit_test_generator] Unexpected error: %s", exc)
            return UnitTestGeneratorOutput(
                function_name="unknown_function",
                language="unknown",
                framework="unknown",
                imports="",
                test_cases=[],
                coverage_note=(
                    f"Unexpected error: {type(exc).__name__}: {exc}"
                ),
                test_code=f"# Error: {type(exc).__name__}: {exc}",
                test_count=0,
            )

    # ========================================================================
    # _execute_inner()（核心逻辑）：
    # 预处理：去除代码首尾空白
    # 语言检测：如果是auto，调用_detect_language()自动识别
    # 函数解析：
    # Python：用_parse_python_ast()精准解析
    # 其他语言：用正则回退提取函数名和参数
    # 策略获取：调用_get_strategy()获取对应语言的策略实例
    # 测试生成：
    # 调用策略的generate_imports()生成导入
    # 调用generate_happy_path()生成正常用例
    # 如果开关打开，调用generate_edge_cases()/generate_exception_cases()生成边界 / 异常用例
    # 组装输出：
    # 计算覆盖率说明（用例数量分类统计）
    # 用textwrap格式化完整测试代码
    # 返回UnitTestGeneratorOutput
    # ========================================================================
    def _execute_inner(self, input_data: UnitTestGeneratorInput) -> UnitTestGeneratorOutput:
        code = input_data.code.strip()
        language = input_data.language.value

        # 自动检测语言
        if language == "auto":
            language = self._detect_language(code)

        # P0-1: Python 优先用 AST 解析
        if language == "python":
            func_info = self._parse_python_ast(code)
            func_name = func_info["name"]
            params = func_info["params"]
            has_return = func_info["has_return"]
            is_async = func_info.get("is_async", False)
            is_method = func_info.get("is_method", False)
        else:
            func_name = self._extract_function_name_fallback(code, language)
            params = self._extract_params_fallback(code, language)
            has_return = bool(_RE_RETURN.search(code))
            is_async = False
            is_method = False

        # 获取语言策略
        strategy = _get_strategy(language)

        # 确定测试框架
        framework = strategy.get_framework()

        # 生成 import
        imports = strategy.generate_imports(func_name, is_method)

        # 生成测试用例
        test_cases: list[TestCase] = []

        # 1. 正常路径
        for tc in strategy.generate_happy_path(
            func_name, params, has_return, is_method, is_async,
        ):
            test_cases.append(TestCase(
                name=tc["name"], description=tc["desc"],
                code=tc["code"], category=TestCategory.HAPPY_PATH,
            ))

        # 2. 边界情况 (P0-9: 所有语言都有)
        if input_data.include_edge_cases:
            for tc in strategy.generate_edge_cases(
                func_name, params, has_return, is_method,
            ):
                test_cases.append(TestCase(
                    name=tc["name"], description=tc["desc"],
                    code=tc["code"], category=TestCategory.EDGE_CASE,
                ))

        # 3. 异常路径 (P0-9: 所有语言都有)
        if input_data.include_exception_cases:
            for tc in strategy.generate_exception_cases(
                func_name, params, is_method,
            ):
                test_cases.append(TestCase(
                    name=tc["name"], description=tc["desc"],
                    code=tc["code"], category=TestCategory.EXCEPTION,
                ))

        # 覆盖率说明 (P1-15: 去中文硬编码)
        happy_count = sum(1 for t in test_cases if t.category == TestCategory.HAPPY_PATH)
        edge_count = sum(1 for t in test_cases if t.category == TestCategory.EDGE_CASE)
        exc_count = sum(1 for t in test_cases if t.category == TestCategory.EXCEPTION)
        coverage_note = (
            f"Generated {len(test_cases)} test cases: "
            f"{happy_count} happy path, {edge_count} edge case, {exc_count} exception. "
            f"Suggest supplementing: mock external dependencies, integration tests, performance tests."
        )

        # 生成完整测试代码 (P3-22: 去多余空行 + v2.1.0 textwrap)
        body_parts = [tc.code for tc in test_cases]
        test_code_str = textwrap.dedent(
            imports.rstrip() + "\n\n" + "\n\n".join(p.rstrip() for p in body_parts)
        ).strip()

        return UnitTestGeneratorOutput(
            function_name=func_name,
            language=language,
            framework=framework,
            imports=imports,
            test_cases=test_cases,
            coverage_note=coverage_note,
            test_code=test_code_str,
            test_count=len(test_cases),
        )

    # ========================================================================
    # 输入：Python 代码字符串
    # 输出：包含函数信息的 dict（name/params/has_return/is_async/is_method）
    # 逻辑：
    # 用ast.parse()解析代码为 AST 树
    # 遍历 AST 找到FunctionDef/AsyncFunctionDef节点
    # 提取参数：跳过self/cls（标记为方法），用_extract_type_hint_from_annotation()提取类型提示
    # 检测 return：遍历 AST 节点看是否有Return节点
    # 检测 async：判断节点是否是AsyncFunctionDef
    # ========================================================================
    def _parse_python_ast(self, code: str) -> dict:
        """
        Use AST to accurately parse Python function information.
        Covers: function name, params (with type hints), return, async, decorators,
        instance/class method detection (self/cls).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.debug("AST parse failed, falling back to regex")
            return {
                "name": "unknown_function", "params": [],
                "has_return": False, "is_async": False, "is_method": False,
            }

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = node
                break

        if not func_node:
            return {
                "name": "unknown_function", "params": [],
                "has_return": False, "is_async": False, "is_method": False,
            }

        # Extract params (skip self/cls for method detection)
        params: list[ParameterInfo] = []
        is_method = False
        for arg in func_node.args.args:
            if arg.arg in ("self", "cls"):
                is_method = True
                continue
            type_hint = self._extract_type_hint_from_annotation(arg.annotation)
            params.append(ParameterInfo(name=arg.arg, type_hint=type_hint))

        # Check for return
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(func_node))

        # Check async
        is_async = isinstance(func_node, ast.AsyncFunctionDef)

        return {
            "name": func_node.name,
            "params": params,
            "has_return": has_return,
            "is_async": is_async,
            "is_method": is_method,
        }

    @staticmethod
    def _extract_type_hint_from_annotation(annotation) -> str:
        """Extract type hint string from AST annotation node."""
        if annotation is None:
            return "any"
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return "NoneType"
        if isinstance(annotation, ast.Subscript):
            # e.g. list[int], dict[str, int], Optional[int]
            base = ""
            if isinstance(annotation.value, ast.Name):
                base = annotation.value.id.lower()
            elif isinstance(annotation.value, ast.Attribute):
                base = annotation.value.attr.lower()
            if base in ("list", "set", "frozenset", "tuple"):
                return "list"
            if base in ("dict", "defaultdict", "ordereddict"):
                return "dict"
            if base in ("optional",):
                return "any"
            return base or "any"
        if isinstance(annotation, ast.BinOp):
            # Python 3.10+ unions: int | float
            if isinstance(annotation.op, ast.BitOr):
                if isinstance(annotation.left, ast.Name):
                    return annotation.left.id
            return "any"
        if isinstance(annotation, ast.Attribute):
            return annotation.attr.lower()
        # Fallback for other complex types
        return "any"

    # ========================================================================
    # Fallback Extraction (non-Python languages)类型提示提取）：
    # ========================================================================
    def _extract_function_name_fallback(self, code: str, language: str) -> str:
        """提取函数/方法名称（非 Python 回退）"""
        patterns = {
            "javascript": _RE_FUNC_JS,
            "java": _RE_FUNC_JAVA,
            "go": _RE_FUNC_GO,
        }
        pattern = patterns.get(language)
        if pattern is None:
            # Generic fallback
            match = re.search(r'(\w+)\s*\(', code)
            return match.group(1) if match else "unknown_function"

        match = pattern.search(code)
        if match:
            for g in match.groups():
                if g is not None:
                    return g
        return "unknown_function"

    #（非 Python 参数提取）：
    def _extract_params_fallback(self, code: str, language: str) -> list[ParameterInfo]:
        """
        Extract function parameters for non-Python languages.
        P0-8: Uses bracket-counting for balanced parentheses.
        P0-7: Go name/type order fixed.
        P1-16: JS destructuring support.
        """
        params_str = self._extract_params_text(code)
        if not params_str or not params_str.strip():
            return []

        params: list[ParameterInfo] = []
        for p in self._split_top_level_params(params_str):
            p = p.strip()
            if not p:
                continue
            if language == "javascript":
                # P1-16: handle destructuring
                if _RE_PARAM_JS_DESTRUCTURE.search(p):
                    name_match = re.match(r'\s*\{([^}]+)\}', p)
                    if name_match:
                        params.append(ParameterInfo(name="options", type_hint="object"))
                    else:
                        params.append(ParameterInfo(name="param", type_hint="any"))
                    continue
                name_match = _RE_PARAM_NAME.match(p)
                if name_match:
                    params.append(ParameterInfo(
                        name=name_match.group(1), type_hint="any",
                    ))
            elif language == "java":
                parts = p.split()
                if len(parts) >= 2:
                    # Correct: type then name
                    params.append(ParameterInfo(
                        name=parts[-1].rstrip(","),
                        type_hint=parts[-2],
                    ))
                else:
                    params.append(ParameterInfo(name=p, type_hint="Object"))
            elif language == "go":
                parts = p.split()
                if len(parts) >= 2:
                    # P0-7 fix: Go is "name type", not "type name"
                    params.append(ParameterInfo(
                        name=parts[0],
                        type_hint=parts[1].rstrip(","),
                    ))
                else:
                    params.append(ParameterInfo(name=p, type_hint="interface{}"))

        return params

    # 参数提取（非 Python 回退）：
    @staticmethod
    def _extract_params_text(code: str) -> str:
        """
        P0-8: Extract the text between the outermost parentheses
        of a function signature, handling nested parentheses.
        """
        # Find the first '('
        start = code.find("(")
        if start == -1:
            return ""
        depth = 0
        for i in range(start, len(code)):
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
                if depth == 0:
                    return code[start + 1 : i]
        return ""

    # 参数分割（非 Python 回退）：
    @staticmethod
    def _split_top_level_params(params_str: str) -> list[str]:
        """Split comma-separated parameters, respecting nested brackets."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in params_str:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    # ========================================================================
    #语言自动检测 Language Detection (P2-18: 去除冗余 import re)
    # ========================================================================
    @staticmethod
    def _detect_language(code: str) -> str:
        """Auto-detect programming language using refined patterns."""
        if _RE_LANG_PYTHON.search(code):
            return "python"
        if _RE_LANG_GO.search(code):
            return "go"
        if _RE_LANG_JAVA.search(code):
            return "java"
        if _RE_LANG_JS.search(code):
            return "javascript"
        return "python"  # default

    # ========================================================================
    # Debug Representation (P3-19)
    # ========================================================================
    def __repr__(self) -> str:
        langs = list(_STRATEGY_REGISTRY.keys())
        return (
            f"UnitTestGeneratorSkill(version={self.version!r}, "
            f"languages={langs})"
        )


# ============================================================================
# 便捷接口层 generate_unit_tests()
# 输入：code（函数代码）、language（语言）、include_edge_cases/include_exception_cases（开关）
# 输出：UnitTestGeneratorOutput
# 逻辑：封装UnitTestGeneratorSkill的初始化和调用，提供一键使用的便捷接口
# ============================================================================
def generate_unit_tests(
    code: str,
    language: SupportedLanguage | str = SupportedLanguage.PYTHON,
    include_edge_cases: bool = True,
    include_exception_cases: bool = True,
) -> UnitTestGeneratorOutput:
    """
    Convenience function: one-liner to generate unit tests.

    Args:
        code: Function/method source code
        language: Programming language (auto for detection)
        include_edge_cases: Whether to include edge case tests
        include_exception_cases: Whether to include exception tests

    Returns:
        UnitTestGeneratorOutput
    """
    skill = UnitTestGeneratorSkill()
    return skill.execute(UnitTestGeneratorInput(
        code=code,
        language=SupportedLanguage(language) if isinstance(language, str) else language,
        include_edge_cases=include_edge_cases,
        include_exception_cases=include_exception_cases,
    ))
# ============================================================================
# generate_unit_tests()
#   [便捷入口函数]
#   功能：封装技能类初始化与调用流程，提供一键生成接口
#   输入：code(函数代码), language(编程语言), include_edge_cases(边界用例开关), include_exception_cases(异常用例开关)
#   输出：UnitTestGeneratorOutput(完整测试生成结果)
#   逻辑：初始化UnitTestGeneratorSkill → 构造UnitTestGeneratorInput → 调用skill.execute()
#   ↓
# UnitTestGeneratorSkill.execute()
#   [异常处理包装层]
#   功能：捕获并处理执行过程中的异常，保证友好错误输出
#   输入：UnitTestGeneratorInput
#   输出：UnitTestGeneratorOutput
#   逻辑：
#     1. 尝试调用 _execute_inner()
#     2. 若捕获 SyntaxError：返回含语法错误位置的友好输出
#     3. 若捕获其他异常：返回含通用错误信息的输出
#   ↓
# UnitTestGeneratorSkill._execute_inner()
#   [核心业务流程控制器]
#   功能：协调整个测试生成的全流程
#   输入：UnitTestGeneratorInput
#   输出：UnitTestGeneratorOutput
#   逻辑：
#     ├→ 1. 代码预处理：去除输入代码首尾空白字符
#     ├→ 2. 语言检测与确认
#     │   └→ _detect_language()  [仅当 language=auto 时调用]
#     │        [自动语言识别器]
#     │        功能：通过函数签名特征自动判断编程语言
#     │        输入：code(预处理后的代码)
#     │        输出：language(识别出的语言字符串，默认python)
#     │        逻辑：按优先级匹配正则 (Python→Go→Java→JS)，无匹配则回退python
#     ├→ 3. 函数信息解析
#     │   ├→ [分支1：language=python]
#     │   │  └→ _parse_python_ast()
#     │   │       [Python AST精准解析器]
#     │   │       功能：通过抽象语法树精准提取Python函数的元信息
#     │   │       输入：code(Python代码)
#     │   │       输出：dict(包含name/params/has_return/is_async/is_method)
#     │   │       逻辑：
#     │   │         1. ast.parse() 将代码解析为AST树
#     │   │         2. 遍历AST找到 FunctionDef/AsyncFunctionDef 节点
#     │   │         3. 提取参数：跳过self/cls(标记is_method=True)，调用 _extract_type_hint_from_annotation() 提取类型
#     │   │         4. 检测return：遍历AST节点判断是否存在 Return 节点
#     │   │         5. 检测async：判断节点是否为 AsyncFunctionDef 类型
#     │   │         └→ _extract_type_hint_from_annotation()
#     │   │              [类型提示提取器]
#     │   │              功能：从AST注解节点中提取简化的类型字符串
#     │   │              输入：annotation(AST注解节点)
#     │   │              输出：type_hint(简化类型字符串，如"int"/"list")
#     │   │              逻辑：处理 Name/Constant/Subscript/BinOp/Attribute 等节点，将复杂类型(如list[int])简化为基础类型
#     │   │
#     │   └→ [分支2：language≠python]
#     │      ├→ _extract_function_name_fallback()
#     │      │    [非Python函数名提取器]
#     │      │    功能：通过正则回退提取非Python语言的函数名
#     │      │    输入：code, language
#     │      │    输出：func_name(函数名字符串)
#     │      │    逻辑：根据语言选择对应正则(JS/Java/Go)匹配函数签名
#     │      ├→ _extract_params_fallback()
#     │      │    [非Python参数提取器]
#     │      │    功能：通过正则+括号计数提取非Python语言的函数参数
#     │      │    输入：code, language
#     │      │    输出：list[ParameterInfo](参数信息列表)
#     │      │    逻辑：
#     │      │      1. _extract_params_text()：提取最外层括号内的参数字符串(处理嵌套括号)
#     │      │      2. _split_top_level_params()：按逗号分割顶层参数(忽略嵌套括号内的逗号)
#     │      │      3. 根据语言定制解析：JS处理解构参数、Java按"类型 名称"解析、Go按"名称 类型"解析
#     │      └→ 正则检测return：通过 _RE_RETURN 正则判断代码中是否包含return语句
#     │
#     ├→ 4. 获取语言策略
#     │   └→ _get_strategy()
#     │        [策略工厂]
#     │        功能：根据语言返回对应的测试生成策略实例
#     │        输入：language(编程语言)
#     │        输出：TestGeneratorStrategy(具体策略类实例，如PythonTestStrategy)
#     │        逻辑：查询 _STRATEGY_REGISTRY 字典，无匹配则回退Python策略
#     │
#     ├→ 5. 生成测试导入语句
#     │   └→ [策略类方法] PythonTestStrategy.generate_imports()
#     │        [导入语句生成器]
#     │        功能：生成对应语言测试框架的导入语句
#     │        输入：func_name(函数名), is_method(是否为类方法)
#     │        输出：imports(导入语句字符串)
#     │        逻辑：
#     │          - 普通函数：生成 pytest 导入 + 被测函数导入
#     │          - 类方法：生成 pytest 导入 + 类导入 + 实例化示例注释
#     │
#     ├→ 6. 生成正常路径测试用例
#     │   └→ [策略类方法] PythonTestStrategy.generate_happy_path()
#     │        [正常用例生成器]
#     │        功能：生成验证基本功能的测试用例
#     │        输入：func_name, params(参数列表), has_return(是否有返回值), is_method, is_async(是否异步)
#     │        输出：list[RawTestCase](原始测试用例列表)
#     │        逻辑：
#     │          1. make_sample_args()：根据 SAMPLE_MAP 生成示例参数(如int→42)
#     │          2. 处理async：函数前加 async，调用前加 await
#     │          3. 处理方法调用：添加 obj. 前缀，插入类实例化代码
#     │          4. 生成断言：有return则断言非None，否则注释验证副作用
#     │        └→ [策略基类公共方法] make_sample_args()
#     │             [示例参数生成器]
#     │             功能：根据参数类型提示生成示例参数值
#     │             输入：params(参数列表)
#     │             输出：sample_args(示例参数字符串，如"42, 'hello'")
#     │             逻辑：遍历参数，查询 SAMPLE_MAP 获取对应类型的示例值
#     │
#     ├→ 7. 生成边界情况测试用例 [仅当 include_edge_cases=True 时调用]
#     │   └→ [策略类方法] PythonTestStrategy.generate_edge_cases()
#     │        [边界用例生成器]
#     │        功能：生成验证边界条件的测试用例
#     │        输入：func_name, params, has_return, is_method
#     │        输出：list[RawTestCase]
#     │        逻辑：
#     │          1. 空值/零值测试：_make_zero_args() 生成零值参数(如int→0)，调用函数并留空断言待填
#     │          2. 大输入测试：生成大体积数据(如长字符串、大列表)，调用函数并断言非None
#     │        └→ [策略基类公共方法] _make_zero_args()
#     │             [零值参数生成器]
#     │             功能：根据参数类型提示生成零值/空值参数
#     │             输入：params(参数列表)
#     │             输出：zero_args(零值参数字符串，如"0, ''")
#     │             逻辑：遍历参数，查询 ZERO_MAP 获取对应类型的零值
#     │
#     ├→ 8. 生成异常路径测试用例 [仅当 include_exception_cases=True 时调用]
#     │   └→ [策略类方法] PythonTestStrategy.generate_exception_cases()
#     │        [异常用例生成器]
#     │        功能：生成验证错误处理的测试用例
#     │        输入：func_name, params, is_method
#     │        输出：list[RawTestCase]
#     │        逻辑：
#     │          1. 构造错误类型参数：如int类型传字符串、str类型传数字
#     │          2. 生成测试代码：使用 pytest.raises 捕获 TypeError/ValueError
#     │
#     └→ 9. 组装最终输出
#          功能：将所有生成的内容组装为标准化输出模型
#          逻辑：
#            1. 统计用例数量：分类统计 happy_path/edge_case/exception 用例数
#            2. 生成覆盖率说明：提示需补充Mock、集成测试、性能测试
#            3. 格式化完整测试代码：用 textwrap 整理 imports + 所有测试用例代码
#            4. 构造并返回 UnitTestGeneratorOutput
# ============================================================================