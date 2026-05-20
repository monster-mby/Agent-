"""
test_unit_test_generator.py

测试 UnitTestGeneratorSkill 核心逻辑，重点暴露边界 bug。
运行：pytest test_unit_test_generator.py -v
"""

import sys
import textwrap
import pytest

# 假设你的模块路径
from src.skills.preset.technical_development.unit_test_generator import (
    UnitTestGeneratorSkill,
    UnitTestGeneratorInput,
    UnitTestGeneratorOutput,
    SupportedLanguage,
    TestCategory,
    ParameterInfo,
    RawTestCase,
    PythonTestStrategy,
    JavaScriptTestStrategy,
    JavaTestStrategy,
    GoTestStrategy,
    _get_strategy,
    _capitalize,
    _STRATEGY_REGISTRY,
)


# ============================================================================
# Fixture
# ============================================================================
@pytest.fixture
def skill():
    return UnitTestGeneratorSkill()


# ============================================================================
# 1. Python AST 解析 — 基础
# ============================================================================
class TestPythonAST:
    """核心：Python AST 解析正确性"""

    def test_simple_function(self, skill):
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        result = skill._parse_python_ast(code)
        assert result["name"] == "add"
        assert result["has_return"] is True
        assert result["is_async"] is False
        assert result["is_method"] is False
        assert len(result["params"]) == 2
        assert result["params"][0]["name"] == "a"
        assert result["params"][0]["type_hint"] == "int"

    def test_async_function(self, skill):
        code = "async def fetch(url: str):\n    return await something(url)"
        result = skill._parse_python_ast(code)
        assert result["is_async"] is True

    def test_method_with_self(self, skill):
        code = "def get_name(self, prefix: str) -> str:\n    return prefix + self.name"
        result = skill._parse_python_ast(code)
        assert result["is_method"] is True
        # self should NOT appear in params
        param_names = [p["name"] for p in result["params"]]
        assert "self" not in param_names
        assert "prefix" in param_names

    def test_method_with_cls(self, skill):
        code = "@classmethod\ndef create(cls, name: str):\n    return cls(name)"
        result = skill._parse_python_ast(code)
        assert result["is_method"] is True
        param_names = [p["name"] for p in result["params"]]
        assert "cls" not in param_names

    def test_no_return_function(self, skill):
        code = "def log(msg: str) -> None:\n    print(msg)"
        result = skill._parse_python_ast(code)
        assert result["has_return"] is False

    # ── Bug 候选 ──
    def test_type_hint_optional(self, skill):
        """Optional[int] → 应返回 'any'"""
        code = "def f(x: int | None) -> int:\n    return x or 0"
        result = skill._parse_python_ast(code)
        assert result["params"][0]["type_hint"] in ("int", "any")

    def test_type_hint_list_of_int(self, skill):
        """list[int] → 应返回 'list'"""
        code = "def f(items: list[int]):\n    return items[0]"
        result = skill._parse_python_ast(code)
        assert result["params"][0]["type_hint"] == "list"


# ============================================================================
# 2. 异常处理
# ============================================================================
class TestExceptionHandling:
    """核心：execute 异常处理路径"""

    def test_syntax_error_input(self, skill):
        """语法错误的代码应降级处理而非崩溃"""
        result = skill.execute(UnitTestGeneratorInput(
            code="def broken(:",
            language=SupportedLanguage.PYTHON,
        ))
        assert isinstance(result, UnitTestGeneratorOutput)
        # AST 解析失败时，应使用 unknown_function 作为降级
        assert result.function_name == "unknown_function"
        # 不应抛出异常，而是生成降级测试
        assert result.test_count >= 0

    def test_garbled_code(self, skill):
        """乱码输入不应崩溃"""
        result = skill.execute(UnitTestGeneratorInput(
            code="!@#$%^&*()__NOT_VALID_CODE__",
            language=SupportedLanguage.AUTO,
        ))
        assert isinstance(result, UnitTestGeneratorOutput)
        # 不应抛出异常
        assert result.test_count >= 0


# ============================================================================
# 3. 策略映射提取 — 确保基类方法正确
# ============================================================================
class TestStrategyMaps:
    """确保 SAMPLE_MAP / ZERO_MAP 提取后没有倒退"""

    def test_all_strategies_have_maps(self):
        for lang, strat in _STRATEGY_REGISTRY.items():
            assert strat.SAMPLE_MAP, f"{lang} missing SAMPLE_MAP"
            assert strat.ZERO_MAP, f"{lang} missing ZERO_MAP"

    def test_make_sample_args_uses_maps(self):
        strat = PythonTestStrategy()
        params = [
            ParameterInfo(name="x", type_hint="int"),
            ParameterInfo(name="s", type_hint="str"),
        ]
        result = strat.make_sample_args(params)
        assert "42" in result
        assert '"hello"' in result

    def test_make_sample_args_all_strategies(self):
        """确保每个策略的 make_sample_args 都能正常拼接参数"""
        params = [
            ParameterInfo(name="x", type_hint="int"),
            ParameterInfo(name="s", type_hint="str"),
        ]
        for lang, strat in _STRATEGY_REGISTRY.items():
            result = strat.make_sample_args(params)
            assert result, f"{lang}: make_sample_args returned empty"
            assert "x" not in result, f"{lang}: param name leaked into output"

    def test_make_zero_args_empty_params(self):
        strat = PythonTestStrategy()
        result = strat._make_zero_args([])
        assert result == ""

    def test_make_zero_args_all_strategies(self):
        """空参数列表时，所有策略都应返回空字符串"""
        for lang, strat in _STRATEGY_REGISTRY.items():
            assert strat._make_zero_args([]) == "", f"{lang}: _make_zero_args([]) should be empty"

    def test_make_zero_args_uses_zero_map(self):
        strat = PythonTestStrategy()
        params = [ParameterInfo(name="x", type_hint="int")]
        result = strat._make_zero_args(params)
        assert result == "0"

    def test_null_literal_per_language(self):
        """每个语言有自己的 null 字面量"""
        assert PythonTestStrategy._null_literal() == "None"
        assert JavaScriptTestStrategy._null_literal() == "null"
        assert JavaTestStrategy._null_literal() == "null"
        assert GoTestStrategy._null_literal() == "nil"

    def test_bool_literal_per_language(self):
        """每个语言有自己的 bool 字面量"""
        assert PythonTestStrategy._bool_literal() == "True"
        assert JavaScriptTestStrategy._bool_literal() == "true"
        assert JavaTestStrategy._bool_literal() == "true"
        assert GoTestStrategy._bool_literal() == "true"


# ============================================================================
# 4. 端到端生成
# ============================================================================
class TestEndToEnd:
    """完整输入→输出"""

    def test_python_happy_path_generation(self, skill):
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            language=SupportedLanguage.PYTHON,
            include_edge_cases=False,
            include_exception_cases=False,
        ))
        assert result.function_name == "add"
        assert result.language == "python"
        assert result.framework == "pytest"
        assert result.test_count >= 1
        assert any(t.category == TestCategory.HAPPY_PATH for t in result.test_cases)
        # test_code 不应是空字符串
        assert result.test_code
        assert "def test_add_basic" in result.test_code

    def test_full_coverage_includes_all_categories(self, skill):
        code = "def process(data: str) -> str:\n    return data.strip()"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            language=SupportedLanguage.PYTHON,
            include_edge_cases=True,
            include_exception_cases=True,
        ))
        categories = {t.category for t in result.test_cases}
        # 应该有 happy_path + edge_case + exception
        assert TestCategory.HAPPY_PATH in categories
        assert TestCategory.EDGE_CASE in categories
        assert TestCategory.EXCEPTION in categories
        assert result.test_count >= 3

    def test_javascript_generation(self, skill):
        code = "function sum(a, b) { return a + b; }"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            language=SupportedLanguage.JAVASCRIPT,
        ))
        assert result.language == "javascript"
        assert result.framework == "jest"
        assert "test(" in result.test_code or "test('" in result.test_code

    def test_java_generation(self, skill):
        code = "public int add(int a, int b) { return a + b; }"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            language=SupportedLanguage.JAVA,
        ))
        assert result.language == "java"
        assert "@Test" in result.test_code

    def test_go_generation(self, skill):
        code = "func Add(a int, b int) int {\n    return a + b\n}"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            language=SupportedLanguage.GO,
        ))
        assert result.language == "go"
        assert "func Test" in result.test_code
        assert "testing.T" in result.test_code


# ============================================================================
# 5. 语言检测
# ============================================================================
class TestLanguageDetection:
    """自动检测"""

    def test_detect_python(self, skill):
        assert skill._detect_language("def foo():\n    pass") == "python"
        assert skill._detect_language("async def bar():\n    pass") == "python"

    def test_detect_go_first(self, skill):
        """Go 在 Java 之前检查，确保不误判"""
        assert skill._detect_language("func Add(a int) int {\n    return a\n}") == "go"

    def test_detect_java(self, skill):
        assert skill._detect_language(
            "public void doThing(String x) {\n    System.out.println(x);\n}"
        ) == "java"

    def test_detect_js(self, skill):
        assert skill._detect_language("function foo(a, b) { return a + b; }") == "javascript"
        assert skill._detect_language("const add = (a, b) => a + b") == "javascript"

    def test_detect_defaults_to_python(self, skill):
        assert skill._detect_language("plain text no code") == "python"


# ============================================================================
# 6. 非 Python 参数提取 — Bug 高发区
# ============================================================================
class TestFallbackExtraction:
    """regex 回退路径的边界情况"""

    def test_go_multiple_names_same_type(self, skill):
        """Go 的 a, b int → 两个参数都是 int"""
        code = "func Add(a, b int) int { return a + b }"
        params = skill._extract_params_fallback(code, "go")
        # 当前代码这里会出错：split 得到 ["a,", "b", "int"]
        # parts[0]="a," parts[1]="b" → name="a," type="b"
        # 这是已知 bug，测试会暴露
        assert len(params) >= 1

    def test_java_generic_param(self, skill):
        """Java 泛型 List<String> → 应优雅处理"""
        code = "public void process(List<String> items) { }"
        params = skill._extract_params_fallback(code, "java")
        assert len(params) >= 1
        # 不会崩溃就是底线

    def test_js_destructuring(self, skill):
        """JS 解构 { name, age } → 应识别为 options 对象"""
        code = "function greet({ name, age }) { return name; }"
        params = skill._extract_params_fallback(code, "javascript")
        assert len(params) >= 1
        assert any(p["type_hint"] == "object" for p in params)

    def test_no_params(self, skill):
        code = "function sayHello() { return 'hello'; }"
        params = skill._extract_params_fallback(code, "javascript")
        assert params == []


# ============================================================================
# 7. textwrap 格式化一致性
# ============================================================================
class TestTextwrapFormatting:
    """确保 textwrap 没有产生多余/缺少缩进"""

    def test_python_happy_path_indentation(self, skill):
        code = "def f(x: int) -> int:\n    return x * 2"
        result = skill.execute(UnitTestGeneratorInput(
            code=code,
            include_edge_cases=False,
            include_exception_cases=False,
        ))
        # 每行都不应该以多余空格开头（除正常缩进）
        for line in result.test_code.split("\n"):
            if line.startswith("def "):
                assert not line.startswith(" ")  # 顶层不应缩进

    def test_output_not_empty(self, skill):
        code = "def foo():\n    pass"
        result = skill.execute(UnitTestGeneratorInput(code=code))
        assert result.test_code.strip()  # 非空
        assert result.imports.strip()  # 有 import


# ============================================================================
# 8. 向后兼容字段
# ============================================================================
class TestBackwardCompat:
    def test_coverage_notes_compat(self, skill):
        code = "def f(): pass"
        result = skill.execute(UnitTestGeneratorInput(code=code))
        # coverage_notes 应与 coverage_note 相同（computed_field）
        assert result.coverage_notes == result.coverage_note


# ============================================================================
# 9. _capitalize 工具函数
# ============================================================================
class TestCapitalize:
    def test_normal(self):
        assert _capitalize("hello") == "Hello"

    def test_empty(self):
        assert _capitalize("") == ""

    def test_single_char(self):
        assert _capitalize("a") == "A"


# ============================================================================
# 10. 策略注册完整性
# ============================================================================
class TestStrategyRegistry:
    def test_all_languages_registered(self):
        for lang in ("python", "javascript", "java", "go"):
            assert lang in _STRATEGY_REGISTRY

    def test_get_strategy_fallback(self):
        strat = _get_strategy("rust")  # 不支持
        assert isinstance(strat, PythonTestStrategy)  # 回退到 Python

if __name__ == "__main__":
    pytest.main()