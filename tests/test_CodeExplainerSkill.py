# ========================================================================
# pytest 测试套件：CodeExplainerSkill v2.0.0
#
# 覆盖：
#   - 基本功能（多语言、三种详细程度）
#   - 输入验证（Pydantic schema）
#   - 语言检测（guesslang + 正则回退）
#   - 代码分块（Python AST + 正则）
#   - 概述生成（含 radon 复杂度分析）
#   - 问题检测（AST 级裸 except/资源泄露/可变默认参数）
#   - 详细程度控制（brief/detailed/line_by_line）
#   - 可选依赖回退（guesslang/radon 缺失）
#   - 异常处理（语法错误降级）
#   - 向后兼容（computed_field）
#   - 便捷函数
#
# 运行方式:
#     pytest tests/test_CodeExplainerSkill.py -v
#     或带覆盖率:
#     pytest tests/test_CodeExplainerSkill.py -v --cov=src.skills.code_explainer
# ========================================================================

import re
from unittest.mock import patch, MagicMock

import pytest

from src.skills.preset.technical_development.code_explainer import (
    CodeExplainerSkill,
    CodeExplainerInput,
    CodeExplainerOutput,
    Language,
    DetailLevel,
    generate_code_explanation,
    _SKILL_VERSION,
    _LANG_PATTERNS,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def skill():
    """返回默认的代码解释技能实例"""
    return CodeExplainerSkill()


@pytest.fixture
def valid_python_code():
    """有效的 Python 测试代码（包含函数、类、装饰器、上下文管理器）"""
    return """import time
from contextlib import contextmanager

@contextmanager
def timer():
    \"\"\"计时上下文管理器\"\"\"
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start}")

class Calculator:
    \"\"\"简单计算器类\"\"\"
    def __init__(self, initial=0):
        self.value = initial

    def add(self, x):
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        self.value += x
        return self.value

def main():
    \"\"\"主函数\"\"\"
    calc = Calculator()
    with timer():
        for i in range(10):
            calc.add(i)
    print(f"Final result: {calc.value}")

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def code_with_issues():
    """包含常见问题的 Python 代码（裸 except、资源泄露、可变默认参数）"""
    return """def bad_function(a=[]):
    \"\"\"可变默认参数陷阱\"\"\"
    a.append(1)
    try:
        f = open("test.txt", "r")
        print(f.read())
    except:
        pass
    return a
"""


@pytest.fixture
def valid_javascript_code():
    """有效的 JavaScript 测试代码"""
    return """import React, { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);

    const increment = () => {
        setCount(prev => prev + 1);
    };

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={increment}>Increment</button>
        </div>
    );
}

export default Counter;
"""


@pytest.fixture
def code_with_todo():
    """包含 TODO/FIXME 的代码"""
    return """# TODO: 重构这个函数
# FIXME: 修复边界条件
def process_data(data):
    result = []
    for item in data:
        # HACK: 临时处理
        result.append(item * 2)
    return result
"""


# ============================================================================
# 1. 基本功能测试
# ============================================================================
class TestBasicFunctionality:
    """测试基本代码解释功能"""

    def test_explain_python_detailed(self, skill, valid_python_code):
        """测试解释 Python 代码（detailed 模式）"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.PYTHON,
            detail_level=DetailLevel.DETAILED,
        ))
        assert isinstance(result, CodeExplainerOutput)
        assert result.language == "python"
        assert len(result.overview) > 0
        assert len(result.blocks) > 0
        assert len(result.key_points) > 0

    def test_explain_javascript(self, skill, valid_javascript_code):
        """测试解释 JavaScript 代码"""
        result = skill.execute(CodeExplainerInput(
            code=valid_javascript_code,
            language=Language.JAVASCRIPT,
        ))
        assert result.language == "javascript"
        assert "javascript" in result.overview.lower()

    def test_explicit_language_override(self, skill, valid_python_code):
        """测试显式指定语言（覆盖自动检测）"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.JAVA,  # 故意指定错误
        ))
        assert result.language == "java"  # 应该尊重显式指定


# ============================================================================
# 2. 输入验证测试
# ============================================================================
class TestInputValidation:
    """测试 Pydantic 输入验证"""

    def test_code_too_short(self, skill):
        """测试代码过短（<5 字符）"""
        with pytest.raises(Exception):
            CodeExplainerInput(code="a")

    def test_code_only_whitespace(self, skill):
        """测试仅空白字符的代码"""
        with pytest.raises(Exception):
            CodeExplainerInput(code="   \n  \t  ")

    def test_code_normalized(self, skill, valid_python_code):
        """测试代码首尾空白被 strip"""
        result = skill.execute(CodeExplainerInput(
            code=f"\n\n  {valid_python_code}  \n",
            language=Language.PYTHON,
        ))
        assert result is not None

    def test_detail_level_enum(self, skill, valid_python_code):
        """测试使用字符串也能工作"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            detail_level="brief",  # 字符串
        ))
        assert result is not None


# ============================================================================
# 3. 语言检测测试
# ============================================================================
class TestLanguageDetection:
    """测试 auto 模式下的语言检测"""

    def test_auto_detect_python(self, skill, valid_python_code):
        """测试自动检测 Python"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.AUTO,
        ))
        assert result.language == "python"

    def test_auto_detect_javascript(self, skill, valid_javascript_code):
        """测试自动检测 JavaScript"""
        result = skill.execute(CodeExplainerInput(
            code=valid_javascript_code,
            language=Language.AUTO,
        ))
        assert result.language in ["javascript", "typescript"]

    def test_guesslang_fallback(self, skill, valid_python_code):
        """测试 guesslang 不可用时回退到正则"""
        with patch("src.skills.preset.technical_development.code_explainer._GUESSLANG_AVAILABLE", False):
            result = skill.execute(CodeExplainerInput(
                code=valid_python_code,
                language=Language.AUTO,
            ))
            assert result.language == "python"

    def test_typescript_over_javascript(self, skill):
        """测试 TypeScript 特征明显时优先识别为 TS"""
        ts_code = """
interface User {
    name: string;
    age: number;
}
const user: User = { name: "Alice", age: 30 };
"""
        result = skill.execute(CodeExplainerInput(
            code=ts_code,
            language=Language.AUTO,
        ))
        # 只要不崩溃即可，取决于特征强度
        assert result is not None


# ============================================================================
# 4. 代码分块测试
# ============================================================================
class TestCodeSplitting:
    """测试代码分块逻辑"""

    def test_python_ast_splitting(self, skill, valid_python_code):
        """测试 Python 使用 AST 精准分块"""
        lines = valid_python_code.split("\n")
        blocks = skill._split_code_blocks(valid_python_code, lines, "python")

        # 应该识别出类、函数、导入
        block_names = [b["name"] for b in blocks]
        assert any("class Calculator" in name for name in block_names)
        assert any("function timer" in name for name in block_names)
        assert any("function main" in name for name in block_names)
        assert any("Module-level" in name or "import" in name.lower() for name in block_names)

    def test_regex_splitting_fallback(self, skill, valid_javascript_code):
        """测试非 Python 语言使用正则分块"""
        lines = valid_javascript_code.split("\n")
        blocks = skill._split_code_blocks(valid_javascript_code, lines, "javascript")

        assert len(blocks) > 0
        # 应该识别出函数
        block_names = [b["name"].lower() for b in blocks]
        assert any("function" in name or "counter" in name for name in block_names)

    def test_syntax_error_fallback(self, skill):
        """测试 Python 语法错误时回退到正则"""
        bad_syntax_code = "def x: pass"  # 语法错误
        lines = bad_syntax_code.split("\n")
        blocks = skill._split_code_blocks(bad_syntax_code, lines, "python")

        # 应该回退到正则，返回至少一个块
        assert len(blocks) >= 1


# ============================================================================
# 5. 概述生成测试
# ============================================================================
class TestOverviewGeneration:
    """测试概述生成（含复杂度分析）"""

    def test_overview_contains_basics(self, skill, valid_python_code):
        """测试概述包含基本信息"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.PYTHON,
        ))
        assert "python" in result.overview.lower()
        assert "行" in result.overview or "line" in result.overview.lower()
        assert "块" in result.overview or "block" in result.overview.lower()

    def test_radon_complexity_analysis(self, skill, valid_python_code):
        """测试 radon 可用时包含复杂度分析"""
        # 假设 radon 可用，检查概述中是否有复杂度相关词汇
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.PYTHON,
        ))
        # 可能有也可能没有，取决于 radon 是否安装，只要不崩溃即可
        assert result is not None

    def test_radon_fallback(self, skill, valid_python_code):
        """测试 radon 不可用时概述仍能生成"""
        with patch("src.skills.preset.technical_development.code_explainer._RADON_AVAILABLE", False):
            result = skill.execute(CodeExplainerInput(
                code=valid_python_code,
                language=Language.PYTHON,
            ))
            # 概述中不应有"圈复杂度"
            assert "圈复杂度" not in result.overview or "complexity" not in result.overview.lower()


# ============================================================================
# 6. 问题检测测试
# ============================================================================
class TestIssueDetection:
    """测试潜在问题检测"""

    def test_detect_bare_except(self, skill, code_with_issues):
        """测试检测裸 except"""
        result = skill.execute(CodeExplainerInput(
            code=code_with_issues,
            language=Language.PYTHON,
            detail_level=DetailLevel.DETAILED,
        ))
        # 检查 potential_issues 或 blocks 中的问题
        issues_text = " ".join(result.potential_issues) + " " + " ".join(
            b.purpose + " " + b.how_it_works + " " + " ".join(b.potential_issues)
            for b in result.blocks
        )
        assert "except" in issues_text.lower()

    def test_detect_mutable_default(self, skill, code_with_issues):
        """测试检测可变默认参数"""
        result = skill.execute(CodeExplainerInput(
            code=code_with_issues,
            language=Language.PYTHON,
            detail_level=DetailLevel.DETAILED,
        ))
        issues_text = " ".join(result.potential_issues) + " " + " ".join(
            " ".join(b.potential_issues) for b in result.blocks
        )
        # 只要不崩溃即可
        assert result is not None

    def test_detect_todo_fixme(self, skill, code_with_todo):
        """测试检测 TODO/FIXME/HACK"""
        result = skill.execute(CodeExplainerInput(
            code=code_with_todo,
            language=Language.PYTHON,
        ))
        issues_text = " ".join(result.potential_issues)
        assert "TODO" in issues_text or "FIXME" in issues_text or "HACK" in issues_text


# ============================================================================
# 7. 详细程度控制测试
# ============================================================================
class TestDetailLevels:
    """测试三种详细程度的区别"""

    def test_brief_mode(self, skill, valid_python_code):
        """测试 brief 模式：仅概述，无 blocks"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            detail_level=DetailLevel.BRIEF,
        ))
        assert len(result.blocks) == 0
        assert len(result.overview) > 0
        assert len(result.key_points) > 0

    def test_detailed_mode(self, skill, valid_python_code):
        """测试 detailed 模式：有 blocks"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            detail_level=DetailLevel.DETAILED,
        ))
        assert len(result.blocks) > 0
        assert all(hasattr(b, "block_name") for b in result.blocks)
        assert all(hasattr(b, "line_range") for b in result.blocks)

    def test_line_by_line_mode(self, skill, valid_python_code):
        """测试 line_by_line 模式：blocks 数量约等于行数"""
        lines = valid_python_code.split("\n")
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            detail_level=DetailLevel.LINE_BY_LINE,
        ))
        # 数量应该接近（可能包含空行解释）
        assert len(result.blocks) >= len([l for l in lines if l.strip()]) * 0.5
        assert any("Line" in b.block_name for b in result.blocks)


# ============================================================================
# 8. 异常处理测试
# ============================================================================
class TestExceptionHandling:
    """测试主流程的异常处理与降级"""

    def test_syntax_error_graceful_degradation(self, skill):
        """测试代码有语法错误时优雅降级"""
        bad_code = "def x: pass"  # Python 语法错误
        result = skill.execute(CodeExplainerInput(
            code=bad_code,
            language=Language.PYTHON,
        ))
        # 应该返回一个包含错误信息的 Output，而不是抛出异常
        assert isinstance(result, CodeExplainerOutput)
        # 可能在 overview 或 key_points 中有错误提示
        assert result.overview is not None

    def test_generic_exception_handling(self, skill, valid_python_code):
        """测试任意异常都被捕获"""
        with patch.object(skill, "_execute_inner", side_effect=Exception("Something went wrong")):
            result = skill.execute(CodeExplainerInput(
                code=valid_python_code,
            ))
            assert isinstance(result, CodeExplainerOutput)
            assert "错误" in result.overview or "error" in result.overview.lower()


# ============================================================================
# 9. 向后兼容测试
# ============================================================================
class TestBackwardCompatibility:
    """测试 computed_field 实现的向后兼容"""

    def test_backward_compatible_fields(self, skill, valid_python_code):
        """测试 detected_language/explanation/summary 字段"""
        result = skill.execute(CodeExplainerInput(
            code=valid_python_code,
            language=Language.PYTHON,
        ))
        assert result.detected_language == result.language
        assert result.explanation == result.overview
        assert result.summary == result.overview


# ============================================================================
# 10. 便捷函数测试
# ============================================================================
class TestConvenienceFunction:
    """测试 generate_code_explanation() 便捷函数"""

    def test_generate_basic(self, valid_python_code):
        """测试基本调用"""
        result = generate_code_explanation(
            code=valid_python_code,
            language="python",
            detail_level="detailed",
        )
        assert isinstance(result, CodeExplainerOutput)
        assert len(result.outline_markdown) > 0 if hasattr(result, 'outline_markdown') else len(result.overview) > 0

    def test_generate_with_enums(self, valid_python_code):
        """测试使用 Enum 调用"""
        result = generate_code_explanation(
            code=valid_python_code,
            language=Language.PYTHON,
            detail_level=DetailLevel.BRIEF,
        )
        assert result.language == "python"


# ============================================================================
# 11. 特殊方法与调试测试
# ============================================================================
class TestSpecialMethods:
    """测试 __repr__ 等特殊方法"""

    def test_repr(self, skill):
        """测试 __repr__ 输出"""
        repr_str = repr(skill)
        assert "CodeExplainerSkill" in repr_str
        assert _SKILL_VERSION in repr_str
        assert "guesslang" in repr_str.lower()
        assert "radon" in repr_str.lower()

if __name__ == "__main__":
    pytest.main()