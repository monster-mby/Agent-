from sympy.testing import pytest

from skills.custom.code_skills.code_review.skill import CodeReviewSkill, CodeReviewInput, CodeReviewOutput


class TestCodeReviewSkill:

    def setup_method(self):
        self.skill = CodeReviewSkill()

    # ========== 基本功能测试 ==========

    def test_skill_name_and_description(self):
        assert self.skill.name == "code_review"
        assert self.skill.description == "审查代码质量，返回 issues / summary / score"

    def test_schemas_are_pydantic_models(self):
        assert issubclass(self.skill.input_schema, CodeReviewInput)
        assert issubclass(self.skill.output_schema, CodeReviewOutput)

    def test_execute_returns_output_model(self):
        result = self.skill.execute(CodeReviewInput(code="x = 1", language="python"))
        assert isinstance(result, CodeReviewOutput)

    # ========== 空文件测试 ==========

    def test_empty_code_returns_error(self):
        result = self.skill.execute(CodeReviewInput(code="", language="python"))
        assert len(result.issues) == 1
        assert result.issues[0].severity == "error"
        assert result.score == 0
        assert "空" in result.summary

    def test_whitespace_only_code(self):
        result = self.skill.execute(CodeReviewInput(code="   \n  \n  ", language="python"))
        assert result.score == 0

    # ========== 行长度检查 ==========

    def test_long_line_warning(self):
        long_line = "x" * 150
        result = self.skill.execute(CodeReviewInput(code=long_line, language="python"))
        warnings = [i for i in result.issues if i.severity == "warning" and "长度超过" in i.message]
        assert len(warnings) >= 1
        assert result.score < 100

    def test_short_line_no_warning(self):
        result = self.skill.execute(CodeReviewInput(code="x = 1", language="python"))
        long_line_issues = [i for i in result.issues if "长度超过" in i.message]
        assert len(long_line_issues) == 0

    # ========== 行尾空白检查 ==========

    def test_trailing_whitespace_detected(self):
        code = "x = 1   \ny = 2"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        trailing = [i for i in result.issues if "行尾有多余空白" in i.message]
        assert len(trailing) >= 1

    # ========== TODO 检查 ==========

    def test_todo_detected(self):
        code = "# TODO: refactor this\nx = 1\n// TODO: fix later"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        todos = [i for i in result.issues if "TODO" in i.message]
        assert len(todos) >= 1

    def test_no_todo_no_issue(self):
        code = "# this is fine\nx = 1"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        todos = [i for i in result.issues if "TODO" in i.message]
        assert len(todos) == 0

    # ========== Python 专项检查 ==========

    def test_python_function_naming_error(self):
        code = "def BadName():\n    pass"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        errors = [i for i in result.issues if "函数名应使用小写" in i.message]
        assert len(errors) >= 1

    def test_python_class_naming_error(self):
        code = "class myclass:\n    pass"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        errors = [i for i in result.issues if "类名应使用大写驼峰" in i.message]
        assert len(errors) >= 1

    def test_python_correct_naming_no_error(self):
        code = "def good_name():\n    pass\n\nclass GoodClass:\n    pass"
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        func_naming = [i for i in result.issues if "函数名应使用小写" in i.message]
        class_naming = [i for i in result.issues if "类名应使用大写驼峰" in i.message]
        assert len(func_naming) == 0
        assert len(class_naming) == 0

    # ========== JavaScript 专项检查 ==========

    def test_javascript_var_detected(self):
        code = "var x = 1;\nlet y = 2;"
        result = self.skill.execute(CodeReviewInput(code=code, language="javascript"))
        vars = [i for i in result.issues if "建议使用 let 或 const 替代 var" in i.message]
        assert len(vars) >= 1

    def test_javascript_console_log_detected(self):
        code = "console.log('debug');"
        result = self.skill.execute(CodeReviewInput(code=code, language="javascript"))
        logs = [i for i in result.issues if "console.log" in i.message]
        assert len(logs) >= 1

    # ========== Java 检查 ==========

    def test_java_system_out_detected(self):
        code = "System.out.println(\"debug\");"
        result = self.skill.execute(CodeReviewInput(code=code, language="java"))
        sysouts = [i for i in result.issues if "System.out" in i.message]
        assert len(sysouts) >= 1

    # ========== 未知语言 ==========

    def test_unsupported_language_info(self):
        code = "fn main() {}"
        result = self.skill.execute(CodeReviewInput(code=code, language="rust"))
        unsupported = [i for i in result.issues if "暂不支持专项检查" in i.message]
        assert len(unsupported) >= 1

    # ========== 评分范围测试 ==========

    def test_score_range(self):
        # 完美代码
        result = self.skill.execute(CodeReviewInput(code="x = 1", language="python"))
        assert 0 <= result.score <= 100

        # 非常差的代码
        bad_code = "\n".join([
            "# TODO: fix this",
            "class badclass:",
            "    def BAD_FUNC():",
            "        pass"
        ])
        result2 = self.skill.execute(CodeReviewInput(code=bad_code, language="python"))
        assert 0 <= result2.score <= 100

    # ========== 输出完整性测试 ==========

    def test_output_has_all_fields(self):
        result = self.skill.execute(CodeReviewInput(code="x = 1", language="python"))
        assert hasattr(result, "issues")
        assert hasattr(result, "summary")
        assert hasattr(result, "score")
        assert isinstance(result.issues, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.score, int)

    def test_issue_fields(self):
        code = "# TODO\n" + "x" * 150
        result = self.skill.execute(CodeReviewInput(code=code, language="python"))
        if result.issues:
            issue = result.issues[0]
            assert hasattr(issue, "line")
            assert hasattr(issue, "severity")
            assert hasattr(issue, "message")
            assert hasattr(issue, "suggestion")
            assert issue.severity in ("error", "warning", "info")

    # ========== 无问题代码 ==========

    def test_clean_code_high_score(self):
        clean_code = """def hello(name):
    return f"Hello, {name}!"


class Greeter:
    def greet(self, name):
        return hello(name)
"""
        result = self.skill.execute(CodeReviewInput(code=clean_code, language="python"))
        # 应无错误级别问题
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) == 0
        assert result.score >= 80

    # ========== 多语言用例测试 ==========

    def test_multiple_languages(self):
        languages = ["python", "javascript", "java", "go", "ruby"]
        for lang in languages:
            result = self.skill.execute(CodeReviewInput(code="test", language=lang))
            assert isinstance(result, CodeReviewOutput)

if __name__ == "__main__":
    pytest.main()