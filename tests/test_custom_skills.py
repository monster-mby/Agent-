"""
测试迁移到 custom/ 目录下的技能
"""

import pytest
from src.skills.custom.learning_skills.hello.hello_skill import HelloSkill
from src.skills.custom.code_skills.code_review.skill import CodeReviewSkill, CodeReviewInput


class TestHelloSkill:
    """测试 HelloSkill（迁移后）"""

    def setup_method(self):
        self.skill = HelloSkill()

    def test_meta(self):
        assert self.skill.name == "hello"
        assert self.skill.description

    def test_execute_default(self):
        result = self.skill.execute()
        assert "message" in result
        assert "朋友" in result["message"]

    def test_execute_with_name(self):
        result = self.skill.execute(name="张三")
        assert "张三" in result["message"]

    def test_execute_empty_name(self):
        result = self.skill.execute(name="")
        assert result["message"] == "你好，！"


class TestCodeReviewSkill:
    """测试 CodeReviewSkill（迁移后）"""

    def setup_method(self):
        self.skill = CodeReviewSkill()

    def test_meta(self):
        assert self.skill.name == "code_review"
        assert self.skill.description

    def test_review_python(self):
        code = """
def add(a, b):
    return a + b
"""
        input_data = CodeReviewInput(code=code, language="python")
        result = self.skill.execute(input_data)
        # result 是 CodeReviewOutput 对象，使用属性访问
        assert hasattr(result, 'issues')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'score')

    def test_review_js(self):
        code = """
function add(a, b) {
    return a + b;
}
"""
        input_data = CodeReviewInput(code=code, language="javascript")
        result = self.skill.execute(input_data)
        assert result.score >= 0
        assert result.score <= 100

    def test_review_with_bugs(self):
        code = """
def divide(a, b):
    return a / b
"""
        input_data = CodeReviewInput(code=code, language="python")
        result = self.skill.execute(input_data)
        assert len(result.issues) > 0  # 应该有除零警告
if __name__ == "__main__":
    pytest.main()
