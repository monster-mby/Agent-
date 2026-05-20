"""测试 BaseSkill 基类和 HelloSkill"""
from pydantic import BaseModel
import pytest
from skills.custom.learning_skills.hello.hello_skill import HelloSkill
from src.skills.base.base_skill import BaseSkill


@pytest.fixture
def hello_skill():
    """创建 HelloSkill 实例"""
    return HelloSkill()


class TestHelloSkill:
    """测试 HelloSkill 的各项功能"""

    def test_skill_metadata(self, hello_skill):
        """验证技能元数据"""
        assert hello_skill.name == "hello"
        assert hello_skill.description
        assert hello_skill.version == "0.1.0"
        assert hello_skill.author == "demo"

    def test_input_schema(self, hello_skill):
        """验证输入 Schema 是 Pydantic Model"""
        assert hello_skill.input_schema is not None
        assert issubclass(hello_skill.input_schema, BaseModel)  # 是 Pydantic Model

    def test_output_schema(self, hello_skill):
        """验证输出 Schema 是 Pydantic Model"""
        assert hello_skill.output_schema is not None
        assert issubclass(hello_skill.output_schema, BaseModel)

    def test_execute_default(self, hello_skill):
        """测试默认执行"""
        result = hello_skill.execute()
        assert "你好" in result["message"]
        assert result["greeted_name"] == "朋友"

    def test_execute_with_name(self, hello_skill):
        """测试传入姓名"""
        result = hello_skill.execute(name="张三")
        assert "张三" in result["message"]
        assert result["greeted_name"] == "张三"

    def test_execute_english(self, hello_skill):
        """测试英文问候"""
        result = hello_skill.execute(name="Alice", language="en")
        assert "Hello" in result["message"]
        assert result["greeted_name"] == "Alice"

    def test_validate_input_valid(self, hello_skill):
        """测试输入验证通过（不应抛出异常）"""
        # 正常调用不会 raise
        hello_skill.validate_input(name="小明", language="zh")

    def test_validate_input_invalid(self, hello_skill):
        """测试输入验证不通过（应抛出 ValueError）"""
        with pytest.raises(ValueError) as exc_info:
            hello_skill.validate_input(language="invalid_lang")
        assert "输入验证失败" in str(exc_info.value)

    def test_validate_output_valid(self, hello_skill):
        """测试输出验证通过"""
        output = {"message": "你好", "greeted_name": "小明", "language": "zh"}
        result = hello_skill.validate_output(output)
        assert result["message"] == "你好"

    def test_validate_output_missing_field(self, hello_skill):
        """测试输出缺少必填字段"""
        output = {"message": "你好"}  # 缺少 greeted_name 和 language
        with pytest.raises(ValueError) as exc_info:
            hello_skill.validate_output(output)
        assert "输出验证失败" in str(exc_info.value)

    def test_is_abstract(self):
        """验证 BaseSkill 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseSkill()

if __name__ == "__main__":
    pytest.main()