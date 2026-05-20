"""
测试 SkillManager
"""
import unittest

import pytest
from pydantic import BaseModel
from src.skills.base import BaseSkill, SkillManager


# ==================== 测试用技能 ====================

class GreetInput(BaseModel):
    name: str
    greeting: str = "你好"


class GreetOutput(BaseModel):
    message: str


class GreetSkill(BaseSkill):
    name = "greet"
    description = "向某人打招呼"
    version = "1.0.0"
    author = "tester"
    triggers = ["你好", "hello", "打招呼", "问候"]
    input_schema = GreetInput
    output_schema = GreetOutput

    def execute(self, input_data: GreetInput) -> GreetOutput:
        """执行问候逻辑"""
        return GreetOutput(message=f"{input_data.greeting}，{input_data.name}！")


class CalcSkill(BaseSkill):
    name = "calculator"
    description = "简单的加法计算器"
    version = "0.1.0"
    author = "tester"
    triggers = ["计算", "加法", "加"]

    def execute(self, **kwargs) -> dict:
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        return {"result": a + b}


# ==================== 测试 SkillManager ====================

class TestSkillManager:
    """测试技能管理器"""

    def setup_method(self):
        self.manager = SkillManager()

    def test_register(self):
        """测试注册技能"""
        self.manager.register(GreetSkill)
        assert "greet" in self.manager.get_all()
        assert self.manager.count == 1

    def test_register_duplicate(self):
        """测试重复注册报错"""
        self.manager.register(GreetSkill)
        with pytest.raises(ValueError, match="已注册"):
            self.manager.register(GreetSkill)

    def test_register_invalid(self):
        """测试注册非 BaseSkill 报错"""
        class NotASkill:
            name = "fake"
        with pytest.raises(TypeError, match="必须是 BaseSkill"):
            self.manager.register(NotASkill)

    def test_register_many(self):
        """测试批量注册"""
        registered = self.manager.register_many(GreetSkill, CalcSkill)
        assert "greet" in registered
        assert "calculator" in registered
        assert self.manager.count == 2

    def test_unregister(self):
        """测试注销技能"""
        self.manager.register(GreetSkill)
        assert self.manager.unregister("greet") is True
        assert self.manager.count == 0

    def test_unregister_not_found(self):
        """测试注销不存在的技能"""
        assert self.manager.unregister("not_exists") is False

    def test_get(self):
        """测试获取技能类"""
        self.manager.register(GreetSkill)
        assert self.manager.get("greet") == GreetSkill
        assert self.manager.get("not_exists") is None

    def test_find_by_trigger(self):
        """测试按触发词查找"""
        self.manager.register_many(GreetSkill, CalcSkill)
        matched = self.manager.find_by_trigger("你好，请问有人在吗")
        assert GreetSkill in matched
        assert CalcSkill not in matched

    def test_find_by_trigger_multiple(self):
        """测试按触发词查找（匹配多个）"""
        self.manager.register_many(GreetSkill, CalcSkill)
        matched = self.manager.find_by_trigger("请帮我计算一下")
        assert CalcSkill in matched
        assert GreetSkill not in matched

    def test_find_by_trigger_empty(self):
        """测试空文本触发"""
        self.manager.register(GreetSkill)
        assert self.manager.find_by_trigger("") == []

    def test_find_by_keyword_name(self):
        """测试按关键词匹配名称"""
        self.manager.register_many(GreetSkill, CalcSkill)
        matched = self.manager.find_by_keyword("greet")
        assert GreetSkill in matched
        assert CalcSkill not in matched

    def test_find_by_keyword_description(self):
        """测试按关键词匹配描述"""
        self.manager.register_many(GreetSkill, CalcSkill)
        matched = self.manager.find_by_keyword("打招呼")
        assert GreetSkill in matched

    def test_call_success(self):
        """测试成功调用技能"""
        self.manager.register(GreetSkill)
        result = self.manager.call("greet", name="张三", greeting="嗨")
        assert result["status"] == "success"
        assert result["output"]["message"] == "嗨，张三！"

    def test_call_not_found(self):
        """测试调用不存在的技能"""
        result = self.manager.call("not_exists", x=1)
        assert result["status"] == "error"
        assert "未注册" in result["error"]

    def test_call_validation_error(self):
        """测试调用时输入验证失败"""
        self.manager.register(GreetSkill)
        result = self.manager.call("greet")  # 缺少 name 参数
        assert result["status"] == "error"

    def test_get_tools_for_llm(self):
        """测试生成 LLM 工具描述"""
        self.manager.register(GreetSkill)
        tools = self.manager.get_tools_for_llm()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "greet"
        assert "parameters" in tools[0]["function"]

    def test_get_metadata(self):
        """测试获取技能元数据"""
        self.manager.register(GreetSkill)
        meta = self.manager.get_metadata("greet")
        assert meta["name"] == "greet"
        assert meta["version"] == "1.0.0"
        assert meta["author"] == "tester"
        assert meta["has_input_schema"] is True

    def test_get_metadata_not_found(self):
        """测试获取不存在的技能元数据"""
        assert self.manager.get_metadata("not_exists") is None

    def test_list_all_metadata(self):
        """测试列出所有技能元数据"""
        self.manager.register_many(GreetSkill, CalcSkill)
        all_meta = self.manager.list_all_metadata()
        assert len(all_meta) == 2
        names = [m["name"] for m in all_meta]
        assert "greet" in names
        assert "calculator" in names

    def test_reset_via_operations(self):
        """测试注册注销的综合操作"""
        self.manager.register(GreetSkill)
        assert self.manager.count == 1
        self.manager.unregister("greet")
        assert self.manager.count == 0
        self.manager.register(CalcSkill)
        assert self.manager.count == 1
        assert self.manager.get_all() == ["calculator"]

#总入口
if __name__ == "__main__":
    unittest.main()