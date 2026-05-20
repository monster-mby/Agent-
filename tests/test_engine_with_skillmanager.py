"""
测试 AgentEngine 集成 SkillManager 后的功能
"""

import pytest
from src.agent.engine import AgentEngine
from src.skills.custom.learning_skills.hello.hello_skill import HelloSkill
from src.skills.custom.code_skills.code_review.skill import CodeReviewSkill


class TestAgentEngineWithSkillManager:
    """测试集成 SkillManager 后的 AgentEngine"""

    def setup_method(self):
        self.engine = AgentEngine()

    def test_register_skill(self):
        self.engine.register_skill(HelloSkill)
        assert "hello" in self.engine.get_registered_skills()

    def test_register_skills_batch(self):
        self.engine.register_skills(HelloSkill, CodeReviewSkill)
        skills = self.engine.get_registered_skills()
        assert "hello" in skills
        assert "code_review" in skills

    def test_unregister_skill(self):
        self.engine.register_skill(HelloSkill)
        assert self.engine.unregister_skill("hello") is True
        assert "hello" not in self.engine.get_registered_skills()

    def test_run_hello(self):
        self.engine.register_skill(HelloSkill)
        result = self.engine.run("你好，请打招呼")
        assert "response" in result
        assert "skill_calls" in result

    def test_run_code_review(self):
        self.engine.register_skill(CodeReviewSkill)
        result = self.engine.run("审查这段代码：def add(a,b): return a+b")
        assert "response" in result

    def test_multiple_skills(self):
        self.engine.register_skills(HelloSkill, CodeReviewSkill)
        skills = self.engine.get_registered_skills()
        # 由于自动发现机制，可能包含其他技能，只断言目标技能存在
        assert "hello" in skills
        assert "code_review" in skills

    def test_reset_history(self):
        self.engine.run("你好")
        assert len(self.engine.get_history()) > 0
        self.engine.reset_history()
        assert len(self.engine.get_history()) == 0

    def test_run_with_system_prompt(self):
        self.engine.register_skill(HelloSkill)
        result = self.engine.run("你好", system_prompt="你是一个助手")
        assert result["response"]

if __name__ == "__main__":
    pytest.main()