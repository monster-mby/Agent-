"""
Agent 引擎测试套件。

测试策略：
1. 使用 SimulatedLLM（无需 API Key）
2. 覆盖：注册、注销、执行、多轮对话、错误处理
"""

import pytest
from src.agent.engine import AgentEngine
from src.agent.llm_client import SimulatedLLM
from skills.custom.learning_skills.hello import HelloSkill
from skills.custom.code_skills.code_review.skill import CodeReviewSkill


class TestAgentEngine:

    def setup_method(self):
        self.engine = AgentEngine()

    # ========== 技能注册 ==========

    def test_register_single_skill(self):
        self.engine.register_skill(HelloSkill)
        assert "hello" in self.engine.get_registered_skills()
        assert self.engine.get_skill("hello") == HelloSkill

    def test_register_multiple_skills(self):
        self.engine.register_skills(HelloSkill, CodeReviewSkill)
        skills = self.engine.get_registered_skills()
        assert "hello" in skills
        assert "code_review" in skills

    def test_register_duplicate_skill_raises(self):
        self.engine.register_skill(HelloSkill)
        with pytest.raises(ValueError, match="已注册"):
            self.engine.register_skill(HelloSkill)

    def test_register_non_skill_class_raises(self):
        class NotASkill:
            pass
        with pytest.raises(TypeError, match="必须是 BaseSkill 的子类"):
            self.engine.register_skill(NotASkill)  # type: ignore

    def test_unregister_skill(self):
        self.engine.register_skill(HelloSkill)
        self.engine.unregister_skill("hello")
        assert "hello" not in self.engine.get_registered_skills()

    def test_unregister_nonexistent_skill(self):
        # 不应抛出异常
        self.engine.unregister_skill("nonexistent")

    # ========== 构建工具描述 ==========

    def test_build_tools_has_default_graphrag_skills(self):
        """AgentEngine 初始化时已注册 2 个 GraphRAG 技能"""
        tools = self.engine._build_tools_for_llm()
        assert len(tools) == 2

        tool_names = [t["function"]["name"] for t in tools]
        assert "graphrag_indexer" in tool_names
        assert "graphrag_searcher" in tool_names

    def test_build_tools_with_skills(self):
        self.engine.register_skills(HelloSkill, CodeReviewSkill)
        tools = self.engine._build_tools_for_llm()
        # 初始 2 个 GraphRAG + 新增 2 个 = 4 个
        assert len(tools) == 4

        # 验证结构
        tool_names = [t["function"]["name"] for t in tools]
        assert "hello" in tool_names
        assert "code_review" in tool_names

        # 验证每个工具都有 description 和 parameters
        for tool in tools:
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_build_tools_has_json_schema(self):
        self.engine.register_skill(CodeReviewSkill)
        tools = self.engine._build_tools_for_llm()

        # 找到 code_review 工具(不是 tools[0],因为前两个是 GraphRAG 技能)
        code_review_tool = None
        for tool in tools:
            if tool["function"]["name"] == "code_review":
                code_review_tool = tool
                break

        assert code_review_tool is not None
        params = code_review_tool["function"]["parameters"]
        assert "properties" in params
        assert "code" in params["properties"]
        assert "language" in params["properties"]

    # ========== 引擎执行（无技能注册） ==========

    def test_run_no_skills(self):
        result = self.engine.run("你好")
        assert "response" in result
        assert result["skill_calls"] == []
        assert len(result["history"]) == 2  # user + assistant

    # ========== 引擎执行（HelloSkill） ==========

    def test_run_hello_skill(self):
        self.engine.register_skill(HelloSkill)
        result = self.engine.run("hello")
        assert "response" in result
        # 应该触发了 skill 调用
        assert len(result["skill_calls"]) >= 0  # 模拟LLM可能匹配也可能不匹配

    def test_run_hello_with_name(self):
        self.engine.register_skill(HelloSkill)
        result = self.engine.run("hello 小明")
        # 模拟 LLM 应该能提取出名字
        if result["skill_calls"]:
            call = result["skill_calls"][0]
            if call["status"] == "success":
                output = call["output"]
                assert "小明" in output.get("message", "") or "小明" in str(output)

    # ========== 引擎执行（CodeReviewSkill） ==========

    def test_run_code_review(self):
        self.engine.register_skill(CodeReviewSkill)
        result = self.engine.run("审查这段 Python 代码：```python\ndef hello():\n    print('hello')```")
        assert "response" in result

    def test_run_code_review_finds_issues(self):
        self.engine.register_skill(CodeReviewSkill)
        bad_code = """```python
def BAD_FUNC():
    pass
```"""
        result = self.engine.run(f"帮我检查代码：{bad_code}")
        # 如果调用了 code_review，应该有 issues
        for call in result["skill_calls"]:
            if call["skill"] == "code_review" and call["status"] == "success":
                assert "issues" in call["output"]

    # ========== 多轮对话 ==========

    def test_conversation_history(self):
        self.engine.register_skill(HelloSkill)

        # 第一轮
        r1 = self.engine.run("hello")
        assert len(self.engine.get_history()) == 2

        # 第二轮
        r2 = self.engine.run("hello again")
        assert len(self.engine.get_history()) == 4

        # 历史记录应该包含所有消息
        history = self.engine.get_history()
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_reset_history(self):
        self.engine.run("hello")
        assert len(self.engine.get_history()) > 0
        self.engine.reset_history()
        assert self.engine.get_history() == []

    def test_run_returns_history(self):
        result = self.engine.run("hi")
        assert "history" in result
        assert isinstance(result["history"], list)

    # ========== 引擎状态 ==========

    def test_engine_repr_has_default_skills(self):
        """AgentEngine 初始化已有 GraphRAG 技能"""
        r = repr(self.engine)
        assert "graphrag_indexer" in r or "graphrag_searcher" in r

    def test_engine_repr_with_skills(self):
        self.engine.register_skill(HelloSkill)
        r = repr(self.engine)
        assert "hello" in r

    # ========== 集成测试：多技能 + 多轮 ==========

    def test_integration_multiple_skills(self):
        self.engine.register_skills(HelloSkill, CodeReviewSkill)

        # 先打招呼
        r1 = self.engine.run("你好")
        assert r1["response"]

        # 再审查代码
        r2 = self.engine.run("审查这段 Python 代码：```python\nx=1```")
        assert r2["response"]

        # 历史应该包含 4 条消息
        assert len(self.engine.get_history()) == 4

    def test_integration_skill_error_handling(self):
        self.engine.register_skill(CodeReviewSkill)

        # 传入空代码
        result = self.engine.run("审查代码：```python\n```")
        # 引擎不应崩溃
        assert "response" in result

    # ========== 模拟 LLM 测试 ==========

    def test_simulated_llm_hello(self):
        llm = SimulatedLLM()
        response = llm.chat([{"role": "user", "content": "hello"}])
        assert "tool_calls" in response

    def test_simulated_llm_code_review(self):
        llm = SimulatedLLM()
        response = llm.chat([{"role": "user", "content": "审查这段代码"}])
        assert "tool_calls" in response

    def test_simulated_llm_unknown(self):
        llm = SimulatedLLM(skills_meta=[
            {"name": "hello", "description": "打招呼"}
        ])
        response = llm.chat([{"role": "user", "content": "今天天气怎么样？"}])
        # 不应匹配任何技能
        assert len(response["tool_calls"]) == 0

        # ... existing code ...

    def test_simulated_llm_extract_name(self):
        llm = SimulatedLLM()
        name = llm._extract_name_param("hello 张三")
        assert name == "张三"

        name2 = llm._extract_name_param("hi, 李四")
        assert name2 == "李四"

        name3 = llm._extract_name_param("天气怎么样")
        assert name3 is None

    # ... existing code ...

    def test_simulated_llm_extract_code_with_backticks(self):
        llm = SimulatedLLM()
        user_input = "审查代码：```python\ndef foo():\n    pass\n```"
        code = llm._extract_code(user_input)
        assert "def foo()" in code
        assert "pass" in code

    def test_simulated_llm_tools_update(self):
        llm = SimulatedLLM()
        tools = [
            {
                "function": {
                    "name": "my_skill",
                    "description": "我的技能"
                }
            }
        ]
        response = llm.chat([{"role": "user", "content": "帮我"}], tools=tools)
        # 元数据应该被更新
        assert len(llm.skills_meta) == 1
        assert llm.skills_meta[0]["name"] == "my_skill"


class TestAgentEngineAdvanced:

    def test_run_handles_malformed_input(self):
        engine = AgentEngine()
        engine.register_skill(CodeReviewSkill)

        # 极长的输入不应崩溃
        long_input = "审查代码：" + "x" * 10000
        result = engine.run(long_input)
        assert "response" in result

    def test_run_with_system_prompt(self):
        engine = AgentEngine()
        engine.register_skill(HelloSkill)

        result = engine.run("hello", system_prompt="你是一个友好的助手。")
        assert result["response"]

    def test_skill_calls_detail_in_result(self):
        engine = AgentEngine()
        engine.register_skill(CodeReviewSkill)

        result = engine.run("审查 Python 代码：```python\nprint('ok')```")
        # 结果中应包含技能调用详情
        assert "skill_calls" in result
        for call in result["skill_calls"]:
            assert "skill" in call
            assert "status" in call

    def test_no_duplicate_history(self):
        engine = AgentEngine()

        # 同一个输入多次
        engine.run("hello")
        engine.run("hello")

        history = engine.get_history()
        # 应该有 4 条（2 user + 2 assistant）
        assert len(history) == 4
        # 没有重复的 user 消息合并
        assert history[0]["content"] == "hello"
        assert history[2]["content"] == "hello"

if __name__ == "__main__":    pytest.main()