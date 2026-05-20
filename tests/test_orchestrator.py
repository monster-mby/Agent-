"""
SkillOrchestrator 单元测试
"""
import os
# 配置 HuggingFace 国内镜像（防止间接依赖触发下载）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pytest
import sys
from pathlib import Path


from dotenv import load_dotenv

# 定位项目根目录的 .env 文件
project_root = Path(__file__).resolve().parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent.orchestrator import (
    SkillOrchestrator,
    Pipeline,
    PipelineStep,
    PipelineType,
    StepResult,
    OrchestratorResult,
    get_orchestrator,
    PREDEFINED_PIPELINES,
)
from src.skills.base.skill_manager import SkillManager


# ... existing code ...

def _create_test_orchestrator() -> SkillOrchestrator:
    """创建测试用的编排器实例（跳过自动发现和初始化）。"""
    from src.agent.llm_client import SimulatedLLM
    import sys

    print("\n" + "=" * 60)
    print(" 开始创建测试编排器...")
    print("=" * 60)

    orch = SkillOrchestrator.__new__(SkillOrchestrator)
    print("✅ Step 1: 创建 SkillOrchestrator 实例（跳过 __init__）")

    orch.skill_manager = SkillManager()
    orch._pipelines = {}
    orch._tools_cache = []
    orch.llm_client = SimulatedLLM()
    print("✅ Step 2: 初始化基本属性")

    # 手动注册预定义流水线
    print("📝 Step 3: 注册预定义流水线...")
    for pipeline in PREDEFINED_PIPELINES:
        orch._pipelines[pipeline.name] = pipeline
    print(f"   ✅ 已注册 {len(orch._pipelines)} 条流水线")

    # 逐个注册技能，带详细日志
    core_skills = [
        # 内容创作类
        ("src.skills.preset.content_creation.text_summarizer.skill", "TextSummarizerSkill"),
        ("src.skills.preset.content_creation.outline_generator.skill", "OutlineGeneratorSkill"),
        # 办公效率类
        ("src.skills.preset.office_efficiency.email_drafter.skill", "EmailDrafterSkill"),
        ("src.skills.preset.office_efficiency.translator.skill", "TranslatorSkill"),
        ("src.skills.preset.office_efficiency.meeting_summarizer.skill", "MeetingSummarizerSkill"),
        # 技术开发类
        ("src.skills.preset.technical_development.code_explainer.skill", "CodeExplainerSkill"),
        # 数据分析类
        ("src.skills.preset.data_analysis.data_cleaner.skill", "DataCleanerSkill"),
        ("src.skills.preset.data_analysis.chart_advisor.skill", "ChartAdvisorSkill"),
        # 自定义技能
        ("src.skills.custom.code_skills.code_review.skill", "CodeReviewSkill"),
        ("src.skills.preset.technical_development.unit_test_generator.skill", "UnitTestGeneratorSkill"),
    ]

    print(f"\n📝 Step 4: 开始注册 {len(core_skills)} 个核心技能...")
    registered = 0
    for idx, (mod_path, cls_name) in enumerate(core_skills, 1):
        print(f"   [{idx}/{len(core_skills)}] 正在导入 {cls_name}...", end="", flush=True)
        try:
            module = __import__(mod_path, fromlist=[cls_name])
            skill_cls = getattr(module, cls_name)
            orch.skill_manager.register(skill_cls)
            registered += 1
            print(f" ✅")
        except Exception as e:
            print(f" ❌ 失败: {e}")

    print(f"\n✅ 成功注册 {registered}/{len(core_skills)} 个技能")

    # 跳过 GraphRAG 技能（可能有外部依赖导致卡顿）
    print("\n️  跳过 GraphRAG 技能（避免外部依赖卡顿）")

    print("\n📝 Step 5: 同步工具列表到 LLM...")
    orch._sync_tools_to_llm()
    print("✅ 工具列表同步完成")

    print("\n" + "=" * 60)
    print("✅ 测试编排器创建完成！")
    print("=" * 60 + "\n")

    return orch

# ... existing code ...




class TestSkillOrchestratorInit:
    """初始化测试"""

    def test_create_orchestrator(self):
        orch = _create_test_orchestrator()
        assert orch is not None
        status = orch.get_status()
        assert status["registered_skills"] >= 5
        assert status["registered_pipelines"] >= 5

    def test_auto_registers_skills(self):
        orch = _create_test_orchestrator()
        skills = orch.list_skills()
        skill_names = [s["name"] for s in skills]
        assert "text_summarizer" in skill_names
        assert "code_explainer" in skill_names
        assert "email_drafter" in skill_names
        assert "translator" in skill_names

    def test_auto_registers_pipelines(self):
        orch = _create_test_orchestrator()
        pipelines = orch.list_pipelines()
        pipeline_names = [p["name"] for p in pipelines]
        assert "summarize_then_email" in pipeline_names
        assert "explain_then_test" in pipeline_names
        assert "clean_then_chart" in pipeline_names

    @pytest.mark.skip(reason="全局单例会触发自动技能发现和网络请求")
    def test_global_singleton(self):
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        assert orch1 is orch2


class TestSingleSkillExecution:
    """单技能执行测试"""

    def test_text_summarizer(self):
        orch = _create_test_orchestrator()
        result = orch.process(
            "帮我总结一下：Python是一门非常流行的编程语言，"
            "广泛用于数据科学、Web开发和自动化运维。"
        )
        assert result.success
        assert result.pipeline_type == "single"

    def test_email_drafter(self):
        orch = _create_test_orchestrator()
        result = orch.process("帮我起草一封请假邮件，因为身体不适")
        assert result.success

    def test_translator(self):
        orch = _create_test_orchestrator()
        result = orch.process("翻译成英文：你好世界")
        assert result.success

    def test_code_review(self):
        orch = _create_test_orchestrator()
        result = orch.process(
            "review this code: def foo(x): return x+1"
        )
        # 可能匹配 code_explainer 或 code_review
        assert result.success


class TestPipelineExecution:
    """流水线执行测试"""

    def test_explain_then_test(self):
        orch = _create_test_orchestrator()
        result = orch.process(
            "分析这段代码然后生成测试：def add(a, b): return a + b"
        )
        assert result.success
        # 应该是流水线
        assert len(result.step_results) >= 1

    def test_summarize_then_email(self):
        orch = _create_test_orchestrator()
        result = orch.process(
            "总结这段文本然后发邮件：人工智能正在改变世界..."
        )
        assert result.success

    def test_run_pipeline_explicitly(self):
        orch = _create_test_orchestrator()
        result = orch.run_pipeline(
            "explain_then_test",
            initial_input={"code": "def greet(name): return f'Hello {name}'"},
        )
        assert result.success
        assert len(result.step_results) == 2

    def test_run_nonexistent_pipeline(self):
        orch = _create_test_orchestrator()
        result = orch.run_pipeline("nonexistent", initial_input={})
        assert not result.success
        assert "不存在" in result.summary


class TestPipelineManagement:
    """流水线管理测试"""

    def test_register_custom_pipeline(self):
        orch = _create_test_orchestrator()
        custom = Pipeline(
            name="custom_test_pipeline",
            description="测试自定义流水线",
            triggers=["自定义测试"],
            steps=[
                PipelineStep(
                    skill_name="text_summarizer",
                    description="摘要步骤",
                ),
            ],
        )
        orch.register_pipeline(custom)
        pipelines = orch.list_pipelines()
        names = [p["name"] for p in pipelines]
        assert "custom_test_pipeline" in names

    def test_list_pipelines_has_required_fields(self):
        orch = _create_test_orchestrator()
        pipelines = orch.list_pipelines()
        for p in pipelines:
            assert "name" in p
            assert "description" in p
            assert "type" in p
            assert "steps" in p
            assert "triggers" in p

    def test_list_skills_has_required_fields(self):
        orch = _create_test_orchestrator()
        skills = orch.list_skills()
        for s in skills:
            assert "name" in s
            assert "description" in s


class TestOrchestratorResult:
    """结果对象测试"""

    def test_result_structure(self):
        result = OrchestratorResult(
            success=True,
            pipeline_type="single",
            pipeline_name="test",
            step_results=[],
            final_output={"data": "test"},
        )
        assert result.success
        assert result.pipeline_type == "single"

    def test_failed_result(self):
        result = OrchestratorResult(
            success=False,
            pipeline_type="single",
            pipeline_name="error",
            step_results=[],
            final_output=None,
            summary="失败原因",
        )
        assert not result.success
        assert "失败" in result.summary


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_input(self):
        orch = _create_test_orchestrator()
        result = orch.process("")
        # 应该返回一个结果（可能是失败或兜底）
        assert result is not None

    def test_very_short_input(self):
        orch = _create_test_orchestrator()
        result = orch.process("你好")
        assert result is not None

    def test_nonsense_input(self):
        orch = _create_test_orchestrator()
        result = orch.process("xyzzy123 乱七八糟")
        # 应该能处理（可能匹配失败，返回提示）
        assert result is not None

    def test_multiple_orchestrator_instances(self):
        orch1 = _create_test_orchestrator()
        orch2 = _create_test_orchestrator()
        # 两个实例应该独立工作
        assert orch1.get_status()["registered_skills"] == \
               orch2.get_status()["registered_skills"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
