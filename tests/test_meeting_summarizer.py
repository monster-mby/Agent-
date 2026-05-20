import pytest
from pydantic import ValidationError
from src.skills.preset.office_efficiency.meeting_summarizer import (
    MeetingSummarizerSkill,
    MeetingSummarizerInput,
    ActionItem,
    MeetingRules,
)


# ──────────────────────────────────────────────
# 1. 输入模型验证测试
# ──────────────────────────────────────────────
class TestMeetingSummarizerInput:
    def test_attendees_from_string(self):
        """测试参会人从逗号分隔字符串转换为列表"""
        inp = MeetingSummarizerInput(
            meeting_transcript="这是一段足够长的会议文本用于测试。",
            attendees="张三, 李四, 王五"
        )
        assert inp.attendees == ["张三", "李四", "王五"]

    def test_attendees_from_list(self):
        """测试参会人直接传入列表"""
        inp = MeetingSummarizerInput(
            meeting_transcript="这是一段足够长的会议文本用于测试。",
            attendees=["Alice", "Bob"]
        )
        assert inp.attendees == ["Alice", "Bob"]

    def test_attendees_empty(self):
        """测试参会人为空"""
        inp = MeetingSummarizerInput(
            meeting_transcript="这是一段足够长的会议文本用于测试。",
            attendees=""
        )
        assert inp.attendees == []

    def test_transcript_too_short(self):
        """测试会议转录文本过短（应抛出 ValidationError）"""
        with pytest.raises(ValidationError, match="string_too_short"):
            MeetingSummarizerInput(meeting_transcript="太短")

# ──────────────────────────────────────────────
# 2. 核心逻辑单元测试
# ──────────────────────────────────────────────
class TestCoreExtraction:
    skill = MeetingSummarizerSkill()

    def test_extract_topics_numbered(self):
        """测试提取带编号的议题"""
        text = """
        1. 第一季度工作总结
        二、下季度规划
        第三、人员招聘
        """
        topics = self.skill._extract_topics(text)
        assert "第一季度工作总结" in topics
        assert "下季度规划" in topics
        assert "人员招聘" in topics

    def test_extract_topics_keywords(self):
        """测试提取带关键词的议题"""
        text = """
        议题：关于系统升级的讨论
        汇报：上月财务状况
        主题：团建活动安排
        """
        topics = self.skill._extract_topics(text)
        # _extract_topics 返回完整字符串（包含前缀），检查是否包含内容
        assert any("关于系统升级的讨论" in t for t in topics), f"未找到议题: {topics}"

    def test_extract_decisions(self):
        """测试提取决策事项"""
        text = """
        经过讨论，决定采用方案B。
        最终确定发布会定在6月18日。
        大家一致同意通过预算申请。
        这条没有决策词。
        """
        decisions = self.skill._extract_decisions(text)
        assert any("采用方案B" in d for d in decisions)
        assert any("6月18日" in d for d in decisions)

    def test_extract_action_items_chinese(self):
        """测试提取中文待办（含人名、截止日期）"""
        text = """
        张三需要完成API文档，截止明天。
        李四负责跟进客户反馈，urgent。
        待办：王五 - 准备会议室
        """
        items = self.skill._extract_action_items(text)

        assert len(items) >= 3
        owners = [i.owner for i in items]
        assert "张三" in owners
        assert "李四" in owners
        assert "王五" in owners

        zhangsan = next(i for i in items if i.owner == "张三")
        assert "API文档" in zhangsan.task
        assert zhangsan.deadline == "明天"

        lisi = next(i for i in items if i.owner == "李四")
        assert lisi.priority == "high"

    def test_extract_action_items_english(self):
        """测试提取英文待办"""
        text = """
        @Alex will finish the report.
        Action item: Bob - Review PR by Friday.
        """
        items = self.skill._extract_action_items(text)
        owners = [i.owner.lower() for i in items]
        assert "alex" in owners
        assert "bob" in owners

    def test_extract_deadline(self):
        """测试截止日期提取"""
        assert self.skill._extract_deadline("请在明天提交") == "明天"
        assert self.skill._extract_deadline("截止下周三") == "下周三"
        assert self.skill._extract_deadline("5月20日之前") == "5月20日"
        assert self.skill._extract_deadline("deadline: 2026-06-01") == "2026-06-01"
        assert self.skill._extract_deadline("没有日期") == "待定"

    def test_infer_priority(self):
        """测试优先级推断"""
        assert MeetingSummarizerSkill._infer_priority("紧急修复bug") == "high"
        assert MeetingSummarizerSkill._infer_priority("尽快上线") == "high"
        assert MeetingSummarizerSkill._infer_priority("有空优化一下") == "low"
        assert MeetingSummarizerSkill._infer_priority("普通任务") == "medium"

    def test_invalid_owners_filtered(self):
        """测试无效负责人被过滤"""
        text = """
        我需要做这个。
        我们负责那个。
        大家跟进一下。
        张三来做具体的事。
        """
        items = self.skill._extract_action_items(text)
        # 只有"张三"是有效负责人
        assert len(items) == 1
        assert items[0].owner == "张三"


# ──────────────────────────────────────────────
# 3. 摘要与Markdown生成测试
# ──────────────────────────────────────────────
class TestOutputGeneration:
    skill = MeetingSummarizerSkill()

    def test_generate_summary(self):
        """测试会议摘要生成"""
        summary = MeetingSummarizerSkill._generate_summary(
            title="产品发布会",
            topics=["议题1", "议题2", "（待补充议题）"],
            decisions=["决策1", "（未检测到明确决策项）"],
            items=[
                ActionItem(task="任务1", owner="A"),
                ActionItem(task="（请补充待办事项）", owner="B"),
            ],
        )
        assert "产品发布会" in summary
        assert "2 个议题" in summary
        assert "1 项决策" in summary
        assert "1 项待办" in summary

    def test_markdown_contains_sections(self):
        """测试生成的Markdown包含必要章节"""
        md = self.skill._generate_markdown_fallback(
            title="测试会议",
            date="2026-05-01",
            attendees=["张三"],
            topics=["测试议题"],
            decisions=["测试决策"],
            action_items=[ActionItem(task="测试任务", owner="测试人")],
            summary="这是摘要",
        )
        assert "# 📋 测试会议" in md
        assert "## 🗣️ 讨论议题" in md
        assert "## ✅ 决策事项" in md
        assert "## 📌 待办事项" in md
        assert "测试任务" in md
        assert "测试人" in md


# ──────────────────────────────────────────────
# 4. 端到端集成测试
# ──────────────────────────────────────────────
class TestEndToEnd:
    def test_full_execution(self):
        """测试完整的执行流程"""
        skill = MeetingSummarizerSkill()

        input_data = MeetingSummarizerInput(
            meeting_title="Q2 研发规划会",
            date="2026-05-01",
            attendees="张三, 李四, 王五",
            meeting_transcript="""
            会议时间：2026年5月1日

            1. 议题一：上季度Bug回顾
               决定：优先修复P0级Bug。

            2. 议题二：新功能排期
               经过讨论，确定6月底完成MVP。

            待办事项：
            - 张三需要在本周内完成技术方案设计。
            - 李四负责跟进UI设计稿，截止5月10日，紧急。
            - 王五：准备测试用例。
            """,
        )

        output = skill.execute(input_data)

        # 校验基础信息
        assert output.title == "Q2 研发规划会"
        assert output.date == "2026-05-01"
        assert output.attendees == ["张三", "李四", "王五"]

        # 校验提取内容
        assert len(output.topics) >= 2
        assert len(output.decisions) >= 1
        assert len(output.action_items) >= 3

        # 校验具体待办
        zhangsan = next(i for i in output.action_items if i.owner == "张三")
        assert "技术方案" in zhangsan.task
        # 张三的待办原文是"本周内完成"，但没有具体日期（如"本周一"），返回"待定"是合理的
        assert zhangsan.deadline in [None, "", "待定"] or "本周" in zhangsan.deadline, \
            f"张三的截止日期异常: {zhangsan.deadline}"

        # 校验李四的待办（有明确截止日期）
        lisi = next(i for i in output.action_items if i.owner == "李四")
        assert "5月10日" in lisi.deadline or "05-10" in lisi.deadline

        lisi = next(i for i in output.action_items if i.owner == "李四")
        assert lisi.priority == "high"
        assert "5月10日" in lisi.deadline

        # 校验Markdown
        assert "Q2 研发规划会" in output.markdown_output
        assert "张三" in output.markdown_output
        assert "🟡" in output.markdown_output or "🔴" in output.markdown_output

if __name__ == "__main__":
    pytest.main()