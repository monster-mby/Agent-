import pytest
from pydantic import ValidationError
from src.skills.preset.office_efficiency.email_drafter import (
    EmailDrafterSkill,
    EmailDrafterInput,
)


@pytest.fixture
def skill():
    """创建 EmailDrafterSkill 实例的 fixture"""
    return EmailDrafterSkill()


# ==========================================
# 1. 测试输入验证 (EmailDrafterInput)
# ==========================================
def test_input_key_points_list():
    """测试 key_points 接受列表输入"""
    inp = EmailDrafterInput(
        key_points=["要点1", "要点2", "要点3"]
    )
    assert inp.key_points == "要点1\n要点2\n要点3"


def test_input_key_points_string():
    """测试 key_points 接受字符串输入"""
    inp = EmailDrafterInput(
        key_points="要点1,要点2\n要点3；要点4"
    )
    assert inp.key_points == "要点1,要点2\n要点3；要点4"


def test_input_key_points_empty_error():
    """测试 key_points 为空时抛出验证错误"""
    with pytest.raises(ValidationError, match="key_points 不能为空"):
        EmailDrafterInput(key_points="")


def test_input_key_points_only_separators_error():
    """测试 key_points 仅包含分隔符时抛出验证错误"""
    with pytest.raises(ValidationError, match="key_points 不能为空"):
        EmailDrafterInput(key_points="\n,，;；   ")


def test_input_default_values():
    """测试输入字段的默认值"""
    inp = EmailDrafterInput(key_points="测试要点")
    assert inp.scenario == "general"
    assert inp.tone == "formal"
    assert inp.language == "zh"
    assert inp.recipient_name == ""
    assert inp.sender_name == ""


# ==========================================
# 2. 测试要点解析 (_parse_points)
# ==========================================
def test_parse_points_newline():
    """测试换行符分隔"""
    points = EmailDrafterSkill._parse_points("要点1\n要点2\n要点3")
    assert points == ["要点1", "要点2", "要点3"]


def test_parse_points_comma():
    """测试逗号分隔"""
    points = EmailDrafterSkill._parse_points("要点1,要点2,要点3")
    assert points == ["要点1", "要点2", "要点3"]


def test_parse_points_chinese_comma():
    """测试中文逗号分隔"""
    points = EmailDrafterSkill._parse_points("要点1，要点2，要点3")
    assert points == ["要点1", "要点2", "要点3"]


def test_parse_points_mixed_separators():
    """测试混合分隔符"""
    points = EmailDrafterSkill._parse_points("要点1,要点2\n要点3；要点4")
    assert points == ["要点1", "要点2", "要点3", "要点4"]


def test_parse_points_whitespace_stripping():
    """测试去除空白字符"""
    points = EmailDrafterSkill._parse_points("  要点1  ,  要点2  \n  要点3  ")
    assert points == ["要点1", "要点2", "要点3"]


# ==========================================
# 3. 测试主题生成 (_build_subject)
# ==========================================
def test_build_subject_leave_request_zh(skill):
    """测试请假申请中文主题"""
    subject = skill._build_subject("leave_request", "zh", sender="张三", date="2026-05-01")
    assert subject == "请假申请"


def test_build_subject_meeting_invitation_en(skill):
    """测试会议邀请英文主题"""
    subject = skill._build_subject("meeting_invitation", "en", sender="John", date="2026-05-01")
    assert subject == "Meeting Invitation"


def test_build_subject_general_zh(skill):
    """测试通用场景中文主题"""
    subject = skill._build_subject("general", "zh", sender="张三", date="2026-05-01")
    assert subject == "[主题]"


# ==========================================
# 4. 测试正文生成 (_generate_body)
# ==========================================
def test_generate_body_formal_zh_leave(skill):
    """测试正式中文请假邮件正文"""
    points = ["请假时间：5.1-5.3", "工作已交接"]
    body = skill._generate_body(points, "leave_request", "formal", "zh")

    assert "您好！" in body
    assert "我因[个人原因/身体不适]" in body
    assert "1. 请假时间：5.1-5.3" in body
    assert "2. 工作已交接" in body
    assert "感谢您的理解与批准" in body
    assert "如有任何问题，请随时与我联系" in body


def test_generate_body_semi_formal_en_project(skill):
    """测试半正式英文项目汇报正文"""
    points = ["Progress: 80%", "Next step: Testing"]
    body = skill._generate_body(points, "project_update", "semi_formal", "en")

    assert "Hope you're doing well." in body
    assert "Here is the latest project update:" in body
    assert "1. Progress: 80%" in body
    assert "2. Next step: Testing" in body
    assert "Looking forward to your feedback." in body
    assert "Let me know if you have any questions." in body


def test_generate_body_casual_zh_general(skill):
    """测试随意中文通用邮件正文"""
    points = ["随便聊聊"]
    body = skill._generate_body(points, "general", "casual", "zh")

    assert "你好！" in body
    assert "1. 随便聊聊" in body
    assert "有问题随时找我～" in body


# ==========================================
# 5. 测试语言组装 (_assemble_by_lang)
# ==========================================
def test_assemble_by_lang_zh():
    """测试纯中文模式组装"""
    greeting, body, closing = EmailDrafterSkill._assemble_by_lang(
        "zh",
        "尊敬的张三：", "Dear Zhang San,",
        "中文正文", "English body",
        "此致\n敬礼\n\n李四", "Best regards,\nLi Si",
    )
    assert greeting == "尊敬的张三："
    assert body == "中文正文"
    assert closing == "此致\n敬礼\n\n李四"


def test_assemble_by_lang_en():
    """测试纯英文模式组装"""
    greeting, body, closing = EmailDrafterSkill._assemble_by_lang(
        "en",
        "尊敬的张三：", "Dear Zhang San,",
        "中文正文", "English body",
        "此致\n敬礼\n\n李四", "Best regards,\nLi Si",
    )
    assert greeting == "Dear Zhang San,"
    assert body == "English body"
    assert closing == "Best regards,\nLi Si"


def test_assemble_by_lang_bilingual():
    """测试双语模式组装"""
    greeting, body, closing = EmailDrafterSkill._assemble_by_lang(
        "bilingual",
        "尊敬的张三：", "Dear Zhang San,",
        "中文正文", "English body",
        "此致\n敬礼\n\n李四", "Best regards,\nLi Si",
    )
    assert "Dear Zhang San," in greeting
    assert "尊敬的张三：" in greeting
    assert "English body" in body
    assert "---" in body
    assert "中文正文" in body
    assert "Best regards,\nLi Si" in closing
    assert "此致\n敬礼\n\n李四" in closing


# ==========================================
# 6. 测试使用建议生成 (_generate_tips)
# ==========================================
def test_generate_tips_base(skill):
    """测试基础建议"""
    tips = skill._generate_tips("general", "semi_formal", "zh")
    assert "💡 发送前请检查收件人邮箱地址是否正确" in tips
    assert "💡 根据实际情况修改方括号 [ ] 中的占位内容" in tips


def test_generate_tips_by_tone(skill):
    """测试按语气添加建议"""
    tips = skill._generate_tips("general", "formal", "zh")
    assert "💡 正式邮件建议检查拼写和语法" in tips


def test_generate_tips_by_scenario(skill):
    """测试按场景添加建议"""
    tips = skill._generate_tips("meeting_invitation", "semi_formal", "zh")
    assert "💡 记得附上会议链接或地点信息" in tips

    tips = skill._generate_tips("leave_request", "semi_formal", "zh")
    assert "💡 建议抄送直接上级和 HR" in tips


def test_generate_tips_by_lang(skill):
    """测试按语言添加建议"""
    tips = skill._generate_tips("general", "semi_formal", "bilingual")
    assert "💡 双语邮件请确认两边内容一致" in tips


# ==========================================
# 7. 测试端到端 execute 方法
# ==========================================
def test_execute_zh_formal_leave(skill):
    """测试端到端：正式中文请假邮件"""
    inp = EmailDrafterInput(
        recipient_name="王经理",
        recipient_role="上级",
        scenario="leave_request",
        key_points=["请假时间：2026.5.1-2026.5.3", "工作已交接给李同事"],
        tone="formal",
        language="zh",
        sender_name="张三",
    )
    out = skill.execute(inp)

    assert out.subject == "请假申请"
    assert "尊敬的王经理：" in out.greeting
    assert "我因[个人原因/身体不适]" in out.body
    assert "1. 请假时间：2026.5.1-2026.5.3" in out.body
    assert "2. 工作已交接给李同事" in out.body
    assert "此致\n敬礼\n\n张三" in out.closing
    assert "💡 建议抄送直接上级和 HR" in out.tips


def test_execute_en_semi_formal_meeting(skill):
    """测试端到端：半正式英文会议邀请"""
    inp = EmailDrafterInput(
        recipient_name="Team",
        scenario="meeting_invitation",
        key_points=["Time: 2026-05-01 10:00", "Topic: Q2 Planning"],
        tone="semi_formal",
        language="en",
        sender_name="John",
    )
    out = skill.execute(inp)

    assert out.subject == "Meeting Invitation"
    assert "Hi Team," in out.greeting
    assert "You are cordially invited to the following meeting:" in out.body
    assert "1. Time: 2026-05-01 10:00" in out.body
    assert "2. Topic: Q2 Planning" in out.body
    assert "Best,\nJohn" in out.closing
    assert "💡 记得附上会议链接或地点信息" in out.tips


def test_execute_bilingual_casual_thank_you(skill):
    """测试端到端：随意双语感谢信"""
    inp = EmailDrafterInput(
        recipient_name="小李",
        scenario="thank_you",
        key_points=["感谢帮忙整理资料", "下次请你喝咖啡"],
        tone="casual",
        language="bilingual",
        sender_name="小王",
    )
    # ... existing code ...
    out = skill.execute(inp)

    assert "Hey 小李," in out.greeting
    assert "小李：" in out.greeting
    assert "I wanted to take a moment to express my sincere gratitude." in out.body
    assert "在此，我想向您表达诚挚的感谢。" in out.body
    assert "1. 感谢帮忙整理资料" in out.body
    assert "2. 下次请你喝咖啡" in out.body
    assert "Thanks,\n小王" in out.closing
    assert "谢谢！\n小王" in out.closing
    assert "💡 双语邮件请确认两边内容一致" in out.tips


# ... existing code ...


def test_execute_with_extra_notes(skill):
    """测试带额外备注的邮件"""
    inp = EmailDrafterInput(
        key_points="测试要点",
        extra_notes="这是额外备注内容",
    )
    out = skill.execute(inp)

    assert "📌 备注：这是额外备注内容" in out.body


def test_execute_default_recipient_sender(skill):
    """测试默认收件人和发件人"""
    inp = EmailDrafterInput(
        key_points="测试要点",
    )
    out = skill.execute(inp)

    assert "相关同事" in out.greeting
    assert "[您的名字]" in out.closing
if __name__ == "__main__":
    pytest.main()