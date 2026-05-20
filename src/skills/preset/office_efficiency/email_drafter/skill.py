"""
这是一个企业级邮件自动起草工具，能根据你输入的收件人、场景、要点，自动生成符合语气（正式 / 半正式 / 随意）和语言（中文 / 英文 / 中英双语）的邮件草稿，包含主题、称呼、正文、落款和使用建议。下面从功能概览、核心结构、类 / 方法详解、调用链路、代码质量五个维度深度解析。
一、功能概览
这个工具的核心能力是：
多场景支持：请假申请、项目汇报、会议邀请、事项跟进、感谢信、道歉信、通用邮件。
多语气 / 语言：支持正式 / 半正式 / 随意三种语气，中文 / 英文 / 中英双语三种语言。
智能要点解析：自动把换行 / 逗号 / 分号分隔的要点拆成列表，插入正文。
数据驱动模板：所有称呼、正文片段、落款、建议都用字典存储，易修改易扩展。
完善的输入校验：用 Pydantic 保证输入合法，要点不能为空。
二、核心结构设计
代码采用分层架构，核心结构如下：
plaintext
┌─────────────────────────────────────────────────────────────┐
│  1. 数据模型层（Pydantic）                                    │
│     - EmailDrafterInput（输入校验+要点归一化）                 │
│     - EmailDrafterOutput（结构化输出）                         │
├─────────────────────────────────────────────────────────────┤
│  2. 常量与模板层                                              │
│     - _SCENARIO_TOPIC（场景→主题词映射）                      │
│     - _GREETINGS/_CLOSINGS（称呼/落款模板）                   │
│     - _BODY_FRAGMENTS（正文片段模板）                          │
│     - _TIPS_*（使用建议模板）                                 │
├─────────────────────────────────────────────────────────────┤
│  3. 主技能类（EmailDrafterSkill）                             │
│     - execute（主入口，协调整个流程）                          │
│     - _parse_points/_build_subject/_generate_body（子方法）   │
│     - _assemble_by_lang/_generate_tips（子方法）              │
└─────────────────────────────────────────────────────────────┘
EmailDrafterSkill.execute()
  ├→ _parse_points(key_points)  [解析要点为列表]
  ├→ _build_subject(scenario, topic_key, sender=sender, date=date)  [构建主题]
  ├→ 生成中英文称呼（从_GREETINGS取模板+填充）
  ├→ _generate_body(points, scenario, tone, "zh")  [生成中文正文]
  ├→ _generate_body(points, scenario, tone, "en")  [生成英文正文]
  ├→ 生成中英文落款（从_CLOSINGS取模板+填充）
  ├→ _assemble_by_lang(lang, ...)  [按语言组装最终内容]
  ├→ （可选）添加额外备注到正文
  └→ _generate_tips(scenario, tone, lang)  [生成使用建议]
"""

import logging
import re
from datetime import date
from typing import ClassVar, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 作用：输入校验模型，同时负责要点归一化（把列表转成换行字符串，统一后续解析）。
# ═══════════════════════════════════════════════════════════════

class EmailDrafterInput(BaseModel):
    """邮件起草输入"""
    recipient_name: str = Field(default="", description="收件人姓名")
    recipient_role: str = Field(default="", description="收件人角色（如：客户、上级、同事）")
    scenario: Literal[
        "leave_request", "project_update", "meeting_invitation",
        "follow_up", "thank_you", "apology", "general"
    ] = Field(default="general", description="邮件场景")
    key_points: Union[str, list[str]] = Field(
        ...,
        description="邮件要点，字符串（换行/逗号分隔）或列表",
    )
    tone: Literal["formal", "semi_formal", "casual"] = Field(
        default="formal",
        description="语气风格"
    )
    language: Literal["zh", "en", "bilingual"] = Field(
        default="zh",
        description="邮件语言：zh=中文, en=英文, bilingual=中英双语"
    )
    sender_name: str = Field(default="", description="发件人姓名")
    extra_notes: str = Field(default="", description="额外备注")

    @field_validator("key_points", mode="before")
    @classmethod
    def _normalize_key_points(cls, v: Union[str, list[str]]) -> str:
        """将 list 统一转为换行分隔的字符串，便于后续统一解析"""
        if isinstance(v, list):
            return "\n".join(v)
        return v

    @field_validator("key_points")
    @classmethod
    def _validate_key_points_non_empty(cls, v: str) -> str:
        """确保解析后有实际内容"""
        parts = re.split(r'[\n,，;；]+', v)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            raise ValueError("key_points 不能为空或仅包含标点/分隔符")
        return v

# ═══════════════════════════════════════════════════════════════
# 作用：结构化输出模型，保证输出格式统一。
# ═══════════════════════════════════════════════════════════════
class EmailDrafterOutput(BaseModel):
    """邮件起草输出"""
    subject: str = Field(..., description="邮件主题")
    body: str = Field(..., description="邮件正文")
    greeting: str = Field(..., description="称呼")
    closing: str = Field(..., description="落款")
    tips: list[str] = Field(default_factory=list, description="使用建议")


# ═══════════════════════════════════════════════════════════════
# Skill
# ═══════════════════════════════════════════════════════════════

class EmailDrafterSkill(BaseSkill):
    """邮件起草技能"""

    name: str = "email_drafter"
    description: str = (
        "根据收件人、场景和要点自动起草正式邮件。"
        "支持请假申请、项目汇报、会议邀请、跟进提醒、感谢信、道歉信等场景。"
    )
    triggers: list[str] = [
        "邮件", "写邮件", "起草邮件", "email", "draft email",
        "帮我写封邮件", "回复邮件", "请假邮件", "汇报邮件",
        "邀请邮件", "跟进邮件", "感谢信", "道歉邮件",
    ]
    version: str = "1.1.0"
    author: str = "EnterpriseLearningAgent"
    changelog: list[dict[str, str]] = [
        {"version": "1.1.0", "date": "2026-04-30", "note": "合并中英文正文生成、数据化模板片段、增强校验"},
        {"version": "1.0.0", "date": "2026-04-01", "note": "初始版本，基于模板+要点插入的邮件起草"},
    ]
    input_schema = EmailDrafterInput
    output_schema = EmailDrafterOutput

    # ═══════════════════════════════════════════════════════════
    # 常量
    # ═══════════════════════════════════════════════════════════

    _DEFAULT_RECIPIENT: ClassVar[str] = "相关同事"
    _DEFAULT_SENDER: ClassVar[str] = "[您的名字]"
    _DEFAULT_TOPIC: ClassVar[str] = "邮件"

    # ── 场景→主题词映射 ──
    _SCENARIO_TOPIC: ClassVar[dict[str, dict[str, str]]] = {
        "leave_request":       {"zh": "请假申请",              "en": "Leave Request"},
        "project_update":      {"zh": "项目进展汇报",          "en": "Project Update"},
        "meeting_invitation":  {"zh": "会议邀请",              "en": "Meeting Invitation"},
        "follow_up":           {"zh": "事项跟进",              "en": "Follow-up"},
        "thank_you":           {"zh": "感谢信",                "en": "Thank You"},
        "apology":             {"zh": "致歉信",                "en": "Apology"},
        "general":             {"zh": "[主题]",                "en": "[Subject]"},
    }

    # ── 称呼模板 ──
    _GREETINGS: ClassVar[dict] = {
        "formal":      {"zh": "尊敬的{recipient}：",  "en": "Dear {recipient},"},
        "semi_formal": {"zh": "{recipient}，您好：",   "en": "Hi {recipient},"},
        "casual":      {"zh": "{recipient}：",        "en": "Hey {recipient},"},
    }

    # ── 落款模板 ──
    _CLOSINGS: ClassVar[dict] = {
        "formal":      {"zh": "此致\n敬礼\n\n{sender}",   "en": "Best regards,\n{sender}"},
        "semi_formal": {"zh": "祝好！\n{sender}",          "en": "Best,\n{sender}"},
        "casual":      {"zh": "谢谢！\n{sender}",          "en": "Thanks,\n{sender}"},
    }

    # ── 正文片段（按语言组织、按位置索引） ──
    _BODY_FRAGMENTS: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "zh": {
            "openers": {
                "formal":      "您好！\n\n",
                "semi_formal": "您好！\n\n",
                "casual":      "你好！\n\n",
            },
            "scenario_intros": {
                "leave_request":       "我因[个人原因/身体不适]，特此申请请假，详细信息如下：\n",
                "project_update":      "现将近期项目进展汇报如下：\n",
                "meeting_invitation":  "特此邀请您参加以下会议：\n",
                "follow_up":           "冒昧打扰，想跟进一下以下事项：\n",
                "thank_you":           "在此，我想向您表达诚挚的感谢。\n",
                "apology":             "首先，请允许我就[相关事宜]表达诚挚的歉意。\n",
                "general":             "",
            },
            "scenario_closers": {
                "leave_request":       "\n感谢您的理解与批准。",
                "project_update":      "\n请查收，期待您的反馈。",
                "meeting_invitation":  "\n烦请回复确认能否参加，谢谢！",
                "follow_up":           "\n期待您的回复，谢谢！",
                "thank_you":           "\n再次感谢！",
                "apology":             "\n再次表示歉意，我将尽力弥补。",
                "general":             "",
            },
            "closers": {
                "formal":      "\n如有任何问题，请随时与我联系。",
                "semi_formal": "\n有任何问题请随时联系我。",
                "casual":      "\n有问题随时找我～",
            },
        },
        "en": {
            "openers": {
                "formal":      "I hope this email finds you well.\n\n",
                "semi_formal": "Hope you're doing well.\n\n",
                "casual":      "Hi!\n\n",
            },
            "scenario_intros": {
                "leave_request":       "I am writing to request leave for the following period:\n",
                "project_update":      "Here is the latest project update:\n",
                "meeting_invitation":  "You are cordially invited to the following meeting:\n",
                "follow_up":           "I'd like to follow up on the matter below:\n",
                "thank_you":           "I wanted to take a moment to express my sincere gratitude.\n",
                "apology":             "Please accept my sincere apologies regarding the matter.\n",
                "general":             "",
            },
            "scenario_closers": {
                "leave_request":       "\nThank you for your understanding.",
                "project_update":      "\nLooking forward to your feedback.",
                "meeting_invitation":  "\nKindly RSVP at your earliest convenience.",
                "follow_up":           "\nLooking forward to hearing from you.",
                "thank_you":           "\nThanks again!",
                "apology":             "\nI will do my best to make it right.",
                "general":             "",
            },
            "closers": {
                "formal":      "\nPlease don't hesitate to reach out if you have any questions.",
                "semi_formal": "\nLet me know if you have any questions.",
                "casual":      "\nLet me know!",
            },
        },
    }

    # ── Tips 数据驱动 ──
    _TIPS_BASE: ClassVar[list[str]] = [
        "💡 发送前请检查收件人邮箱地址是否正确",
        "💡 根据实际情况修改方括号 [ ] 中的占位内容",
    ]
    _TIPS_BY_TONE: ClassVar[dict[str, list[str]]] = {
        "formal": ["💡 正式邮件建议检查拼写和语法"],
    }
    _TIPS_BY_SCENARIO: ClassVar[dict[str, list[str]]] = {
        "meeting_invitation": ["💡 记得附上会议链接或地点信息"],
        "leave_request":      ["💡 建议抄送直接上级和 HR"],
    }
    _TIPS_BY_LANG: ClassVar[dict[str, list[str]]] = {
        "bilingual": ["💡 双语邮件请确认两边内容一致"],
    }

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def execute(self, input_data: EmailDrafterInput) -> EmailDrafterOutput:
        logger.info("EmailDrafterSkill.execute | scenario=%s tone=%s lang=%s",
                     input_data.scenario, input_data.tone, input_data.language)

        recipient = input_data.recipient_name or self._DEFAULT_RECIPIENT
        sender = input_data.sender_name or self._DEFAULT_SENDER
        tone = input_data.tone
        lang = input_data.language
        scenario = input_data.scenario

        # 1. 解析要点
        points = self._parse_points(input_data.key_points)
        logger.debug("Parsed %d points", len(points))

        # 2. 主题词
        topic_key = "zh" if lang in ("zh", "bilingual") else "en"
        topic = self._SCENARIO_TOPIC.get(scenario, {}).get(topic_key, self._DEFAULT_TOPIC)

        # 3. 生成主题
        today = date.today().isoformat()
        subject = self._build_subject(scenario, topic_key, sender=sender, date=today)

        # 4. 生成称呼
        greeting_tmpl = self._GREETINGS[tone]
        greeting_zh = greeting_tmpl["zh"].format(recipient=recipient)
        greeting_en = greeting_tmpl["en"].format(recipient=recipient)

        # 5. 生成正文（合并后的通用方法）
        body_zh = self._generate_body(points, scenario, tone, "zh")
        body_en = self._generate_body(points, scenario, tone, "en")

        # 6. 生成落款
        closing_tmpl = self._CLOSINGS[tone]
        closing_zh = closing_tmpl["zh"].format(sender=sender)
        closing_en = closing_tmpl["en"].format(sender=sender)

        # 7. 按语言组装
        greeting, body, closing = self._assemble_by_lang(
            lang,
            greeting_zh, greeting_en,
            body_zh, body_en,
            closing_zh, closing_en,
        )

        # 8. 附加备注
        if input_data.extra_notes:
            body += f"\n\n📌 备注：{input_data.extra_notes}"

        # 9. 建议
        tips = self._generate_tips(scenario, tone, lang)

        logger.info("EmailDrafterSkill.execute | done, subject=%s", subject)
        return EmailDrafterOutput(
            subject=subject,
            body=body,
            greeting=greeting,
            closing=closing,
            tips=tips,
        )

    # ═══════════════════════════════════════════════════════════
    # 子方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _parse_points(key_points: str) -> list[str]:
        """解析要点字符串或列表为 list[str]"""
        parts = re.split(r'[\n,，;；]+', key_points)
        return [p.strip() for p in parts if p.strip()]

    def _build_subject(self, scenario: str, topic_key: str, *, sender: str, date: str) -> str:
        """构建邮件主题"""
        tmpl = self._SCENARIO_TOPIC.get(scenario, self._SCENARIO_TOPIC["general"])
        # topic_key ∈ {"zh", "en"}
        raw = tmpl.get(topic_key, self._DEFAULT_TOPIC)
        return raw.replace("{sender}", sender).replace("{date}", date)

    def _generate_body(
        self,
        points: list[str],
        scenario: str,
        tone: str,
        lang: str,
    ) -> str:
        """统一的正文生成（zh / en 通过 lang 参数区分）"""
        frag = self._BODY_FRAGMENTS[lang]

        body = frag["openers"].get(tone, "")
        body += frag["scenario_intros"].get(scenario, "")

        for i, point in enumerate(points, 1):
            body += f"{i}. {point}\n"

        body += frag["scenario_closers"].get(scenario, "")
        body += frag["closers"].get(tone, "")

        return body

    @staticmethod
    def _assemble_by_lang(
        lang: str,
        greeting_zh: str, greeting_en: str,
        body_zh: str, body_en: str,
        closing_zh: str, closing_en: str,
    ) -> tuple[str, str, str]:
        """按语言模式组装最终的 greeting / body / closing"""
        if lang == "zh":
            return greeting_zh, body_zh, closing_zh

        if lang == "en":
            return greeting_en, body_en, closing_en

        # bilingual
        zh_part = body_zh.strip()
        en_part = body_en.strip()
        if zh_part and en_part:
            body = f"{en_part}\n\n---\n\n{zh_part}"
        else:
            body = zh_part or en_part

        greeting = f"{greeting_en}\n（{greeting_zh}）"
        closing = f"{closing_en}\n\n{closing_zh}"
        return greeting, body, closing

    def _generate_tips(self, scenario: str, tone: str, lang: str) -> list[str]:
        """数据驱动生成使用建议"""
        tips = list(self._TIPS_BASE)
        tips.extend(self._TIPS_BY_TONE.get(tone, []))
        tips.extend(self._TIPS_BY_SCENARIO.get(scenario, []))
        tips.extend(self._TIPS_BY_LANG.get(lang, []))
        return tips