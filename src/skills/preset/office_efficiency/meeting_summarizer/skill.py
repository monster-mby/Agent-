"""
这是一个企业级会议纪要自动生成技能，能将杂乱的会议转录文本或要点，自动整理成包含「议题、决策、待办、负责人、截止日期、优先级」的结构化 Markdown 纪要。下面从功能概览、核心结构设计、类 / 方法深度详解、方法调用链路、代码质量评价五个维度深度解析。
一、功能概览
这个工具的核心能力是：
多维度信息提取：自动从文本中提取「讨论议题、决策事项、待办事项」三大核心内容。
智能正则匹配：支持中文人名匹配、截止日期提取（相对 / 绝对日期）、中英文混合模式识别。
优先级自动推断：根据任务关键词（如 “紧急”“asap”→ 高优；“有空”“later”→ 低优）自动标注优先级。
结构化 Markdown 输出：用 Jinja2 模板渲染美观的纪要（含 Emoji 图标、待办表格），未安装 Jinja2 时自动回退手动拼接。
完善的容错与兜底：各提取环节都有兜底文案（如 “未检测到明确决策项”），保证输出完整性。
二、核心结构设计
代码采用分层架构，核心结构如下：
plaintext
┌─────────────────────────────────────────────────────────────┐
│  1. 外部化配置层（MeetingRules）                              │
│     - TODO_PATTERNS/DEADLINE_PATTERNS（正则规则）            │
│     - DECISION_KEYWORDS/TOPIC_KEYWORDS（关键词规则）          │
│     - PRIORITY_HIGH/LOW（优先级推断词）                       │
│     - INVALID_OWNERS（无效负责人过滤词）                       │
├─────────────────────────────────────────────────────────────┤
│  2. 数据模型层（Pydantic）                                    │
│     - ActionItem（待办事项结构）                               │
│     - MeetingSummarizerInput（输入校验+参会人归一化）          │
│     - MeetingSummarizerOutput（结构化输出）                   │
├─────────────────────────────────────────────────────────────┤
│  3. Jinja2 模板层                                             │
│     - MD_TEMPLATE（Markdown 纪要模板，含 Emoji/表格）         │
├─────────────────────────────────────────────────────────────┤
│  4. 主技能类（MeetingSummarizerSkill）                        │
│     - execute（主入口，协调整个流程）                          │
│     - _extract_topics/_extract_decisions/_extract_action_items │
│     - _generate_summary/_generate_markdown（子方法）          │
└─────────────────────────────────────────────────────────────┘

MeetingSummarizerSkill.execute()
  ├→ _extract_topics
  │   ├→ _iter_blocks（切分段落）
  │   └→ _deduplicate_preserve_order（去重）
  ├→ _extract_decisions
  │   ├→ _iter_blocks（切分段落）
  │   └→ _deduplicate_preserve_order（去重）
  ├→ _extract_action_items
  │   ├→ _iter_blocks（切分段落）
  │   ├→ _extract_deadline（提取截止日期）
  │   └→ _infer_priority（推断优先级）
  ├→ _generate_summary（生成摘要）
  └→ _generate_markdown（渲染 Markdown）
      └→ _generate_markdown_fallback（兜底拼接）
"""

import re
from typing import ClassVar

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.skills.base.base_skill import BaseSkill


# ──────────────────────────────────────────────
# 外部化配置（未来可迁移至 YAML/JSON）
#作用：集中管理所有提取规则（正则、关键词），避免硬编码，方便后续调整。
# ──────────────────────────────────────────────
class MeetingRules:
    """会议提取规则配置 —— 集中管理，方便调整"""

    # ── 待办事项匹配模式 ──
    TODO_PATTERNS: ClassVar[list[str]] = [
        # 中文模式（关键修复：.+? 匹配中文人名）
        r'(.+?)需要[：:]?\s*(.+)',
        r'(.+?)负责[：:]?\s*(.+)',
        r'(.+?)跟进[：:]?\s*(.+)',
        r'(.+?)来做[：:]?\s*(.+)',
        r'(.+?)去[做干搞弄][：:]?\s*(.+)',
        r'待办[：:]?\s*(.+?)\s*[-–—]\s*(.+)',
        r'任务[：:]?\s*(.+?)\s*[-–—]\s*(.+)',
        # 英文模式
        r'action\s*(?:item)?[：:]\s*(.+?)\s*[-–—]\s*(.+)',
        r'(.+?)\s+will\s+(.+)',
        r'(.+?)\s+should\s+(.+)',
        r'@(\S+?)\s+(.+)',
        # 通用模式：名称：任务（放在最后，避免过度匹配）
        r'^[-*•\d.\s]*(\S{1,10})[：:]\s+(.+)',
    ]

    # ── 截止日期模式 ──
    DEADLINE_PATTERNS: ClassVar[list[str]] = [
        # 相对日期
        r'(今天|明天|后天|大后天)',
        r'(本周[一二三四五六日天]|下周[一二三四五六日天])',
        r'(本?周[一二三四五六日天])',
        # 绝对日期
        r'(\d{1,2}月\d{1,2}[日号])',
        r'(\d{1,2}\.\d{1,2})',
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        # 修饰：XX前/截止/DDL
        r'(\d{1,2}/\d{1,2})\s*[之前]',
        r'(?:截止|DDL|deadline)[：:]?\s*(\S{2,20})',
        r'(?:by|before)\s+(\S{2,20})',
    ]

    # ── 决策关键词 ──
    DECISION_KEYWORDS: ClassVar[list[str]] = [
        "决定", "确定", "决议", "定下来", "就这样", "达成一致", "一致同意",
        "最终方案", "最终决定", "拍板", "敲定", "定夺",
        "decided", "decision", "agreed", "conclusion", "resolved",
    ]

    # ── 议题关键词 ──
    TOPIC_KEYWORDS: ClassVar[list[str]] = [
        "议题", "讨论", "关于", "汇报", "议题：", "主题：", "专题",
        "agenda", "topic", "discussion", "regarding",
        "第一", "第二", "第三", "首先", "其次", "最后",
    ]

    # ── 优先级推断词 ──
    PRIORITY_HIGH: ClassVar[list[str]] = [
        "紧急", "尽快", "立刻", "马上", "今天", "urgent", "asap", "立即",
        "上线", "发布", "deadline", "截止", "bug", "故障", "事故",
    ]
    PRIORITY_LOW: ClassVar[list[str]] = [
        "有空", "抽空", "考虑", "或许", "maybe", "later", "以后", "优化",
        "nice to have", "不急", "慢慢",
    ]

    # ── 无效负责人过滤 ──
    INVALID_OWNERS: ClassVar[set[str]] = {
        "i", "we", "you", "they", "he", "she", "it",
        "我", "我们", "你们", "他们", "她", "它", "大家",
        "自己", "某人", "谁", "有人",
    }

    # ── 议题编号前缀（扩展至十） ──
    TOPIC_NUM_PATTERN: ClassVar[str] = r'^(?:第)?[#\d一二三四五六七八九十]+[、.．)\s]+'


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────
class ActionItem(BaseModel):
    """待办事项"""
    task: str
    owner: str = "待指定"
    deadline: str = "待定"
    priority: str = "medium"  # high / medium / low

# ──────────────────────────────────────────────
# 继承自 BaseSkill（自定义基类），是核心逻辑实现类，包含正则预编译、Jinja2 初始化、各环节提取方法。
# ──────────────────────────────────────────────
class MeetingSummarizerInput(BaseModel):
    """会议纪要输入"""
    meeting_transcript: str = Field(
        ...,
        description="会议转录文本或要点记录",
        min_length=10,
    )
    meeting_title: str = Field(default="会议", description="会议标题")
    attendees: str | list[str] = Field(
        default="",
        description="参会人员，逗号分隔（支持字符串或列表）",
    )
    date: str = Field(default="", description="会议日期")

    @field_validator("attendees", mode="before")
    @classmethod
    def _normalize_attendees(cls, v: str | list[str]) -> list[str]:
        """自动将逗号分隔字符串转换为列表"""
        if isinstance(v, list):
            return [a.strip() for a in v if a.strip()]
        if isinstance(v, str) and v.strip():
            return [a.strip() for a in v.split(",") if a.strip()]
        return []


class MeetingSummarizerOutput(BaseModel):
    """会议纪要输出"""
    title: str = Field(...)
    date: str = Field(...)
    attendees: list[str] = Field(default_factory=list)
    topics: list[str] = Field(..., description="讨论议题")
    decisions: list[str] = Field(..., description="决策事项")
    action_items: list[ActionItem] = Field(..., description="待办事项")
    summary: str = Field(..., description="会议摘要")
    markdown_output: str = Field(..., description="Markdown 格式的完整纪要")


# ──────────────────────────────────────────────
# Jinja2 Markdown 模板（可选，无依赖时回退拼接）
# ──────────────────────────────────────────────
MD_TEMPLATE = """# 📋 {{ title }}

**日期：** {{ date }}

**参会人员：** {% if attendees %}{{ attendees | join(', ') }}{% else %}待补充{% endif %}

---

> {{ summary }}

## 🗣️ 讨论议题

{% for topic in topics %}
{{ loop.index }}. {{ topic }}
{%- endfor %}

## ✅ 决策事项

{% for d in decisions %}
- {{ d }}
{%- endfor %}

## 📌 待办事项

| 任务 | 负责人 | 优先级 | 截止日期 |
|------|--------|--------|----------|
{% for a in action_items -%}
| {{ a.task }} | {{ a.owner }} | {{ {'high':'🔴','medium':'🟡','low':'🟢'}.get(a.priority, '⚪') }} {{ a.priority }} | {{ a.deadline }} |
{% endfor %}

---

*此纪要由系统自动生成，请根据实际情况修改补充。*"""


# ──────────────────────────────────────────────
# Skill 主体
# ──────────────────────────────────────────────
class MeetingSummarizerSkill(BaseSkill):
    """会议纪要技能

    将会议转录文本或要点快速整理为结构化纪要：
    议题、讨论、决策、待办、负责人、截止日期一条龙。
    """

    name: str = "meeting_summarizer"
    description: str = (
        "将会议转录文本或要点快速整理为结构化纪要："
        "议题、讨论、决策、待办、负责人、截止日期一条龙。"
    )
    triggers: list[str] = [
        "会议纪要", "会议记录", "meeting summary", "开会纪要",
        "会议总结", "整理会议", "会议要点", "meeting notes",
        "总结会议", "纪要", "minutes",
    ]
    version: str = "1.1.0"
    author: str = "EnterpriseLearningAgent"
    changelog: str = (
        "v1.1.0: P0 修复中文人名匹配、摘要统计占位符；"
        "新增截止日期提取、正则预编译、Jinja2 模板、Pydantic validator、loguru"
    )
    input_schema = MeetingSummarizerInput
    output_schema = MeetingSummarizerOutput

    # ── 预编译正则（类级别，只编译一次） ──
    _todo_regex: ClassVar[list[re.Pattern]] = [
        re.compile(p, re.IGNORECASE) for p in MeetingRules.TODO_PATTERNS
    ]
    _deadline_regex: ClassVar[list[re.Pattern]] = [
        re.compile(p, re.IGNORECASE) for p in MeetingRules.DEADLINE_PATTERNS
    ]
    _topic_num_regex: ClassVar[re.Pattern] = re.compile(MeetingRules.TOPIC_NUM_PATTERN)

    # ── Jinja2 支持 ──
    _jinja2_env: ClassVar[object | None] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._init_jinja2()

    @classmethod
    def _init_jinja2(cls):
        """尝试加载 Jinja2，失败则回退手动拼接"""
        if cls._jinja2_env is not None:
            return
        try:
            from jinja2 import Environment, BaseLoader
            cls._jinja2_env = Environment(loader=BaseLoader())
            cls._jinja2_template = cls._jinja2_env.from_string(MD_TEMPLATE)
            logger.info("Jinja2 模板引擎已启用")
        except ImportError:
            cls._jinja2_env = False  # 标记为不可用
            logger.info("Jinja2 未安装，使用内置 Markdown 生成")

    # ────────────── 主入口 ──────────────

    def execute(self, input_data: MeetingSummarizerInput) -> MeetingSummarizerOutput:
        text = input_data.meeting_transcript
        title = input_data.meeting_title or "会议纪要"
        date = input_data.date or "待补充"

        # attendees 已在 validator 中转换为列表
        attendees = input_data.attendees or []

        logger.info(f"开始处理会议纪要: '{title}' | 文本长度 {len(text)} 字符")

        topics = self._extract_topics(text)
        logger.debug(f"提取到 {len(topics)} 个议题")

        decisions = self._extract_decisions(text)
        logger.debug(f"提取到 {len(decisions)} 个决策")

        action_items = self._extract_action_items(text)
        logger.debug(f"提取到 {len(action_items)} 个待办")

        summary = self._generate_summary(title, topics, decisions, action_items)

        markdown = self._generate_markdown(
            title, date, attendees, topics, decisions, action_items, summary
        )

        logger.info(f"会议纪要生成完成: {summary}")

        return MeetingSummarizerOutput(
            title=title,
            date=date,
            attendees=attendees,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            summary=summary,
            markdown_output=markdown,
        )

    # ────────────── 议题提取 ──────────────

    def _extract_topics(self, text: str) -> list[str]:
        """提取讨论议题 —— 编号行 + 关键词行 + 段落首句"""
        topics: list[str] = []
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 匹配编号行：1. / 一、 / 第一、
            if self._topic_num_regex.match(stripped):
                clean = self._topic_num_regex.sub("", stripped).strip()
                if len(clean) > 3:
                    topics.append(clean)
                    continue

                # 匹配关键词行
                for kw in MeetingRules.TOPIC_KEYWORDS:
                    if kw in stripped:
                        clean = re.sub(
                            rf'({re.escape(kw)})\s*[：:、]?\s*',
                            "",
                            stripped,
                            flags=re.IGNORECASE,
                        ).strip()
                        if len(clean) > 3 and clean not in topics:
                            topics.append(clean[:80])
                        break

        # 兜底：取前 20 行中包含关键词的句子
        if not topics:
            for line in lines[:20]:
                stripped = line.strip()
                if (
                    any(kw in stripped for kw in MeetingRules.TOPIC_KEYWORDS)
                    and len(stripped) > 5
                ):
                    topics.append(stripped[:80])

        if not topics:
            topics = ["（待补充议题）"]

        return self._deduplicate_preserve_order(topics)[:8]

    # ────────────── 决策提取 ──────────────

    def _extract_decisions(self, text: str) -> list[str]:
        """提取决策事项"""
        decisions: list[str] = []

        for block in self._iter_blocks(text):
            block_lower = block.lower()
            if any(kw in block_lower for kw in MeetingRules.DECISION_KEYWORDS):
                clean = block.strip()
                clean = re.sub(r'^[#\-\*\d\.、\s]+', '', clean)
                if len(clean) > 5:
                    decisions.append(clean[:150])

        if not decisions:
            decisions = ["（未检测到明确决策项）"]

        return self._deduplicate_preserve_order(decisions)[:10]

    # ────────────── 待办提取 ──────────────

    def _extract_action_items(self, text: str) -> list[ActionItem]:
        """提取待办事项（段落级 + 行级）"""
        items: list[ActionItem] = []
        seen: set[tuple[str, str]] = set()

        for block in self._iter_blocks(text):
            for regex in self._todo_regex:
                for match in regex.finditer(block):
                    groups = match.groups()
                    if len(groups) < 2:
                        continue

                    g1, g2 = groups[0].strip(), groups[1].strip()

                    if g1.lower() in MeetingRules.INVALID_OWNERS:
                        continue

                    owner, task = g1, g2
                    owner = re.sub(r'^[-*•\d]+[.、．)\s]*', '', owner).strip()

                    if not task or len(owner) > 20:
                        continue
                    if owner.lower() in MeetingRules.INVALID_OWNERS:
                        continue
                    if len(task) < 2:
                        continue

                    key = (owner, task)
                    if key in seen:
                        continue
                    seen.add(key)

                    # 🔧 修复：只在匹配行内提取截止日期
                    matched_line = self._get_matched_line(block, match)
                    deadline = self._extract_deadline(matched_line)
                    priority = self._infer_priority(task)
                    items.append(ActionItem(
                        task=task[:120],
                        owner=owner,
                        deadline=deadline,
                        priority=priority,
                    ))

        if not items:
            items.append(ActionItem(
                task="（请补充待办事项）",
                owner="待指定",
                priority="medium",
            ))

        return items[:15]

    @staticmethod
    def _swap_or_skip(g1: str, g2: str) -> tuple[str | None, str | None]:
        """尝试交换 owner/task，如果 g2 是无效代词则跳过"""
        if g2.lower() in MeetingRules.INVALID_OWNERS:
            return None, None
        return g2, g1  # swap

    # ────────────── 截止日期提取 ──────────────

    def _extract_deadline(self, text: str) -> str:
        """从上下文中提取截止日期"""
        for regex in self._deadline_regex:
            match = regex.search(text)
            if match:
                result = match.group(1).strip()[:30]
                #  确保返回值不是 None 或空字符串
                if result:
                    return result
        return "待定"

    # ────────────── 优先级推断 ──────────────

    @staticmethod
    def _infer_priority(task: str) -> str:
        """推测待办优先级"""
        task_lower = task.lower()
        if any(kw in task_lower for kw in MeetingRules.PRIORITY_HIGH):
            return "high"
        if any(kw in task_lower for kw in MeetingRules.PRIORITY_LOW):
            return "low"
        return "medium"

    # ────────────── 摘要生成 ──────────────

    @staticmethod
    def _generate_summary(
        title: str,
        topics: list[str],
        decisions: list[str],
        items: list[ActionItem],
    ) -> str:
        """生成会议摘要（排除占位符）"""
        n_topics = len([t for t in topics if not t.startswith("（待补充")])
        n_decisions = len([d for d in decisions if not d.startswith("（未检测到")])
        n_actions = len([a for a in items if not a.task.startswith("（请补充")])

        return (
            f"本次「{title}」会议共讨论了 {n_topics} 个议题，"
            f"达成 {n_decisions} 项决策，"
            f"形成 {n_actions} 项待办。"
        )

    # ────────────── Markdown 生成 ──────────────

    def _generate_markdown(
        self,
        title: str,
        date: str,
        attendees: list[str],
        topics: list[str],
        decisions: list[str],
        action_items: list[ActionItem],
        summary: str,
    ) -> str:
        """生成 Markdown（优先 Jinja2，回退手动拼接）"""
        if self._jinja2_env and self._jinja2_env is not False:
            try:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                return self._jinja2_template.render(
                    title=title,
                    date=date,
                    attendees=attendees,
                    topics=topics,
                    decisions=decisions,
                    action_items=[
                        {
                            "task": a.task,
                            "owner": a.owner,
                            "priority": a.priority,
                            "deadline": a.deadline,
                        }
                        for a in action_items
                    ],
                    summary=summary,
                    priority_icon=priority_icon,
                )
            except Exception as e:
                logger.warning(f"Jinja2 渲染失败，回退手动拼接: {e}")

        return self._generate_markdown_fallback(
            title, date, attendees, topics, decisions, action_items, summary
        )

    @staticmethod
    def _generate_markdown_fallback(
        title: str,
        date: str,
        attendees: list[str],
        topics: list[str],
        decisions: list[str],
        action_items: list[ActionItem],
        summary: str,
    ) -> str:
        """手动拼接 Markdown（无 Jinja2 时的回退方案）"""
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}

        lines = [
            f"# 📋 {title}",
            "",
            f"**日期：** {date}",
            "",
            f"**参会人员：** {', '.join(attendees) if attendees else '待补充'}",
            "",
            "---",
            "",
            f"> {summary}",
            "",
            "## 🗣️ 讨论议题",
            "",
        ]
        for i, topic in enumerate(topics, 1):
            lines.append(f"{i}. {topic}")

        lines += [
            "",
            "## ✅ 决策事项",
            "",
        ]
        for d in decisions:
            lines.append(f"- {d}")

        lines += [
            "",
            "## 📌 待办事项",
            "",
            "| 任务 | 负责人 | 优先级 | 截止日期 |",
            "|------|--------|--------|----------|",
        ]
        for a in action_items:
            icon = priority_icon.get(a.priority, "⚪")
            lines.append(
                f"| {a.task} | {a.owner} | {icon} {a.priority} | {a.deadline} |"
            )

        lines += [
            "",
            "---",
            "",
            "*此纪要由系统自动生成，请根据实际情况修改补充。*",
        ]
        return "\n".join(lines)

    # ────────────── 工具方法 ──────────────

    @staticmethod
    def _iter_blocks(text: str) -> list[str]:
        """
        将文本切分为段落块，支持跨行分析。
        空行作为段落分隔符。
        """
        # 先按空行拆段落
        paragraphs = re.split(r'\n\s*\n', text)
        blocks = []
        for para in paragraphs:
            para = para.strip()
            if para:
                blocks.append(para)
        # 也保留原始行（用于精确行级匹配）
        return blocks

    @staticmethod
    def _get_matched_line(block: str, match: re.Match) -> str:
        """从 block 中提取 match 所在的那一行"""
        start = match.start()
        # 向前找行首
        line_start = block.rfind('\n', 0, start) + 1
        # 向后找行尾
        line_end = block.find('\n', start)
        if line_end == -1:
            line_end = len(block)
        return block[line_start:line_end]

    @staticmethod
    def _deduplicate_preserve_order(items: list[str]) -> list[str]:
        """去重并保持原始顺序"""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result