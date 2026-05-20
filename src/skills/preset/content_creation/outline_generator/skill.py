"""
Outline Generator Skill — 大纲生成（优化版 v2.0.0）

输入主题或关键词，输出结构化 Markdown 大纲。
支持指定深度、领域、编号风格、语言。

v2.0.0 优化改动:
  - P0-1: 三级大纲预定义真实内容，与二级条目语义关联（非固定文案）
  - P0-2: {{TOPIC}} 占位符替代 "XXX"，消除子串误替换风险
  - P0-3: 分层统计 section_count / second_level_count / third_level_count
  - P1-4: _TEMPLATES 深拷贝到实例属性，线程安全
  - P1-5: 模板 YAML 外部化，添加领域/语言只需改配置，不动代码
  - P1-6: Jinja2 模板引擎替代手动字符串拼接（优雅降级到纯 Python）
  - P2-7: 中英文双模板，output_language 参数
  - P2-8: Markdown 顶部 HTML 注释元信息（版本/参数可追溯）
  - P2-9: numbering_style 参数（arabic_dot / chinese / roman）
  - P3-11: extra_instructions 增加 max_length=500
  - P3-12: Literal → Enum (DepthLevel, Domain, NumberingStyle, OutputLanguage)
  - P3-14: 硬编码字符串提取为模块常量
  - P3-15: 添加 __repr__

  1. 用户调用 generate_outline("Python", depth=3, ...)
   └─ 核心作用：用户侧一键调用入口，传入主题、深度、领域、编号风格等生成参数，触发完整的大纲生成流程
   ↓
2. 内部创建 OutlineGeneratorSkill 实例
   └─ 核心作用：初始化大纲生成的核心业务对象，为后续全流程准备执行环境
   ↓
3. 执行 __init__() 实例构造方法
   ├─ 3.1 调用 _load_templates()
   │  └─ 核心作用：加载大纲模板配置（外部YAML文件优先，内置默认模板兜底），通过深拷贝保证多实例/多线程场景下的数据隔离安全
   └─ 3.2 调用 _compile_jinja_template()
      └─ 核心作用：预编译Jinja2渲染模板（仅Jinja2库可用时执行），避免每次生成都重复编译，提升执行性能
   ↓
4. 执行 execute() 核心主方法（全流程业务编排中枢）
   ├─ 4.1 参数清洗与标准化
   │  └─ 核心作用：将Enum枚举值转为原始字符串、对文本参数去空格处理，统一所有入参格式，消除后续处理的格式差异
   ├─ 4.2 调用 _resolve_language()
   │  └─ 核心作用：根据用户指定规则+主题内容，判定最终输出语言（中文/英文），采用「规则优先+第三方库兜底」的策略，保证判定准确性
   ├─ 4.3 模板数据匹配
   │  └─ 核心作用：根据判定好的输出语言+用户指定的领域，从已加载的模板库中，匹配对应的大纲结构原始数据
   ├─ 4.4 调用 _build_sections()
   │  ├─ 核心作用：遍历原始模板，完成主题占位符替换、章节编号生成、各层级条目数量统计，输出渲染所需的标准化结构化数据
   │  └─ 循环内调用 _format_number()
   │     ├─ 核心作用：根据用户指定的编号风格，生成对应格式的章节编号
   │     └─ 按需调用 _num_to_chinese()
   │        └─ 核心作用：中文编号场景下，将阿拉伯数字转为对应的中文数字
   ├─ 4.5 调用 _render_markdown()
   │  ├─ 核心作用：基于结构化数据渲染最终的Markdown格式大纲，双方案实现优雅降级，保证核心功能可用
   │  ├─ 优先执行 _render_via_jinja2()
   │  │  └─ 核心作用：Jinja2库可用时，通过预编译的模板引擎，高效渲染出规范的Markdown文本
   │  └─ 兜底执行 _render_via_python()
   │     └─ 核心作用：Jinja2库不可用时，通过纯Python代码手动拼接字符串，保证渲染功能正常可用
   └─ 4.6 结果统计与封装
      └─ 核心作用：统计大纲总条目数，将最终Markdown内容、分层统计数据、入参追溯信息，组装为标准化的输出模型
   ↓
5. 返回结果给用户
   └─ 核心作用：返回结构化的OutlineGeneratorOutput结果，包含最终Markdown大纲、各层级统计数据、生成参数等完整信息
"""

from __future__ import annotations

import copy
import logging
from enum import IntEnum
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field, field_validator

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ============================================================================
# Optional Dependency: Jinja2
# ============================================================================
try:
    from jinja2 import Template as Jinja2Template
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False
    logger.info("Jinja2 未安装，将使用纯 Python 字符串构建。pip install jinja2 可获得更灵活的模板支持。")


# ============================================================================
# Optional Dependency: langdetect
# ============================================================================
try:
    from langdetect import detect as lang_detect
    from langdetect.lang_detect_exception import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.info("langdetect 未安装，语言检测将使用 CJK 字符统计。pip install langdetect 可获得更精准的语言识别。")

# ============================================================================
# Enums — 替代 Literal，更具自解释性
# ============================================================================
class DepthLevel(IntEnum):
    """大纲深度级别"""
    SHALLOW = 1   # 仅一级标题
    STANDARD = 2  # 二级标题
    DEEP = 3      # 三级标题



# Python 3.11+ 可直接用 StrEnum，这里兼容低版本
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        """兼容 Py<3.11 的 StrEnum"""
        pass



class Domain(StrEnum):
    """大纲领域"""
    COURSE = "course"
    ARTICLE = "article"
    SPEECH = "speech"
    GENERAL = "general"



class NumberingStyle(StrEnum):
    """编号风格"""
    ARABIC_DOT = "arabic_dot"  # 1. 2. 3.
    CHINESE = "chinese"        # 一、二、三（仅一级标题）
    ROMAN = "roman"            # I. II. III.（仅一级标题）



class OutputLanguage(StrEnum):
    """输出语言"""
    ZH = "zh"
    EN = "en"
    AUTO = "auto"



# ============================================================================
# Module Constants
# ============================================================================
_TOPIC_PLACEHOLDER = "{{TOPIC}}"          # 不会出现在自然语言中，安全
_MAX_EXTRA_INSTRUCTIONS_LENGTH = 500
_SKILL_VERSION = "2.0.0"

# 中文数字映射
_CHINESE_NUMERALS = [
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "十一", "十二", "十三", "十四", "十五",
]
# ============================================================================
# Optional Dependency: roman (罗马数字转换)
# ============================================================================
try:
    import roman
    _ROMAN_AVAILABLE = True
except ImportError:
    _ROMAN_AVAILABLE = False
    logger.info("roman 库未安装，罗马数字编号将回退到阿拉伯数字。pip install roman")

# ============================================================================
# Jinja2 Markdown Template (用于 Jinja2 可用时)
# ============================================================================
_MD_JINJA_TEMPLATE = """\
<!-- Generated by OutlineGeneratorSkill v{{ version }} | domain={{ domain }} | depth={{ depth }} | language={{ language }} | numbering={{ numbering }} -->
# {{ title }}

{% for section in sections %}
## {{ section['number'] }} {{ section['title'] }}
{% if depth >= 2 -%}
{% for item in section['items'] %}
  - {{ item['text'] }}
{% if depth >= 3 -%}
{% for detail in item['details'] %}
    - {{ detail }}
{% endfor -%}
{% endif -%}
{% endfor -%}
{% endif %}

{% endfor -%}
{% if extra -%}
---
### {{ extra_label }}
> {{ extra }}
{% endif -%}
"""



# ============================================================================
# Default YAML Templates 它定义了：“当用户选择不同领域（课程 / 文章 / 演讲）时，大纲具体应该长什么样？”
# ============================================================================
_DEFAULT_TEMPLATES_YAML = r"""
zh:
  labels:
    course: "课程大纲"
    article: "文章大纲"
    speech: "演讲提纲"
    general: "大纲"
  extra_label: "📌 定制要求"
  sections:
    course:
      - title: "课程概述与学习目标"
        items:
          - text: "课程背景与意义"
            details:
              - "行业现状与痛点分析"
              - "本课程在知识体系中的定位"
          - text: "学前准备与先修知识"
            details:
              - "必需的编程基础（Python 中级）"
              - "数学与理论基础要求（线性代数、概率论）"
          - text: "本课程的学习路径图"
            details:
              - "各模块递进逻辑说明"
              - "预估学时分配与学习节奏建议"
      - title: "核心概念与理论基础"
        items:
          - text: "核心术语定义"
            details:
              - "术语表与范畴界定"
              - "与相关概念的辨析"
          - text: "关键原理讲解"
            details:
              - "原理推导过程与直觉解释"
              - "可视化 / 动画辅助理解"
          - text: "常见误区辨析"
            details:
              - "典型错误案例拆解"
              - "正确的理解方式与记忆技巧"
      - title: "实践操作与案例分析"
        items:
          - text: "动手实验 / Demo"
            details:
              - "环境搭建与工具准备"
              - "手把手操作步骤"
          - text: "真实案例分析"
            details:
              - "案例背景与问题描述"
              - "解决思路与关键决策点"
          - text: "常见问题排查"
            details:
              - "高频报错与解决方案"
              - "调试技巧与最佳实践"
      - title: "进阶与总结"
        items:
          - text: "进阶技巧与优化"
            details:
              - "性能优化策略"
              - "高级特性与扩展方向"
          - text: "课程总结与回顾"
            details:
              - "核心知识点思维导图"
              - "各模块关联关系梳理"
          - text: "课后练习与延伸阅读"
            details:
              - "分级练习题（基础 / 进阶 / 挑战）"
              - "推荐书籍、论文与在线资源"

    article:
      - title: "引言 / 背景"
        items:
          - text: "问题是什么"
            details:
              - "具体场景与现象描述"
              - "问题的边界定义"
          - text: "为什么重要"
            details:
              - "对行业 / 用户的影响"
              - "不解决的后果"
          - text: "本文要解决什么"
            details:
              - "核心论点一句话概述"
              - "文章结构与阅读路线图"
      - title: "主体论述"
        items:
          - text: "论点一：{{TOPIC}} 的核心机制"
            details:
              - "理论依据与推导"
              - "实证数据支撑"
          - text: "论点二：{{TOPIC}} 的应用场景"
            details:
              - "典型应用案例"
              - "效果对比分析"
          - text: "论证与证据"
            details:
              - "数据来源与可靠性"
              - "图表 / 可视化展示"
      - title: "深入讨论"
        items:
          - text: "反面观点与回应"
            details:
              - "常见质疑整理"
              - "逐条回应与澄清"
          - text: "局限性与边界"
            details:
              - "当前方案的适用范围"
              - "不适用场景说明"
          - text: "与已有工作的对比"
            details:
              - "竞品 / 替代方案对比表"
              - "本方案的独特优势"
      - title: "结论与展望"
        items:
          - text: "核心结论"
            details:
              - "三个关键要点回顾"
              - "一句话总结"
          - text: "实际意义 / 应用建议"
            details:
              - "对读者的行动建议"
              - "落地实施的第一步"
          - text: "未来方向"
            details:
              - "短期改进计划"
              - "长期研究展望"

    speech:
      - title: "开场（Hook）"
        items:
          - text: "引人入胜的开场"
            details:
              - "一个令人惊讶的数据 / 事实"
              - "一段简短的个人故事"
          - text: "自我介绍与演讲目的"
            details:
              - "我是谁 & 为什么有资格讲这个"
              - "今天听众会带走什么"
          - text: "今天的路线图"
            details:
              - "演讲结构预览（3 个要点）"
              - "时间分配说明"
      - title: "核心内容（Body）"
        items:
          - text: "要点一：核心信息 + 案例"
            details:
              - "核心信息一句话表达"
              - "真实案例 / 数据支撑"
          - text: "要点二：核心信息 + 案例"
            details:
              - "核心信息一句话表达"
              - "真实案例 / 数据支撑"
          - text: "要点三：核心信息 + 案例"
            details:
              - "核心信息一句话表达"
              - "真实案例 / 数据支撑"
      - title: "高潮与互动"
        items:
          - text: "关键数据 / 洞见冲击"
            details:
              - "一个让人记住的数字"
              - "背后的深层含义"
          - text: "与听众的互动环节"
            details:
              - "提问设计 / 现场投票"
              - "小组讨论引导"
          - text: "金句 / 记忆点"
            details:
              - "一句话可引用的核心观点"
              - "重复强化策略"
      - title: "收尾（Closing）"
        items:
          - text: "核心信息回顾"
            details:
              - "三个要点快速 Recap"
              - "用听众的话复述"
          - text: "行动号召（Call to Action）"
            details:
              - "明确的下一步指令"
              - "紧迫感营造"
          - text: "感谢与联系方式"
            details:
              - "真诚感谢"
              - "二维码 / 社交媒体 / 邮箱"

    general:
      - title: "背景与动机"
        items:
          - text: "现状描述"
            details:
              - "当前状态的事实性描述"
              - "关键利益相关方分析"
          - text: "核心问题"
            details:
              - "问题的精准定义"
              - "根因分析"
          - text: "目标与范围"
            details:
              - "期望达成的具体目标（SMART）"
              - "明确不做什么（边界）"
      - title: "方案 / 主体内容"
        items:
          - text: "方案设计"
            details:
              - "整体架构与核心思路"
              - "关键技术选型与理由"
          - text: "关键步骤"
            details:
              - "分阶段实施计划"
              - "每个阶段的交付物"
          - text: "注意事项"
            details:
              - "常见陷阱与规避方法"
              - "资源与依赖项检查清单"
      - title: "预期效果与验证"
        items:
          - text: "预期成果"
            details:
              - "定量指标（KPI）"
              - "定性收益"
          - text: "衡量标准"
            details:
              - "数据采集方法"
              - "成功 / 失败的判定阈值"
          - text: "风险与对策"
            details:
              - "主要风险清单（概率 × 影响）"
              - "应急预案与缓解措施"
      - title: "总结与下一步"
        items:
          - text: "核心要点回顾"
            details:
              - "3-5 个关键结论"
              - "一句话总结"
          - text: "行动计划"
            details:
              - "负责人与时间节点"
              - "第一周具体要做的事"
          - text: "资源与参考"
            details:
              - "推荐阅读 / 工具 / 模板"
              - "联系人与支持渠道"

en:
  labels:
    course: "Course Outline"
    article: "Article Outline"
    speech: "Speech Outline"
    general: "Outline"
  extra_label: "📌 Custom Requirements"
  sections:
    course:
      - title: "Course Overview & Learning Objectives"
        items:
          - text: "Course Background & Significance"
            details:
              - "Industry landscape and pain points"
              - "Positioning within the knowledge ecosystem"
          - text: "Prerequisites & Preparatory Knowledge"
            details:
              - "Required programming foundation (Python intermediate)"
              - "Mathematics & theory requirements (linear algebra, probability)"
          - text: "Learning Roadmap"
            details:
              - "Module progression logic"
              - "Estimated time allocation & pacing advice"
      - title: "Core Concepts & Theoretical Foundation"
        items:
          - text: "Key Terminology & Definitions"
            details:
              - "Glossary and scope definition"
              - "Distinction from related concepts"
          - text: "Core Principles Explained"
            details:
              - "Derivation process & intuitive explanation"
              - "Visualization / animation for comprehension"
          - text: "Common Misconceptions"
            details:
              - "Typical error case breakdown"
              - "Correct understanding & memory techniques"
      - title: "Hands-on Practice & Case Analysis"
        items:
          - text: "Lab / Demo"
            details:
              - "Environment setup & tool preparation"
              - "Step-by-step walkthrough"
          - text: "Real-world Case Studies"
            details:
              - "Case background & problem description"
              - "Solution approach & key decision points"
          - text: "Troubleshooting Common Issues"
            details:
              - "Frequent errors & solutions"
              - "Debugging tips & best practices"
      - title: "Advanced Topics & Wrap-up"
        items:
          - text: "Advanced Techniques & Optimization"
            details:
              - "Performance optimization strategies"
              - "Advanced features & extension directions"
          - text: "Course Summary & Review"
            details:
              - "Core knowledge mind map"
              - "Cross-module relationship overview"
          - text: "Exercises & Further Reading"
            details:
              - "Graded exercises (basic / intermediate / challenge)"
              - "Recommended books, papers & online resources"

    article:
      - title: "Introduction / Background"
        items:
          - text: "What Is the Problem?"
            details:
              - "Specific scenario & phenomenon description"
              - "Problem boundary definition"
          - text: "Why Does It Matter?"
            details:
              - "Impact on industry / users"
              - "Consequences of inaction"
          - text: "What This Article Addresses"
            details:
              - "Core thesis in one sentence"
              - "Article structure & reading roadmap"
      - title: "Main Discussion"
        items:
          - text: "Argument 1: Core Mechanism of {{TOPIC}}"
            details:
              - "Theoretical basis & derivation"
              - "Empirical data support"
          - text: "Argument 2: Application Scenarios of {{TOPIC}}"
            details:
              - "Typical use cases"
              - "Comparative effectiveness analysis"
          - text: "Evidence & Support"
            details:
              - "Data sources & reliability"
              - "Charts / visualizations"
      - title: "In-depth Discussion"
        items:
          - text: "Counterarguments & Responses"
            details:
              - "Common objections collected"
              - "Point-by-point rebuttal & clarification"
          - text: "Limitations & Boundaries"
            details:
              - "Applicable scope of current approach"
              - "Scenarios where it does NOT apply"
          - text: "Comparison with Existing Work"
            details:
              - "Competitor / alternative comparison table"
              - "Unique advantages of this approach"
      - title: "Conclusion & Outlook"
        items:
          - text: "Core Conclusions"
            details:
              - "Three key takeaways recap"
              - "One-sentence summary"
          - text: "Practical Implications & Recommendations"
            details:
              - "Actionable advice for readers"
              - "First step for implementation"
          - text: "Future Directions"
            details:
              - "Short-term improvement plan"
              - "Long-term research outlook"

    speech:
      - title: "Opening (Hook)"
        items:
          - text: "Attention-grabbing Opening"
            details:
              - "A surprising statistic / fact"
              - "A short personal story"
          - text: "Self-introduction & Purpose"
            details:
              - "Who I am & why I'm qualified"
              - "What the audience will take away"
          - text: "Today's Roadmap"
            details:
              - "Talk structure preview (3 key points)"
              - "Time allocation overview"
      - title: "Core Content (Body)"
        items:
          - text: "Key Point 1: Core Message + Case Study"
            details:
              - "One-sentence core message"
              - "Real case / data support"
          - text: "Key Point 2: Core Message + Case Study"
            details:
              - "One-sentence core message"
              - "Real case / data support"
          - text: "Key Point 3: Core Message + Case Study"
            details:
              - "One-sentence core message"
              - "Real case / data support"
      - title: "Climax & Interaction"
        items:
          - text: "Impactful Data / Insight"
            details:
              - "A memorable number"
              - "The deeper meaning behind it"
          - text: "Audience Interaction"
            details:
              - "Question design / live polling"
              - "Group discussion facilitation"
          - text: "Memorable Quote / Takeaway"
            details:
              - "One quotable core insight"
              - "Repetition reinforcement strategy"
      - title: "Closing"
        items:
          - text: "Key Message Recap"
            details:
              - "Three points quick recap"
              - "Restate in audience's words"
          - text: "Call to Action"
            details:
              - "Clear next step instruction"
              - "Sense of urgency"
          - text: "Thank You & Contact"
            details:
              - "Sincere gratitude"
              - "QR code / social media / email"

    general:
      - title: "Background & Motivation"
        items:
          - text: "Current State"
            details:
              - "Factual description of current situation"
              - "Key stakeholder analysis"
          - text: "Core Problem"
            details:
              - "Precise problem definition"
              - "Root cause analysis"
          - text: "Goals & Scope"
            details:
              - "Specific objectives (SMART)"
              - "Explicit boundaries (what's NOT included)"
      - title: "Solution / Main Content"
        items:
          - text: "Solution Design"
            details:
              - "Overall architecture & core approach"
              - "Key technology choices & rationale"
          - text: "Key Steps"
            details:
              - "Phased implementation plan"
              - "Deliverables per phase"
          - text: "Important Considerations"
            details:
              - "Common pitfalls & avoidance strategies"
              - "Resource & dependency checklist"
      - title: "Expected Outcomes & Validation"
        items:
          - text: "Expected Results"
            details:
              - "Quantitative metrics (KPIs)"
              - "Qualitative benefits"
          - text: "Measurement Criteria"
            details:
              - "Data collection methods"
              - "Success / failure thresholds"
          - text: "Risks & Mitigations"
            details:
              - "Major risk register (probability × impact)"
              - "Contingency plans & mitigation measures"
      - title: "Summary & Next Steps"
        items:
          - text: "Key Takeaways Review"
            details:
              - "3-5 key conclusions"
              - "One-sentence summary"
          - text: "Action Plan"
            details:
              - "Owners & deadlines"
              - "Concrete first-week tasks"
          - text: "Resources & References"
            details:
              - "Recommended reading / tools / templates"
              - "Contacts & support channels"
"""


# ============================================================================
# Pydantic Schemas
# ============================================================================
class OutlineGeneratorInput(BaseModel):
    """大纲生成输入参数"""
    topic: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="主题或关键词",
        examples=["Python 机器学习", "碳中和政策分析"],
    )
    depth: DepthLevel = Field(
        default=DepthLevel.STANDARD,
        description="大纲深度：1=仅一级标题, 2=二级标题, 3=三级标题（含具体说明）",
    )
    domain: Domain = Field(
        default=Domain.GENERAL,
        description="领域：course=课程大纲, article=文章大纲, speech=演讲提纲, general=通用",
    )
    numbering_style: NumberingStyle = Field(
        default=NumberingStyle.ARABIC_DOT,
        description="编号风格：arabic_dot=1., chinese=一、, roman=I.",
    )
    output_language: OutputLanguage = Field(
        default=OutputLanguage.AUTO,
        description="输出语言：zh=中文, en=英文, auto=根据主题自动检测",
    )
    extra_instructions: str = Field(
        default="",
        max_length=_MAX_EXTRA_INSTRUCTIONS_LENGTH,
        description=f"额外的定制要求（最多 {_MAX_EXTRA_INSTRUCTIONS_LENGTH} 字符）",
    )

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_only_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("topic 不能为空白字符串")
        return stripped


class OutlineGeneratorOutput(BaseModel):
    """大纲生成输出"""
    title: str = Field(..., description="大纲标题")
    outline_markdown: str = Field(..., description="Markdown 格式的结构化大纲")
    section_count: int = Field(..., description="一级章节数量")
    second_level_count: int = Field(..., description="二级条目数量")
    third_level_count: int = Field(..., description="三级条目数量")
    total_items: int = Field(..., description="总大纲条目数（一级+二级+三级）")
    topic: str = Field(default="", description="生成大纲的主题（输入回传，便于追溯）")
    domain: str = Field(default="general", description="大纲领域")
    depth: int = Field(default=2, description="大纲深度级别")


# ============================================================================
# Main Skill Class
# ============================================================================
class OutlineGeneratorSkill(BaseSkill):
    """
    大纲生成技能（v2.0.0）

    根据主题或关键词生成结构化 Markdown 大纲。
    支持 1-3 级深度，覆盖课程/文章/演讲/通用四种场景，
    支持中英文输出、多种编号风格。
    """

    name: str = "outline_generator"
    description: str = (
        "根据主题或关键词生成结构化 Markdown 大纲。支持 1-3 级深度，"
        "覆盖课程大纲、文章大纲、演讲提纲等多种场景。"
    )
    triggers: list[str] = [
        "大纲", "outline", "生成大纲", "列提纲", "目录",
        "结构", "框架", "帮我列", "帮我规划", "梳理结构",
        "课程大纲", "文章结构", "演讲提纲",
    ]
    version: str = _SKILL_VERSION
    author: str = "EnterpriseLearningAgent"
    changelog: str = (
        "v2.0.0: 全面优化 — Jinja2 模板引擎、YAML 外部化模板、"
        "三级大纲真实内容、中英文双模板、编号风格、分层统计、"
        "线程安全深拷贝、Enum 替代 Literal、元注释追溯"
    )
    input_schema = OutlineGeneratorInput
    output_schema = OutlineGeneratorOutput

    # ========================================================================
    # Instance attributes
    # ========================================================================
    _templates: dict[str, Any]           # 深拷贝后的实例级模板
    _jinja_template: Any | None          # 编译后的 Jinja2 模板对象

    # ========================================================================
    # 输入：templates_path（可选外部 YAML 路径）。
    # 做了什么：
    # 调用 _load_templates 加载配置（外部文件优先，否则用内置字符串）。
    # 调用 _compile_jinja_template 预编译模板（如果 Jinja2 可用）。
    # 输出：初始化好的 Skill 实例。
    # ========================================================================
    def __init__(self, templates_path: Optional[str] = None):
        """
        Args:
            templates_path: 外部 YAML 模板文件路径。
                           为 None 时使用内置默认模板。
        """
        super().__init__()
        # P1-5: 从 YAML 加载（外部文件或内置默认）
        self._templates = self._load_templates(templates_path)
        # P1-6: 编译 Jinja2 模板（可用时）
        self._jinja_template = self._compile_jinja_template()
        logger.debug("OutlineGeneratorSkill 初始化完成 | domains=%s | jinja2=%s",
                     list(self._templates.get("zh", {}).get("sections", {}).keys()),
                     _JINJA2_AVAILABLE)

    #功能：生成调试友好的字符串表示。
    def __repr__(self) -> str:
        """P3-15: 调试友好"""
        domains = list(self._templates.get("zh", {}).get("sections", {}).keys())
        return (f"OutlineGeneratorSkill(version={self.version!r}, "
                f"domains={domains}, jinja2={_JINJA2_AVAILABLE})")

    # ========================================================================
    # 输入：templates_path（可选路径）。
    # 功能：
    # 优先尝试从外部文件加载 YAML。
    # 若路径无效或未提供，加载内置的 _DEFAULT_TEMPLATES_YAML 字符串。
    # 对加载结果进行 deepcopy，确保实例隔离、线程安全。
    # 输出：解析后的模板字典（包含 zh/en 双语配置）。
    # ========================================================================
    def _load_templates(self, templates_path: Optional[str]) -> dict[str, Any]:
        """P1-5: 加载模板配置（外部文件优先，回退到内置默认）"""
        if templates_path and Path(templates_path).exists():
            logger.info("从外部文件加载模板: %s", templates_path)
            with open(templates_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        else:
            if templates_path:
                logger.warning("模板文件不存在，使用内置默认: %s", templates_path)
            raw = yaml.safe_load(_DEFAULT_TEMPLATES_YAML)

        # P1-4: 深拷贝确保实例隔离、线程安全
        return copy.deepcopy(raw)


    # 功能：检查 Jinja2 是否可用，若可用则编译 _MD_JINJA_TEMPLATE 字符串。
    # 输出：编译后的 Jinja2 Template 对象，或 None。
    def _compile_jinja_template(self) -> Any:
        """P1-6: 编译 Jinja2 模板（不可用时返回 None）"""
        if _JINJA2_AVAILABLE:
            from jinja2 import Template
            return Template(_MD_JINJA_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
        return None

    # ========================================================================
    # Main Execute (核心主流程)
    # 输入：OutlineGeneratorInput 对象。
    # 功能（流水线逻辑）：
    # 参数清洗：提取并转换参数（Enum 转值、字符串去空格）。
    # 语言决策：调用 _resolve_language 确定用中文还是英文。
    # 数据准备：从 _templates 中取出对应语言和领域的原始结构。
    # 结构构建：调用 _build_sections 生成带编号、替换过占位符的数据。
    # 渲染输出：调用 _render_markdown 生成最终 Markdown 字符串。
    # 统计封装：计算总数，组装并返回 Output 对象。
    # 输出：OutlineGeneratorOutput 对象。
    # ========================================================================
    def execute(self, input_data: OutlineGeneratorInput) -> OutlineGeneratorOutput:
        topic = input_data.topic.strip()
        domain = input_data.domain.value if isinstance(input_data.domain, Domain) else input_data.domain
        depth = int(input_data.depth)
        numbering = input_data.numbering_style.value if isinstance(input_data.numbering_style, NumberingStyle) else input_data.numbering_style
        lang = self._resolve_language(input_data.output_language, topic)
        extra = input_data.extra_instructions.strip()

        logger.info(
            "[outline_generator] topic=%r | domain=%s | depth=%d | numbering=%s | lang=%s",
            topic, domain, depth, numbering, lang,
        )

        # 加载语言配置
        lang_cfg = self._templates.get(lang, self._templates.get("zh", {}))
        labels = lang_cfg.get("labels", {})
        sections_raw = lang_cfg.get("sections", {}).get(domain, lang_cfg.get("sections", {}).get("general", []))

        # 构建带编号和数据的大纲结构
        built_sections, section_count, second_level_count, third_level_count = \
            self._build_sections(sections_raw, topic, depth, numbering)

        # 标题
        outline_type_label = labels.get(domain, labels.get("general", "大纲"))
        title = f"{topic} {outline_type_label}"

        # 渲染 Markdown
        markdown = self._render_markdown(
            title=title,
            sections=built_sections,
            depth=depth,
            domain=domain,
            lang=lang,
            numbering=numbering,
            extra=extra,
            extra_label=lang_cfg.get("extra_label", "📌 定制要求"),
        )

        total_items = section_count + second_level_count + third_level_count

        logger.info(
            "[outline_generator] 生成完成 | 一级=%d | 二级=%d | 三级=%d | 总计=%d",
            section_count, second_level_count, third_level_count, total_items,
        )

        return OutlineGeneratorOutput(
            title=title,
            outline_markdown=markdown,
            section_count=section_count,
            second_level_count=second_level_count,
            third_level_count=third_level_count,
            total_items=total_items,
            topic=topic,
            domain=domain,
            depth=depth,
        )

    # ========================================================================
    # Language Detection
    # 输入：output_language（指定语言）、topic（主题）。
    # 功能：
    # 若指定了非 auto，直接返回。
    # 策略：先用规则（统计中日韩文字符占比 > 20% 则为中文），更准；规则不确定时用 langdetect 库兜底。
    # 输出："zh" 或 "en"。
    # ========================================================================
    @staticmethod
    def _resolve_language(output_language: OutputLanguage | str, topic: str) -> str:
        """P2-7: 解析输出语言，auto 时基于 CJK 字符统计（优先）或 langdetect（辅助）"""
        if isinstance(output_language, OutputLanguage):
            lang_val = output_language.value
        else:
            lang_val = output_language
        if lang_val != "auto":
            return lang_val

        # 优先使用 CJK 字符统计（更可靠）
        cjk_count = sum(1 for c in topic if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
        cjk_ratio = cjk_count / max(len(topic), 1)

        # 如果中文字符比例超过 20%，直接判定为中文
        if cjk_ratio > 0.2:
            return "zh"

        # 否则使用 langdetect 辅助判断
        if _LANGDETECT_AVAILABLE:
            try:
                detected = lang_detect(topic)
                if detected.startswith("zh"):
                    return "zh"
                return "en"
            except LangDetectException:
                logger.debug("[outline_generator] langdetect 检测失败，使用默认英文")

        # 默认返回英文
        return "en"

    # ========================================================================
    # 输入：sections_raw（原始模板列表）、topic、depth、numbering。
    # 功能：
    # 遍历每一个 section，调用 _format_number 生成编号。
    # 将所有 {{TOPIC}} 占位符替换为用户输入的主题。
    # 根据 depth 决定是否包含二级 (items) 和三级 (details) 内容。
    # 边遍历边统计各层级条目数量。
    # ========================================================================
    def _build_sections(
        self,
        sections_raw: list[dict],
        topic: str,
        depth: int,
        numbering: str,
    ) -> tuple[list[dict], int, int, int]:
        """
        遍历原始模板，替换 {{TOPIC}} 占位符，应用编号，统计各层级数量。

        Returns:
            (built_sections, section_count, second_level_count, third_level_count)
        """
        built: list[dict] = []
        section_count = 0
        second_level_count = 0
        third_level_count = 0

        for i, section in enumerate(sections_raw):
            section_count += 1
            section_num = self._format_number(numbering, level=1, index=i + 1)
            section_title = section["title"].replace(_TOPIC_PLACEHOLDER, topic)

            items: list[dict] = []
            if depth >= 2:
                raw_items = section.get("items", [])
                for item in raw_items:
                    second_level_count += 1
                    item_text = item["text"].replace(_TOPIC_PLACEHOLDER, topic)

                    details: list[str] = []
                    if depth >= 3:
                        raw_details = item.get("details", [])
                        for detail in raw_details:
                            third_level_count += 1
                            details.append(detail.replace(_TOPIC_PLACEHOLDER, topic))

                    items.append({
                        "text": item_text,
                        "details": details,
                    })

            built.append({
                "number": section_num,
                "title": section_title,
                "items": items,
            })

        return built, section_count, second_level_count, third_level_count

    # ========================================================================
    # _num_to_chinese(num: int) -> str (静态方法)
    # 输入：整数 (0-99)。
    # 功能：手写算法将数字转换为中文数字（如 21 -> "二十一"）。
    # 输出：中文数字字符串。
    # ========================================================================
    @staticmethod
    def _num_to_chinese(num: int) -> str:
        """将数字转换为中文数字（支持 0-99）"""
        if num < 0:
            return str(num)
        if num < 21:
            return _CHINESE_NUMERALS[num]

        # 处理 20-99
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        tens = num // 10
        units = num % 10
        if units == 0:
            return digits[tens] + "十"
        else:
            return digits[tens] + "十" + digits[units]

    # ========================================================================
    # 输入：numbering（风格）、level（层级）、index（序号）。
    # 功能：
    # 中文：调用 _num_to_chinese 转换并加 “、”。
    # 罗马数字：优先用 roman 库，否则回退阿拉伯数字。
    # 阿拉伯数字或非一级标题：直接返回 "{index}."。
    # 输出：格式化后的编号字符串（如 "一、" 或 "II."）
    # ========================================================================
    @staticmethod
    def _format_number(numbering: str, level: int, index: int) -> str:
        """P2-9: 根据编号风格和层级生成编号字符串"""
        if numbering == "chinese" and level == 1:
            return OutlineGeneratorSkill._num_to_chinese(index) + "、"
        elif numbering == "roman" and level == 1:
            if _ROMAN_AVAILABLE:
                return roman.toRoman(index) + "."
            else:
                # 回退到阿拉伯数字
                return f"{index}."
        else:
            # arabic_dot 或非一级标题的 chinese/roman
            return f"{index}."

    # ========================================================================
    # 输入：title, sections, depth 等所有渲染所需数据。
    # 功能：策略模式。根据 Jinja2 是否可用，选择渲染方式。
    # 输出：最终的 Markdown 字符串。
    # ========================================================================
    def _render_markdown(
        self,
        title: str,
        sections: list[dict],
        depth: int,
        domain: str,
        lang: str,
        numbering: str,
        extra: str,
        extra_label: str,
    ) -> str:
        """P1-6: 使用 Jinja2 渲染，不可用时回退到纯 Python 构建"""
        if _JINJA2_AVAILABLE and self._jinja_template is not None:
            return self._render_via_jinja2(
                title=title, sections=sections, depth=depth,
                domain=domain, lang=lang, numbering=numbering,
                extra=extra, extra_label=extra_label,
            )
        else:
            return self._render_via_python(
                title=title, sections=sections, depth=depth,
                domain=domain, lang=lang, numbering=numbering,
                extra=extra, extra_label=extra_label,
            )


    def _render_via_jinja2(
        self, title, sections, depth, domain, lang, numbering, extra, extra_label,
    ) -> str:
        """Jinja2 模板渲染"""
        rendered = self._jinja_template.render(
            version=self.version,
            title=title,
            sections=sections,
            depth=depth,
            domain=domain,
            language=lang,
            numbering=numbering,
            extra=extra,
            extra_label=extra_label,
        )
        return rendered.strip() + "\n"

    def _render_via_python(
        self, title, sections, depth, domain, lang, numbering, extra, extra_label,
    ) -> str:
        """纯 Python 字符串构建（Jinja2 不可用时的回退）"""
        lines: list[str] = []

        # P2-8: 元注释
        lines.append(
            f"<!-- Generated by OutlineGeneratorSkill v{self.version} "
            f"| domain={domain} | depth={depth} | language={lang} | numbering={numbering} -->"
        )
        lines.append(f"# {title}")
        lines.append("")

        for section in sections:
            lines.append(f"## {section['number']} {section['title']}")
            if depth >= 2:
                for item in section["items"]:
                    lines.append(f"  - {item['text']}")
                    if depth >= 3:
                        for detail in item["details"]:
                            lines.append(f"    - {detail}")
            lines.append("")

        if extra:
            lines.append("---")
            lines.append(f"### {extra_label}")
            lines.append(f"> {extra}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Convenience Function
# ============================================================================
def generate_outline(
    topic: str,
    depth: DepthLevel | int = DepthLevel.STANDARD,
    domain: Domain | str = Domain.GENERAL,
    numbering_style: NumberingStyle | str = NumberingStyle.ARABIC_DOT,
    output_language: OutputLanguage | str = OutputLanguage.AUTO,
    extra_instructions: str = "",
    templates_path: Optional[str] = None,
) -> OutlineGeneratorOutput:
    """
    便捷函数：一行调用生成大纲。

    Args:
        topic: 主题
        depth: 深度 (1/2/3)
        domain: 领域 (course/article/speech/general)
        numbering_style: 编号风格 (arabic_dot/chinese/roman)
        output_language: 输出语言 (zh/en/auto)
        extra_instructions: 额外定制要求
        templates_path: 外部模板文件路径

    Returns:
        OutlineGeneratorOutput
    """
    skill = OutlineGeneratorSkill(templates_path=templates_path)
    return skill.execute(OutlineGeneratorInput(
        topic=topic,
        depth=DepthLevel(depth) if isinstance(depth, int) else depth,
        domain=Domain(domain) if isinstance(domain, str) else domain,
        numbering_style=NumberingStyle(numbering_style) if isinstance(numbering_style, str) else numbering_style,
        output_language=OutputLanguage(output_language) if isinstance(output_language, str) else output_language,
        extra_instructions=extra_instructions,
    ))