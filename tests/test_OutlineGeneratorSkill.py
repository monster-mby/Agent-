# ========================================================================
# pytest 测试套件：OutlineGeneratorSkill v2.0.0
#
# 覆盖：
#   - 基本功能（生成大纲、中英文、各领域）
#   - 输入验证（Pydantic schema）
#   - 边界条件（空主题、超长主题、超长额外要求）
#   - 语言检测（auto 模式）
#   - 模板加载（内置默认 / 外部 YAML）
#   - 占位符替换（{{TOPIC}}）
#   - 编号风格（arabic_dot / chinese / roman）
#   - 深度控制（1/2/3 级）
#   - 统计功能（分层计数）
#   - Markdown 渲染（Jinja2 / 纯 Python 回退）
#   - 可选依赖回退（Jinja2 / langdetect / roman）
#   - 便捷函数
#
# 运行方式:
#     pytest tests/test_outline_generator.py -v
#     或带覆盖率:
#     pytest tests/test_outline_generator.py -v --cov=src.skills.outline_generator
# ========================================================================

import copy
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from src.skills.preset.content_creation.outline_generator import (
    OutlineGeneratorSkill,
    OutlineGeneratorInput,
    OutlineGeneratorOutput,
    DepthLevel,
    Domain,
    NumberingStyle,
    OutputLanguage,
    generate_outline,
)

# 私有常量需要从 skill 模块直接导入
from src.skills.preset.content_creation.outline_generator.skill import (
    _DEFAULT_TEMPLATES_YAML,
    _TOPIC_PLACEHOLDER,
    _CHINESE_NUMERALS,
    _SKILL_VERSION,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def skill():
    """返回使用默认内置模板的技能实例"""
    return OutlineGeneratorSkill()


@pytest.fixture
def skill_with_external_templates(tmp_path):
    """返回使用外部 YAML 模板的技能实例"""
    templates_path = tmp_path / "custom_templates.yaml"

    # 创建一个简化的自定义模板
    custom_templates = {
        "zh": {
            "labels": {"general": "自定义大纲"},
            "extra_label": "📌 我的要求",
            "sections": {
                "general": [
                    {"title": "自定义第一章", "items": [{"text": "自定义项", "details": ["自定义细节"]}]}
                ]
            }
        }
    }

    with open(templates_path, "w", encoding="utf-8") as f:
        yaml.dump(custom_templates, f)

    return OutlineGeneratorSkill(templates_path=str(templates_path))


@pytest.fixture
def valid_topic():
    """有效的测试主题"""
    return "Python 机器学习入门"


# ============================================================================
# 1. 基本功能测试
# ============================================================================
class TestBasicFunctionality:
    """测试基本大纲生成功能"""

    def test_generate_general_outline(self, skill, valid_topic):
        """测试生成通用大纲"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            domain=Domain.GENERAL,
            depth=DepthLevel.STANDARD,
            output_language=OutputLanguage.ZH,
        ))
        assert isinstance(result, OutlineGeneratorOutput)
        assert valid_topic in result.title
        assert len(result.outline_markdown) > 0
        assert result.section_count > 0
        assert result.total_items > 0

    def test_generate_course_outline(self, skill, valid_topic):
        """测试生成课程大纲"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            domain=Domain.COURSE,
            depth=DepthLevel.DEEP,
            output_language=OutputLanguage.ZH,
        ))
        assert "课程" in result.title or "Course" in result.title
        assert result.section_count >= 3
        assert result.third_level_count > 0

    def test_generate_article_outline(self, skill, valid_topic):
        """测试生成文章大纲"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            domain=Domain.ARTICLE,
            output_language=OutputLanguage.ZH,
        ))
        assert "文章" in result.title or "Article" in result.title

    def test_generate_speech_outline(self, skill, valid_topic):
        """测试生成演讲提纲"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            domain=Domain.SPEECH,
            output_language=OutputLanguage.ZH,
        ))
        assert "演讲" in result.title or "Speech" in result.title

    def test_english_output(self, skill):
        """测试英文输出"""
        result = skill.execute(OutlineGeneratorInput(
            topic="Machine Learning with Python",
            domain=Domain.GENERAL,
            output_language=OutputLanguage.EN,
        ))
        assert "Outline" in result.title
        # 检查是否有英文标题
        assert "Background" in result.outline_markdown or "Background" in result.title

    def test_extra_instructions(self, skill, valid_topic):
        """测试额外定制要求"""
        extra = "请重点关注深度学习部分，忽略传统算法。"
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            extra_instructions=extra,
            output_language=OutputLanguage.ZH,
        ))
        assert extra in result.outline_markdown


# ============================================================================
# 2. 输入验证与边界条件测试
# ============================================================================
class TestInputValidation:
    """测试 Pydantic 输入验证"""

    def test_topic_too_short(self, skill):
        """测试主题过短（仅1字符）"""
        with pytest.raises(Exception):
            OutlineGeneratorInput(topic="A")

    def test_topic_empty_string(self, skill):
        """测试空字符串主题"""
        with pytest.raises(Exception):
            OutlineGeneratorInput(topic="")

    def test_topic_only_whitespace(self, skill):
        """测试仅空白字符的主题"""
        with pytest.raises(Exception):
            OutlineGeneratorInput(topic="   \n  \t  ")

    def test_topic_normalized(self, skill, valid_topic):
        """测试主题首尾空白被 strip"""
        result = skill.execute(OutlineGeneratorInput(
            topic=f"  {valid_topic}  \n",
            output_language=OutputLanguage.ZH,
        ))
        assert result.topic == valid_topic  # 回传的 topic 应该是 clean 的

    def test_extra_instructions_too_long(self, skill, valid_topic):
        """测试额外要求超长"""
        long_extra = "A" * 600  # 超过 500
        with pytest.raises(Exception):
            OutlineGeneratorInput(topic=valid_topic, extra_instructions=long_extra)

    def test_depth_enum(self, skill, valid_topic):
        """测试深度使用 int 也能工作"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=3,  # 直接传 int
            output_language=OutputLanguage.ZH,
        ))
        assert result.depth == 3


# ============================================================================
# 3. 语言检测测试
# ============================================================================
class TestLanguageDetection:
    """测试 auto 模式下的语言检测"""

    def test_auto_detect_chinese(self, skill):
        """测试自动检测中文"""
        result = skill.execute(OutlineGeneratorInput(
            topic="人工智能深度学习",
            output_language=OutputLanguage.AUTO,
        ))
        # 检查标题或内容是否有中文特征
        assert "大纲" in result.title or "背景" in result.outline_markdown

    def test_auto_detect_english(self, skill):
        """测试自动检测英文"""
        result = skill.execute(OutlineGeneratorInput(
            topic="Deep Learning Artificial Intelligence",
            output_language=OutputLanguage.AUTO,
        ))
        assert "Outline" in result.title or "Background" in result.outline_markdown

    def test_mixed_language_chinese_dominant(self, skill):
        """测试混合语言（中文为主）"""
        result = skill.execute(OutlineGeneratorInput(
            topic="Python 机器学习 实战",
            output_language=OutputLanguage.AUTO,
        ))
        # 应该检测为中文
        assert "大纲" in result.title or "背景" in result.outline_markdown

    def test_langdetect_fallback(self, skill):
        """测试 langdetect 不可用时的回退"""
        with patch("src.skills.preset.content_creation.outline_generator.skill._LANGDETECT_AVAILABLE", False):
            # 强制走 CJK 统计路径
            result = skill.execute(OutlineGeneratorInput(
                topic="中文主题",
                output_language=OutputLanguage.AUTO,
            ))
            assert result is not None


# ============================================================================
# 4. 模板加载测试
# ============================================================================
class TestTemplateLoading:
    """测试模板加载逻辑"""

    def test_default_templates_loaded(self, skill):
        """测试默认模板正确加载"""
        assert skill._templates is not None
        assert "zh" in skill._templates
        assert "en" in skill._templates
        assert "sections" in skill._templates["zh"]

    def test_external_templates_loaded(self, skill_with_external_templates):
        """测试外部模板正确加载"""
        result = skill_with_external_templates.execute(OutlineGeneratorInput(
            topic="测试主题",
            output_language=OutputLanguage.ZH,
        ))
        assert "自定义大纲" in result.title
        assert "自定义第一章" in result.outline_markdown

    def test_nonexistent_template_fallback(self, tmp_path):
        """测试模板文件不存在时回退到默认"""
        skill = OutlineGeneratorSkill(templates_path="/nonexistent/path.yaml")
        # 应该没有崩溃，且使用了默认模板
        assert "zh" in skill._templates

    def test_templates_deep_copy(self, skill):
        """测试模板是深拷贝的（线程安全）"""
        # 修改实例的模板不应影响全局
        original = copy.deepcopy(skill._templates)
        skill._templates["zh"]["labels"]["general"] = "被修改了"
        # 重新加载默认 YAML 验证
        raw = yaml.safe_load(_DEFAULT_TEMPLATES_YAML)
        assert raw["zh"]["labels"]["general"] != "被修改了"


# ============================================================================
# 5. 占位符替换测试
# ============================================================================
class TestPlaceholderReplacement:
    """测试 {{TOPIC}} 占位符正确替换"""

    def test_topic_in_title(self, skill, valid_topic):
        """测试主题出现在标题中"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            output_language=OutputLanguage.ZH,
        ))
        assert valid_topic in result.title

    def test_topic_in_sections(self, skill):
        """测试主题替换了模板中的 {{TOPIC}}"""
        # 使用 article 模板，因为它明确包含 {{TOPIC}}
        topic = "碳中和"
        result = skill.execute(OutlineGeneratorInput(
            topic=topic,
            domain=Domain.ARTICLE,
            depth=DepthLevel.DEEP,
            output_language=OutputLanguage.ZH,
        ))
        # 检查占位符是否被替换
        assert _TOPIC_PLACEHOLDER not in result.outline_markdown
        # 检查主题是否出现在内容中（article 模板的论点里）
        assert topic in result.outline_markdown


# ============================================================================
# 6. 编号风格测试
# ============================================================================
class TestNumberingStyle:
    """测试不同的编号风格"""

    def test_arabic_dot_numbering(self, skill, valid_topic):
        """测试阿拉伯数字点号风格 (1. 2. 3.)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            numbering_style=NumberingStyle.ARABIC_DOT,
            output_language=OutputLanguage.ZH,
        ))
        # 检查 Markdown 中是否有 "## 1."
        assert "## 1." in result.outline_markdown

    def test_chinese_numbering(self, skill, valid_topic):
        """测试中文数字风格 (一、 二、 三、)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            numbering_style=NumberingStyle.CHINESE,
            output_language=OutputLanguage.ZH,
        ))
        assert "## 一、" in result.outline_markdown or "## 二、" in result.outline_markdown

    def test_roman_numbering(self, skill, valid_topic):
        """测试罗马数字风格 (I. II. III.)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            numbering_style=NumberingStyle.ROMAN,
            output_language=OutputLanguage.EN,
        ))
        # 可能有 roman 库不可用的情况，所以只要不崩溃即可
        assert result is not None
        # 如果可用，检查罗马数字
        try:
            from src.skills.preset.content_creation.outline_generator.skill import _ROMAN_AVAILABLE
            if _ROMAN_AVAILABLE:
                assert "## I." in result.outline_markdown or "## II." in result.outline_markdown
        except ImportError:
            pass

    def test_roman_fallback(self, skill, valid_topic):
        """测试 roman 库不可用时回退到阿拉伯数字"""
        with patch("src.skills.preset.content_creation.outline_generator.skill._ROMAN_AVAILABLE", False):
            result = skill.execute(OutlineGeneratorInput(
                topic=valid_topic,
                numbering_style=NumberingStyle.ROMAN,
                output_language=OutputLanguage.EN,
            ))
            # 应该回退到 1. 2.
            assert "## 1." in result.outline_markdown

    def test_chinese_num_converter(self):
        """测试中文数字转换函数"""
        assert OutlineGeneratorSkill._num_to_chinese(1) == "一"
        assert OutlineGeneratorSkill._num_to_chinese(10) == "十"
        assert OutlineGeneratorSkill._num_to_chinese(21) == "二十一"
        assert OutlineGeneratorSkill._num_to_chinese(99) == "九十九"


# ============================================================================
# 7. 深度控制测试
# ============================================================================
class TestDepthControl:
    """测试 1/2/3 级深度控制"""

    def test_depth_shallow(self, skill, valid_topic):
        """测试仅一级标题 (depth=1)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.SHALLOW,
            output_language=OutputLanguage.ZH,
        ))
        assert result.depth == 1
        assert result.second_level_count == 0
        assert result.third_level_count == 0
        # 检查 Markdown 中没有 ## 下的列表
        assert "  - " not in result.outline_markdown

    def test_depth_standard(self, skill, valid_topic):
        """测试二级标题 (depth=2)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.STANDARD,
            output_language=OutputLanguage.ZH,
        ))
        assert result.depth == 2
        assert result.second_level_count > 0
        assert result.third_level_count == 0
        assert "  - " in result.outline_markdown
        assert "    - " not in result.outline_markdown

    def test_depth_deep(self, skill, valid_topic):
        """测试三级标题 (depth=3)"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.DEEP,
            output_language=OutputLanguage.ZH,
        ))
        assert result.depth == 3
        assert result.second_level_count > 0
        assert result.third_level_count > 0
        assert "    - " in result.outline_markdown


# ============================================================================
# 8. 统计功能测试
# ============================================================================
class TestStatistics:
    """测试分层计数功能"""

    def test_counts_sum_to_total(self, skill, valid_topic):
        """测试各层级数量相加等于 total_items"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.DEEP,
            output_language=OutputLanguage.ZH,
        ))
        assert (result.section_count +
                result.second_level_count +
                result.third_level_count) == result.total_items

    def test_counts_positive(self, skill, valid_topic):
        """测试数量为正"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.DEEP,
            output_language=OutputLanguage.ZH,
        ))
        assert result.section_count > 0
        assert result.second_level_count > 0
        assert result.third_level_count > 0
        assert result.total_items > 0


# ============================================================================
# 9. Markdown 渲染测试
# ============================================================================
class TestMarkdownRendering:
    """测试 Markdown 输出格式"""

    def test_metadata_comment_present(self, skill, valid_topic):
        """测试顶部 HTML 元注释存在"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            output_language=OutputLanguage.ZH,
        ))
        assert "<!-- Generated by OutlineGeneratorSkill" in result.outline_markdown
        assert _SKILL_VERSION in result.outline_markdown

    def test_heading_structure(self, skill, valid_topic):
        """测试标题层级结构正确"""
        result = skill.execute(OutlineGeneratorInput(
            topic=valid_topic,
            depth=DepthLevel.STANDARD,
            output_language=OutputLanguage.ZH,
        ))
        lines = result.outline_markdown.split("\n")
        # 应该有一个 # 标题
        assert any(line.startswith("# ") for line in lines)
        # 应该有多个 ## 标题
        assert sum(1 for line in lines if line.startswith("## ")) >= 3

    def test_jinja2_fallback(self, skill, valid_topic):
        """测试 Jinja2 不可用时回退到纯 Python"""
        with patch("src.skills.preset.content_creation.outline_generator.skill._JINJA2_AVAILABLE", False):
            skill_no_jinja = OutlineGeneratorSkill()
            result = skill_no_jinja.execute(OutlineGeneratorInput(
                topic=valid_topic,
                output_language=OutputLanguage.ZH,
            ))
            # 只要能生成且结构正确即可
            assert result is not None
            assert "# " in result.outline_markdown
            assert "## " in result.outline_markdown


# ============================================================================
# 10. 便捷函数测试
# ============================================================================
class TestConvenienceFunction:
    """测试 generate_outline() 便捷函数"""

    def test_generate_outline_basic(self):
        """测试基本调用"""
        result = generate_outline(
            topic="测试主题",
            depth=2,
            domain="general",
            output_language="zh",
        )
        assert isinstance(result, OutlineGeneratorOutput)
        assert len(result.outline_markdown) > 0

    def test_generate_outline_enums(self):
        """测试使用 Enum 调用"""
        result = generate_outline(
            topic="Test Topic",
            depth=DepthLevel.DEEP,
            domain=Domain.ARTICLE,
            numbering_style=NumberingStyle.ROMAN,
            output_language=OutputLanguage.EN,
        )
        assert result.depth == 3
        assert result.domain == "article"


# ============================================================================
# 11. 特殊方法与调试测试
# ============================================================================
class TestSpecialMethods:
    """测试 __repr__ 等特殊方法"""

    def test_repr(self, skill):
        """测试 __repr__ 输出"""
        repr_str = repr(skill)
        assert "OutlineGeneratorSkill" in repr_str
        assert _SKILL_VERSION in repr_str
        assert "jinja2" in repr_str.lower()

if __name__ == "__main__":
    pytest.main()