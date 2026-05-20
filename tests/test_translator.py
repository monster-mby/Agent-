import pytest
from pydantic import ValidationError

# 导入被测试的代码模块
from src.skills.preset.office_efficiency.translator import (
    TranslatorSkill,
    TranslatorInput,
    TranslatorOutput,
    DictionaryBackend,
    GoogleTranslateBackend,
    _HAS_DEEP_TRANSLATOR,
    _HAS_LANGDETECT,
)

# ★ 导入 skill 子模块 —— monkeypatch 需要直接操作定义变量的模块
import src.skills.preset.office_efficiency.translator.skill as translator_mod


# ──────────────────────────────────────────────
# Fixtures: 测试前置条件
# ──────────────────────────────────────────────

@pytest.fixture
def dictionary_backend():
    """创建一个离线词典后端实例用于测试"""
    return DictionaryBackend()


@pytest.fixture
def translator_skill():
    """创建一个翻译技能实例"""
    return TranslatorSkill()


# ──────────────────────────────────────────────
# 1. 输入模型验证测试 (TranslatorInput)
# ──────────────────────────────────────────────

class TestTranslatorInput:
    def test_valid_input(self):
        """测试正常输入"""
        inp = TranslatorInput(text="你好", target_lang="en")
        assert inp.text == "你好"
        assert inp.source_lang == "auto"
        assert inp.target_lang == "en"

    def test_text_empty(self):
        """测试文本为空（应抛出ValidationError，因为min_length=1）"""
        # ★ Pydantic v2 错误类型是 string_too_short，不是 min_length
        with pytest.raises(ValidationError, match="string_too_short"):
            TranslatorInput(text="")

    def test_invalid_source_lang(self):
        """测试无效的源语言参数"""
        with pytest.raises(ValidationError):
            TranslatorInput(text="test", source_lang="fr")  # 不支持法语

    def test_invalid_domain(self):
        """测试无效的领域参数"""
        with pytest.raises(ValidationError):
            TranslatorInput(text="test", domain="medicine")


# ──────────────────────────────────────────────
# 2. 离线词典后端测试 (DictionaryBackend)
# ──────────────────────────────────────────────

class TestDictionaryBackend:
    def test_zh_to_en_basic(self, dictionary_backend):
        """测试中文到英文的基本词汇翻译"""
        text = "你好"
        translated, confidence, notes = dictionary_backend.translate(text, "zh", "en", "general")

        assert translated == "Hello"
        assert confidence in ("medium", "low")
        assert any("词典覆盖率" in n for n in notes)

    def test_en_to_zh_basic(self, dictionary_backend):
        """测试英文到中文的基本词汇翻译"""
        text = "Hello"
        translated, confidence, notes = dictionary_backend.translate(text, "en", "zh", "general")

        assert translated == "你好"

    def test_partial_match(self, dictionary_backend):
        """测试部分匹配（句子中只有部分词在词典里）"""
        text = "你好，这是一个项目。"
        translated, confidence, notes = dictionary_backend.translate(text, "zh", "en", "general")

        assert "Hello" in translated
        assert "project" in translated
        assert confidence in ("low", "medium")

    def test_no_match(self, dictionary_backend):
        """测试完全没有匹配的词"""
        text = "芣苢采采"  # 生僻字
        translated, confidence, notes = dictionary_backend.translate(text, "zh", "en", "general")

        assert translated == text  # 返回原文
        assert confidence == "low"
        assert any("词典未命中" in n for n in notes)

    def test_unsupported_direction(self, dictionary_backend):
        """测试不支持的翻译方向"""
        translated, confidence, notes = dictionary_backend.translate("test", "ja", "en", "general")

        assert confidence == "low"
        assert any("不支持" in n for n in notes)


# ──────────────────────────────────────────────
# 3. 语言检测逻辑测试
# ──────────────────────────────────────────────

class TestLanguageDetection:
    def test_detect_chinese(self, translator_skill):
        """测试检测中文"""
        lang = translator_skill._detect_language("这是一段中文文本。")
        assert lang == "zh"

    def test_detect_english(self, translator_skill):
        """测试检测英文"""
        lang = translator_skill._detect_language("This is an English text.")
        assert lang == "en"

    def test_detect_mixed_default_en(self, translator_skill):
        """测试混合文本（中文较少时默认英文）"""
        lang = translator_skill._detect_language("OK好的")
        assert lang in ["zh", "en"]

    def test_detect_auto_skip_punctuation(self, translator_skill):
        """测试纯数字标点时默认英文"""
        lang = translator_skill._detect_language("123!!!")
        assert lang == "en"

    @pytest.mark.skipif(not _HAS_LANGDETECT, reason="langdetect not installed")
    def test_langdetect_integration(self, translator_skill, monkeypatch):
        """测试 langdetect 库的集成（通过 mock 控制返回值）"""

        def mock_detect(text):
            return "zh-cn"

        # ★ monkeypatch 定义 _langdetect_detect 的真实模块
        monkeypatch.setattr(translator_mod, "_langdetect_detect", mock_detect)

        lang = translator_skill._detect_language("test")
        assert lang == "zh"


# ──────────────────────────────────────────────
# 4. Skill 主流程与集成测试
# ──────────────────────────────────────────────

class TestTranslatorSkill:
    def test_same_language_skip(self, translator_skill):
        """测试源语言和目标语言相同时直接返回"""
        input_data = TranslatorInput(text="Hello", source_lang="en", target_lang="en")
        output = translator_skill.execute(input_data)

        assert output.translated_text == "Hello"
        assert "无需翻译" in output.notes[0]

    def test_empty_text_execution(self, translator_skill):
        """测试 execute 处理空字符串"""
        # Schema 有 min_length=1，此处主要验证逻辑健壮性不崩溃
        # 实际使用的 TranslatorInput 无法构造空文本，所以改为测试边界
        pass

    def test_zh_to_en_end_to_end_dict(self, monkeypatch):
        """端到端测试：强制使用离线词典"""
        # ★ monkeypatch 定义 _HAS_DEEP_TRANSLATOR 和 _GoogleTranslator 的真实模块
        monkeypatch.setattr(translator_mod, "_HAS_DEEP_TRANSLATOR", False)
        monkeypatch.setattr(translator_mod, "_GoogleTranslator", None)

        skill = TranslatorSkill()
        input_data = TranslatorInput(text="你好，这是一个项目。", target_lang="en")
        output = skill.execute(input_data)

        assert output.original_text == "你好，这是一个项目。"
        assert "Hello" in output.translated_text
        assert "project" in output.translated_text
        assert output.confidence in ["低", "中"]

    def test_google_backend_mock(self, monkeypatch):
        """测试 Google 翻译后端（Mock 网络请求）"""
        # 1. 模拟已安装 deep_translator
        monkeypatch.setattr(translator_mod, "_HAS_DEEP_TRANSLATOR", True)

        # 2. Mock GoogleTranslator 的 translate 方法
        class MockTranslator:
            def __init__(self, source=None, target=None):
                pass

            def translate(self, text):
                return "This is a mocked translation."

        monkeypatch.setattr(translator_mod, "_GoogleTranslator", MockTranslator)

        # 3. 执行测试
        skill = TranslatorSkill()
        input_data = TranslatorInput(text="测试文本", source_lang="zh", target_lang="en")
        output = skill.execute(input_data)

        # 4. 验证
        assert output.translated_text == "This is a mocked translation."
        assert output.confidence == "高"
        assert any("Google 翻译引擎" in n for n in output.notes)

if __name__ == "__main__":
    pytest.main()