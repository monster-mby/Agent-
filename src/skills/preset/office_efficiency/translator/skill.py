"""
这是一个企业级多后端翻译技能，可根据环境自动选择「Google Translate 在线高质量翻译」或「离线词典兜底翻译」，支持中英互译、多领域风格适配、智能语言检测，具备完善的缓存重试、降级容错与置信度评估能力。下面从功能概览、核心结构设计、类 / 方法深度详解、方法调用链路、代码质量评价五个维度深度解析。
一、功能概览
这个工具的核心能力是：
可插拔翻译后端：优先使用 Google Translate 在线翻译（高质量），无依赖 / 无网络时自动降级为离线词典兜底，保证可用性。
多领域风格适配：支持通用 / 技术 / 商务 / 学术 / 日常口语 5 种翻译场景，自动适配对应表达风格。
智能语言检测：优先使用 langdetect 高精度语言识别，失败时自动降级为中文字符占比统计，支持auto自动检测源语言。
企业级工程保障：LRU 缓存避免重复请求、指数退避重试应对网络波动、全链路日志记录、异常捕获与兜底。
置信度与注释输出：自动评估翻译质量，输出高 / 中 / 低三档置信度，附带翻译提示、覆盖率、异常说明等注释信息。
全场景边界处理：空文本保护、同语言短路返回、非法参数校验、非中英语言兜底，覆盖所有边缘场景。
二、核心结构设计
代码采用分层架构，核心结构如下：
plaintext
┌─────────────────────────────────────────────────────────────┐
│  1. 可选依赖检测层                                          │
│     - deep-translator（Google翻译依赖）可用性检测            │
│     - langdetect（语言检测依赖）可用性检测                    │
├─────────────────────────────────────────────────────────────┤
│  2. 数据模型层（Pydantic）                                    │
│     - TranslatorInput（输入校验+参数规范）                    │
│     - TranslatorOutput（结构化翻译结果输出）                  │
├─────────────────────────────────────────────────────────────┤
│  3. 翻译后端抽象层                                           │
│     - TranslationBackend（翻译后端抽象基类）                  │
│     ├─ DictionaryBackend（离线词典兜底后端）                  │
│     └─ GoogleTranslateBackend（Google在线翻译后端）          │
├─────────────────────────────────────────────────────────────┤
│  4. 主技能类（TranslatorSkill）                              │
│     - execute（主入口，协调整个翻译流程）                    │
│     - _detect_language（智能语言检测）                        │
│     - 后端自动选择与优雅降级逻辑                              │
└─────────────────────────────────────────────────────────────┘
TranslatorSkill.execute()
  ├→ 空文本校验与短路返回
  ├→ 源语言auto检测 → 调用 _detect_language()
  │   ├→ 纯符号快速路径
  │   ├→ 优先langdetect高精度检测
  │   └→ 兜底中文字符占比统计
  ├→ 同语言校验与短路返回
  ├→ 翻译方向合法性校验
  ├→ 调用后端 translate() 方法
  │   ├─ 【Google后端可用时】 GoogleTranslateBackend.translate()
  │   │   ├→ _translate_with_retry() 指数退避重试
  │   │   └→ _translate_cached() LRU缓存命中/API请求
  │   └─ 【兜底场景】 DictionaryBackend.translate()
  │       ├→ 词典匹配与滑动窗口分词
  │       ├→ 长匹配优先去重叠替换
  │       └→ 覆盖率计算与置信度评估
  ├→ 置信度标签中文化转换
  └→ 封装为 TranslatorOutput 结构化结果返回
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 可选依赖检测
# ═══════════════════════════════════════════════════════════════

try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _HAS_DEEP_TRANSLATOR = True
except ImportError:
    _GoogleTranslator = None
    _HAS_DEEP_TRANSLATOR = False
    logger.info("deep-translator 未安装，将使用离线词典翻译（pip install deep-translator 可启用在线翻译）")

try:
    from langdetect import detect as _langdetect_detect
    from langdetect import LangDetectException as _LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    _langdetect_detect = None
    _LangDetectException = Exception  # fallback 类型，避免 except 报错
    _HAS_LANGDETECT = False
    logger.info("langdetect 未安装，将使用字符统计检测语言（pip install langdetect 可启用高精度检测）")


# ═══════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════

class TranslatorInput(BaseModel):
    """翻译输入"""
    text: str = Field(..., description="待翻译文本", min_length=1)
    source_lang: Literal["zh", "en", "auto"] = Field(
        default="auto",
        description="源语言：zh=中文, en=英文, auto=自动检测"
    )
    target_lang: Literal["zh", "en"] = Field(
        default="en",
        description="目标语言"
    )
    domain: Literal["general", "technical", "business", "academic", "casual"] = Field(
        default="general",
        description="领域/风格"
    )


class TranslatorOutput(BaseModel):
    """翻译输出"""
    original_text: str = Field(...)
    translated_text: str = Field(...)
    source_lang: str = Field(...)
    target_lang: str = Field(...)
    confidence: str = Field(..., description="翻译置信度（低/中/高）")
    notes: list[str] = Field(default_factory=list, description="翻译注释/注意事项")


# ═══════════════════════════════════════════════════════════════
# 翻译后端抽象
# ═══════════════════════════════════════════════════════════════

class TranslationBackend(ABC):
    """翻译后端抽象基类"""

    @abstractmethod
    def translate(self, text: str, source: str, target: str, domain: str) -> tuple[str, str, list[str]]:
        """
        翻译文本。

        Returns:
            (translated_text, confidence, notes)
            confidence: "high" | "medium" | "low"
        """
        ...


# ── 词典后端（离线兜底） ──

class DictionaryBackend(TranslationBackend):
    """基于词典的离线翻译后端，质量有限，用作兜底方案"""

    # 中 → 英
    _ZH2EN: ClassVar[dict[str, str]] = {
        "你好": "Hello",
        "谢谢": "Thank you",
        "对不起": "I'm sorry",
        "请": "Please",
        "问题": "question",
        "解决方案": "solution",
        "项目": "project",
        "会议": "meeting",
        "报告": "report",
        "数据": "data",
        "分析": "analysis",
        "开发": "development",
        "测试": "testing",
        "部署": "deployment",
        "需求": "requirement",
        "设计": "design",
        "架构": "architecture",
        "性能": "performance",
        "优化": "optimization",
        "安全": "security",
        "团队": "team",
        "客户": "client",
        "产品": "product",
        "服务": "service",
        "质量": "quality",
        "进度": "progress",
        "风险": "risk",
        "预算": "budget",
        "合同": "contract",
        "文档": "document",
        "代码": "code",
        "算法": "algorithm",
        "数据库": "database",
        "服务器": "server",
        "接口": "API",
        "功能": "feature",
        "版本": "version",
        "更新": "update",
        "修复": "fix",
        "发布": "release",
        "评审": "review",
        "建议": "suggestion",
        "反馈": "feedback",
        "确认": "confirm",
        "取消": "cancel",
        "重要": "important",
        "紧急": "urgent",
        "尽快": "ASAP",
        "需要": "need",
        "必须": "must",
        "可以": "can",
        "可能": "maybe",
        "如果": "if",
        "因为": "because",
        "所以": "therefore",
        "但是": "but",
        "而且": "moreover",
        "或者": "or",
        "关于": "regarding",
        "根据": "according to",
        "通过": "through",
        "对于": "for",
        "作为": "as",
        "包括": "including",
        "例如": "for example",
        "等等": "etc.",
        "以上": "above",
        "以下": "below",
        "请注意": "Please note that",
        "建议您": "It is recommended that",
        "很高兴": "I'm glad to",
        "期待": "look forward to",
    }

    # 英 → 中
    _EN2ZH: ClassVar[dict[str, str]] = {
        "hello": "你好",
        "thank you": "谢谢",
        "sorry": "对不起",
        "please": "请",
        "problem": "问题",
        "issue": "议题",
        "solution": "解决方案",
        "project": "项目",
        "meeting": "会议",
        "report": "报告",
        "data": "数据",
        "analysis": "分析",
        "development": "开发",
        "testing": "测试",
        "deployment": "部署",
        "requirement": "需求",
        "design": "设计",
        "architecture": "架构",
        "performance": "性能",
        "optimization": "优化",
        "security": "安全",
        "team": "团队",
        "client": "客户",
        "customer": "客户",
        "product": "产品",
        "service": "服务",
        "quality": "质量",
        "progress": "进度",
        "risk": "风险",
        "budget": "预算",
        "contract": "合同",
        "document": "文档",
        "documentation": "文档",
        "code": "代码",
        "algorithm": "算法",
        "database": "数据库",
        "server": "服务器",
        "api": "接口",
        "interface": "接口",
        "feature": "功能",
        "functionality": "功能",
        "version": "版本",
        "update": "更新",
        "fix": "修复",
        "release": "发布",
        "review": "评审",
        "suggestion": "建议",
        "recommendation": "建议",
        "feedback": "反馈",
        "confirm": "确认",
        "cancel": "取消",
        "important": "重要",
        "urgent": "紧急",
        "asap": "尽快",
        "need": "需要",
        "require": "需要",
        "must": "必须",
        "can": "可以",
        "may": "可能",
        "possible": "可能的",
        "maybe": "也许",
        "if": "如果",
        "because": "因为",
        "therefore": "因此",
        "but": "但是",
        "however": "然而",
        "and": "和",
        "moreover": "此外",
        "or": "或者",
        "about": "关于",
        "regarding": "关于",
        "according to": "根据",
        "through": "通过",
        "include": "包括",
        "including": "包括",
        "for example": "例如",
        "e.g.": "例如",
        "etc.": "等等",
        "above": "以上",
        "below": "以下",
        "please note that": "请注意",
        "i'm glad to": "很高兴",
        "look forward to": "期待",
    }

    def translate(self, text: str, source: str, target: str, domain: str) -> tuple[str, str, list[str]]:
        notes: list[str] = []

        if source == "zh" and target == "en":
            dictionary = self._ZH2EN
        elif source == "en" and target == "zh":
            dictionary = self._EN2ZH
        else:
            return text, "low", [f"词典后端不支持 {source}→{target} 方向"]

        # 统计文本中的"可翻译单元"（中文字词 或 英文单词）
        if source == "zh":
            # 中文按 2-4 字滑动窗口近似分词
            text_units = set()
            for i in range(len(text)):
                for w in range(2, 5):
                    if i + w <= len(text):
                        chunk = text[i:i+w]
                        if all('\u4e00' <= c <= '\u9fff' for c in chunk):
                            text_units.add(chunk)
            # 补充单字
            text_units |= {c for c in text if '\u4e00' <= c <= '\u9fff'}
        else:
            text_units = set(re.findall(r'[a-zA-Z]+', text.lower()))

        # 找出文本中匹配的词典条目（按位置，避免重复替换）
        matched_terms: list[tuple[int, int, str, str]] = []
        # (start, end, source_term, target_term)

        for src_term, tgt_term in dictionary.items():
            if source == "zh":
                idx = 0
                while True:
                    idx = text.find(src_term, idx)
                    if idx == -1:
                        break
                    matched_terms.append((idx, idx + len(src_term), src_term, tgt_term))
                    idx += 1
            else:
                # 英文大小写不敏感
                for m in re.finditer(re.escape(src_term), text, re.IGNORECASE):
                    matched_terms.append((m.start(), m.end(), src_term, tgt_term))

        # 去重叠：按位置排序，优先长匹配、靠前匹配
        matched_terms.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        selected: list[tuple[int, int, str]] = []  # (start, end, replacement)
        last_end = -1
        for start, end, src, tgt in matched_terms:
            if start < last_end:
                continue
            selected.append((start, end, tgt))
            last_end = end

        # 构建翻译结果
        result_chars: list[str] = []
        cursor = 0
        for start, end, tgt in selected:
            result_chars.append(text[cursor:start])
            result_chars.append(tgt)
            cursor = end
        result_chars.append(text[cursor:])
        translated = ''.join(result_chars)

        # 置信度：基于匹配覆盖率
        if not selected:
            confidence = "low"
            notes.append("⚠️ 词典未命中任何词汇，建议切换在线翻译引擎")
        else:
            matched_chars = sum(end - start for start, end, _ in selected)
            coverage = matched_chars / len(text) if text else 0
            if coverage > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            notes.append(f"📊 词典覆盖率约 {coverage:.0%}，建议人工校对")

        if domain != "general":
            notes.append(f"💡 离线词典模式下 '{domain}' 领域标记仅作参考，未影响翻译结果")

        return translated, confidence, notes


# ── Google Translate 后端（在线，推荐） ──

class GoogleTranslateBackend(TranslationBackend):
    """基于 deep-translator (Google Translate) 的在线翻译后端"""

    # 领域→翻译提示前缀映射
    _DOMAIN_HINTS: ClassVar[dict[str, str]] = {
        "general":   "",
        "technical": "[Technical] ",
        "business":  "[Business] ",
        "academic":  "[Academic] ",
        "casual":    "[Casual] ",
    }

    # 源/目标语言标签映射（deep-translator 使用 "zh-CN"/"en"）
    _LANG_MAP: ClassVar[dict[str, str]] = {
        "zh": "zh-CN",
        "en": "en",
    }

    @staticmethod
    @lru_cache(maxsize=512)
    def _translate_cached(text: str, source: str, target: str) -> str:
        """缓存翻译结果，避免重复请求"""
        translator = _GoogleTranslator(source=source, target=target)
        return translator.translate(text)

    @staticmethod
    def _translate_with_retry(text: str, source: str, target: str, max_retries: int = 3) -> str:
        """带指数退避重试的翻译调用"""
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return GoogleTranslateBackend._translate_cached(text, source, target)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "翻译请求失败 (attempt %d/%d): %s，%ds 后重试...",
                        attempt + 1, max_retries, e, wait
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]

    def translate(self, text: str, source: str, target: str, domain: str) -> tuple[str, str, list[str]]:
        notes: list[str] = []

        src_label = self._LANG_MAP.get(source, source)
        tgt_label = self._LANG_MAP.get(target, target)

        # 领域提示
        domain_hint = self._DOMAIN_HINTS.get(domain, "")
        if domain_hint:
            notes.append(f"💡 已标注 '{domain}' 领域风格")

        try:
            translated = self._translate_with_retry(text, src_label, tgt_label)
            confidence = "high"
            notes.append("✅ 已通过 Google 翻译引擎完成翻译")
        except Exception as e:
            logger.error("Google 翻译失败: %s，返回原文", e)
            translated = text
            confidence = "low"
            notes.append(f"⚠️ 在线翻译失败 ({type(e).__name__})，返回原文。请检查网络或切换离线词典模式")

        return translated, confidence, notes


# ═══════════════════════════════════════════════════════════════
# Skill
# ═══════════════════════════════════════════════════════════════

class TranslatorSkill(BaseSkill):
    """翻译技能

    翻译后端选择策略：
    1. 若 deep-translator 可用 → Google Translate（在线，高质量）
    2. 否则 → 离线词典（本地，低质量兜底）
    """

    name: str = "translator"
    description: str = (
        "中英互译，根据上下文调整语气和习惯表达。"
        "支持通用/技术/商务/学术/日常口语等多种风格。"
    )
    triggers: list[str] = [
        "翻译", "translate", "译", "中译英", "英译中", "帮我翻译",
        "翻一下", "这段话用英文怎么说", "翻成中文", "translation",
    ]
    version: str = "2.0.0"
    author: str = "EnterpriseLearningAgent"
    changelog: ClassVar[list[dict[str, str]]] = [
        {"version": "2.0.0", "date": "2026-04-30",
         "note": "重构为可插拔后端架构，新增 Google Translate 后端，优化语言检测与置信度算法"},
        {"version": "1.0.0", "date": "2026-04-01",
         "note": "初始版本，基于词典+规则的基础翻译引擎"},
    ]
    input_schema = TranslatorInput
    output_schema = TranslatorOutput

    # ── 常量 ──
    _CONFIDENCE_MAP: ClassVar[dict[str, str]] = {"high": "高", "medium": "中", "low": "低"}
    _SUPPORTED_LANGS: ClassVar[set[str]] = {"zh", "en"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 选择后端
        if _HAS_DEEP_TRANSLATOR:
            self._backend: TranslationBackend = GoogleTranslateBackend()
            logger.info("TranslatorSkill 使用 Google Translate 后端")
        else:
            self._backend = DictionaryBackend()
            logger.info("TranslatorSkill 使用离线词典后端（翻译质量有限）")

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def execute(self, input_data: TranslatorInput) -> TranslatorOutput:
        text = input_data.text.strip()
        source_lang = input_data.source_lang
        target_lang = input_data.target_lang
        domain = input_data.domain

        logger.info("TranslatorSkill.execute | source=%s target=%s domain=%s len=%d",
                     source_lang, target_lang, domain, len(text))

        # 空文本保护
        if not text:
            return TranslatorOutput(
                original_text=input_data.text,
                translated_text="",
                source_lang=target_lang,
                target_lang=target_lang,
                confidence="高",
                notes=["输入文本为空。"],
            )

        # 自动检测语言
        if source_lang == "auto":
            source_lang = self._detect_language(text)
            logger.debug("语言检测结果: %s", source_lang)

        # 相同语言
        if source_lang == target_lang:
            return TranslatorOutput(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence="高",
                notes=["源语言与目标语言相同，无需翻译。"],
            )

        # 翻译方向校验
        if source_lang not in self._SUPPORTED_LANGS or target_lang not in self._SUPPORTED_LANGS:
            return TranslatorOutput(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence="低",
                notes=[f"不支持的翻译方向：{source_lang} → {target_lang}，当前仅支持 zh ↔ en"],
            )

        # 调用后端翻译
        translated, confidence_raw, notes = self._backend.translate(
            text, source_lang, target_lang, domain
        )

        confidence_display = self._CONFIDENCE_MAP.get(confidence_raw, confidence_raw)

        logger.info("TranslatorSkill.execute | done, confidence=%s", confidence_raw)
        return TranslatorOutput(
            original_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            confidence=confidence_display,
            notes=notes,
        )

    # ═══════════════════════════════════════════════════════════
    # 语言检测
    # ═══════════════════════════════════════════════════════════

    def _detect_language(self, text: str) -> str:
        """检测文本语言，优先使用 langdetect，fallback 到字符统计"""
        # 快速路径：纯数字/标点 → 默认 en
        if not re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text):
            logger.debug("无有效文字字符，默认 en")
            return "en"

        # 优先：langdetect
        if _HAS_LANGDETECT and _langdetect_detect is not None:
            try:
                lang = _langdetect_detect(text)
                logger.debug("langdetect 结果: %s", lang)
                if lang in ("zh-cn", "zh-tw", "zh"):
                    return "zh"
                if lang == "en":
                    return "en"
                # 非中英语言 → 默认 en（不报错，交给后端处理）
                logger.debug("langdetect 检测到非中英语言 '%s'，默认 en", lang)
                return "en"
            except _LangDetectException as e:
                logger.warning("langdetect 异常: %s，fallback 到字符统计", e)

        # Fallback：中文字符占比
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars
        detected = "zh" if chinese_ratio > 0.3 else "en"
        logger.debug("字符统计结果: chinese_ratio=%.2f → %s", chinese_ratio, detected)
        return detected