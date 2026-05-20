# src/skills/preset/__init__.py
from .content_creation import OutlineGeneratorSkill, TextSummarizerSkill
from .data_analysis import ChartAdvisorSkill, DataCleanerSkill
from .office_efficiency import EmailDrafterSkill, MeetingSummarizerSkill, TranslatorSkill
from .technical_development import CodeExplainerSkill, UnitTestGeneratorSkill

__all__ = [
    # Content Creation
    "OutlineGeneratorSkill",
    "TextSummarizerSkill",
    # Data Analysis
    "ChartAdvisorSkill",
    "DataCleanerSkill",
    # Office Efficiency
    "EmailDrafterSkill",
    "MeetingSummarizerSkill",
    "TranslatorSkill",
    # Technical Development
    "CodeExplainerSkill",
    "UnitTestGeneratorSkill",
]
