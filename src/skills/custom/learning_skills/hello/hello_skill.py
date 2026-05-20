"""HelloSkill - 技能基类验证示例"""

from pydantic import BaseModel, Field
from src.skills.base.base_skill import BaseSkill


# ── 输入模型 ──
class HelloInput(BaseModel):
    name: str = Field(default="朋友", description="要问候的人名")
    language: str = Field(default="zh", pattern="^(zh|en)$", description="问候语言")


# ── 输出模型 ──
class HelloOutput(BaseModel):
    message: str = Field(description="问候消息")
    greeted_name: str = Field(description="被问候的人名")
    language: str = Field(default="zh", description="使用的语言")


class HelloSkill(BaseSkill):
    """最简单的技能：返回问候语"""

    name = "hello"
    description = "向用户打招呼，返回友好的问候语"
    version = "0.1.0"
    author = "demo"

    # 用 Pydantic Model 类作为 Schema
    input_schema = HelloInput
    output_schema = HelloOutput

    def execute(self, **kwargs) -> dict:
        """执行问候逻辑"""
        name = kwargs.get("name", "朋友")
        language = kwargs.get("language", "zh")

        if language == "zh":
            message = f"你好，{name}！"
        else:
            message = f"Hello, {name}!"

        return {
            "message": message,
            "greeted_name": name,
            "language": language,
        }