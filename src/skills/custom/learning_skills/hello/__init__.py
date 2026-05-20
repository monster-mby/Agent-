"""
Hello 技能包
对外暴露问候技能的输入模型、输出模型、核心执行类
"""
# 从当前目录的hello_skill模块，导入所有要对外暴露的内容
from .hello_skill import HelloInput, HelloOutput, HelloSkill

# 声明包的公共接口，严格对应上面导入的内容
__all__ = ["HelloInput", "HelloOutput", "HelloSkill"]