"""
Skill Manager - 技能注册、发现、调用管理器。

职责：
1. 技能注册与注销（单/批量）
2. 技能发现（按名称、关键词、触发器）
3. 技能调用（实例化 + 执行 + 容错）
4. 将技能描述格式化为 LLM 工具列表
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from .base_skill import BaseSkill

logger = logging.getLogger(__name__)


class SkillManager:
    """
    技能管理器。

    用法：
        manager = SkillManager()
        manager.register(HelloSkill)
        manager.register(CodeReviewSkill)

        # 发现技能
        skills = manager.find_by_trigger("审查代码")

        # 调用技能
        result = manager.call("code_review", code="print('hello')", language="python")

        # 获取 LLM 工具描述
        tools = manager.get_tools_for_llm()
    """

    def __init__(self):
        """初始化技能注册表"""
        self._registry: Dict[str, Type[BaseSkill]] = {}

    # ==================== 注册与注销 ====================

    def register(self, skill_cls: Type[BaseSkill]) -> None:
        """
        注册一个技能。

        Args:
            skill_cls: BaseSkill 的子类

        Raises:
            TypeError: 如果不是 BaseSkill 的子类
            ValueError: 如果技能名重复
        """
        if not isinstance(skill_cls, type) or not issubclass(skill_cls, BaseSkill):
            raise TypeError(f"{skill_cls.__name__} 必须是 BaseSkill 的子类")

        name = skill_cls.name
        if not name:
            raise ValueError(f"技能类 {skill_cls.__name__} 必须定义 name 属性")

        if name in self._registry:
            raise ValueError(f"技能 '{name}' 已注册")

        self._registry[name] = skill_cls
        logger.info(f"✅ 技能已注册: {name} v{skill_cls.version} by {skill_cls.author or 'unknown'}")

    def register_many(self, *skill_cls_list: Type[BaseSkill]) -> List[str]:
        """批量注册，返回成功注册的技能名列表"""
        registered = []
        for skill_cls in skill_cls_list:
            try:
                self.register(skill_cls)
                registered.append(skill_cls.name)
            except (TypeError, ValueError) as e:
                logger.warning(f"⚠️ 注册失败 {skill_cls.__name__}: {e}")
        return registered

    def unregister(self, name: str) -> bool:
        """
        注销一个技能。

        Returns:
            bool: 是否成功注销
        """
        if name in self._registry:
            del self._registry[name]
            logger.info(f"🗑️ 技能已注销: {name}")
            return True
        logger.warning(f"⚠️ 技能 '{name}' 未注册，无法注销")
        return False

    # ==================== 发现 ====================

    def get_all(self) -> List[str]:
        """获取所有已注册的技能名称列表"""
        return list(self._registry.keys())

    def get(self, name: str) -> Optional[Type[BaseSkill]]:
        """根据名称获取技能类"""
        return self._registry.get(name)

    def find_by_trigger(self, text: str) -> List[Type[BaseSkill]]:
        """
        根据用户输入文本，匹配触发关键词。

        返回按匹配度排序的技能列表（匹配关键词最多的在前）。
        """
        if not text:
            return []

        text_lower = text.lower()
        matched = []

        for skill_cls in self._registry.values():
            triggers = skill_cls.triggers
            if not triggers:
                continue

            # 计算匹配数
            match_count = sum(1 for t in triggers if t.lower() in text_lower)
            if match_count > 0:
                matched.append((match_count, skill_cls))

        # 按匹配数降序排序
        matched.sort(key=lambda x: x[0], reverse=True)
        return [skill_cls for _, skill_cls in matched]

    def find_by_keyword(self, keyword: str) -> List[Type[BaseSkill]]:
        """
        根据关键词搜索技能（匹配 name 和 description）。

        返回按相关性排序的技能列表。
        """
        if not keyword:
            return []

        keyword_lower = keyword.lower()
        scored = []

        for skill_cls in self._registry.values():
            score = 0
            # name 匹配（权重高）
            if keyword_lower in skill_cls.name.lower():
                score += 10
            # description 匹配
            if keyword_lower in skill_cls.description.lower():
                score += 5
            # triggers 匹配
            for t in skill_cls.triggers:
                if keyword_lower in t.lower():
                    score += 3

            if score > 0:
                scored.append((score, skill_cls))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill_cls for _, skill_cls in scored]

    # ==================== 调用 ====================

    def call(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        """
        调用一个技能。

        Args:
            skill_name: 技能名称
            **kwargs: 传递给技能 execute 的参数

        Returns:
            {
                "skill": str,
                "status": "success" | "error",
                "input": dict,
                "output": Any,
                "error": str | None
            }
        """
        skill_cls = self._registry.get(skill_name)
        if skill_cls is None:
            return {
                "skill": skill_name,
                "status": "error",
                "input": kwargs,
                "output": None,
                "error": f"技能 '{skill_name}' 未注册"
            }

        try:
            # 实例化
            skill_instance = skill_cls()

            # 第一层容错：输入验证 + 默认值填充
            if skill_instance.input_schema is not None:
                # 使用 Pydantic 模型验证并填充默认值
                validated_input = skill_instance.input_schema(**kwargs)
                # 将 Pydantic 对象转为 dict，传给 execute()
                result = skill_instance.execute(validated_input)
            else:
                # 无 schema 时，手动验证输入
                skill_instance.validate_input(**kwargs)
                result = skill_instance.execute(**kwargs)

            # 第三层容错：输出验证
            result = skill_instance.validate_output(
                result.model_dump() if hasattr(result, "model_dump")
                else (result if isinstance(result, dict) else result)
            )

            return {
                "skill": skill_name,
                "status": "success",
                "input": kwargs,
                "output": result,
                "error": None
            }

        except Exception as e:
            logger.error("技能 '%s' 执行失败: %s", skill_name, e)
            return {
                "skill": skill_name,
                "status": "error",
                "input": kwargs,
                "output": None,
                "error": str(e)
            }

    # ==================== LLM 工具描述 ====================

    def get_tools_for_llm(self) -> List[Dict]:
        """
        将注册的所有技能转换为 LLM 可理解的工具描述列表。

        返回格式符合 OpenAI Function Calling 规范。
        """
        tools = []
        for name, skill_cls in self._registry.items():
            parameters = {}
            # 临时实例化以获取 input_schema（可能失败，如 GraphRAG 未配置）
            try:
                skill_instance = skill_cls()
                if skill_instance.input_schema is not None:
                    try:
                        parameters = skill_instance.input_schema.model_json_schema()
                    except Exception:
                        parameters = {"type": "object", "properties": {}}
            except Exception as e:
                # 实例化失败时，尝试从类变量获取 schema
                logger.debug(f"⚠️ 技能 '{name}' 实例化失败，使用类变量 schema: {e}")
                input_schema = getattr(skill_cls, 'input_schema', None)
                if input_schema is not None:
                    try:
                        parameters = input_schema.model_json_schema()
                    except Exception:
                        parameters = {"type": "object", "properties": {}}

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": skill_cls.description,
                    "parameters": parameters
                }
            })
        return tools

    # ==================== 元数据 ====================

    def get_metadata(self, name: str) -> Optional[Dict]:
        """获取技能元数据"""
        skill_cls = self._registry.get(name)
        if skill_cls is None:
            return None

        return {
            "name": skill_cls.name,
            "description": skill_cls.description,
            "version": skill_cls.version,
            "author": skill_cls.author,
            "triggers": skill_cls.triggers,
            "changelog": skill_cls.changelog,
            "has_input_schema": skill_cls.input_schema is not None,
            "has_output_schema": skill_cls.output_schema is not None,
        }

    def list_all_metadata(self) -> List[Dict]:
        """获取所有技能的元数据"""
        return [self.get_metadata(name) for name in self._registry]

        # ==================== Orchestrator 兼容接口 ====================

    def generate_tool_descriptions(self) -> List[Dict]:
        """别名，供 SkillOrchestrator._sync_tools_to_llm() 调用"""
        return self.get_tools_for_llm()

    def invoke(self, skill_name: str, **kwargs) -> Any:
        """薄封装 call()，成功返回 output，失败抛异常"""
        result = self.call(skill_name, **kwargs)
        if result["status"] == "error":
            raise RuntimeError(result["error"] or f"技能 '{skill_name}' 执行失败")
        return result["output"]

    def list_all(self) -> List[Dict]:
        """别名，供 SkillOrchestrator._heuristic_match_from_text() 调用"""
        return self.list_all_metadata()

    @property
    def count(self) -> int:
        """已注册技能数量"""
        return len(self._registry)

    @property
    def count(self) -> int:
        """已注册技能数量"""
        return len(self._registry)

    def __repr__(self) -> str:
        skills = ", ".join(self._registry.keys()) if self._registry else "无"
        return f"SkillManager(count={self.count}, skills=[{skills}])"