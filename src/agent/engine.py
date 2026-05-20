"""
Agent 编排引擎 — 集成 SkillManager + 三层容错
"""

import json
import os
from typing import Any, Dict, List, Optional, Type

from src.skills.base import BaseSkill, SkillManager
from src.core.model_client import LiteLLMClient
from src.skills.custom.rag_skills.text_embedder.skill import TextEmbedderSkill
from src.skills.custom.rag_skills.graphrag_indexer.skill import GraphRAGIndexerSkill
from src.skills.custom.rag_skills.graphrag_searcher.skill import GraphRAGSearcherSkill
from src.skills.custom.rag_skills.graphrag_searcher import skill as graphrag_skill
from .llm_client import LLMClient, SimulatedLLM, OpenAICompatibleLLM



class AgentEngine:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.skill_manager = SkillManager()
        self._history: List[Dict[str, str]] = []
        self.llm = llm_client or SimulatedLLM()
        # 注册 GraphRAG 技能
        self._register_graphrag_skills()

    def _register_graphrag_skills(self):
        """注册 GraphRAG 相关技能并注入依赖"""
        try:
            # 1. 注册两个 GraphRAG 技能
            self.skill_manager.register(GraphRAGIndexerSkill)
            self.skill_manager.register(GraphRAGSearcherSkill)

            # 2. 为检索技能注入 LLM 和 Embedding 客户端
            # 使用 LiteLLMClient 作为 LLM 客户端
            llm_client = LiteLLMClient()

            # 使用 TextEmbeddingSkill 作为 Embedding 客户端
            embedding_client = TextEmbedderSkill()

            # 注入到 graphrag_searcher 模块
            graphrag_skill.configure_graphrag_searcher(
                llm_client=llm_client,
                embedding_client=embedding_client,
            )

            import logging
            logger = logging.getLogger(__name__)
            logger.info("✅ GraphRAG 技能已注册并配置完成")

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ GraphRAG 技能注册失败: {e}")

    def register_skill(self, skill_cls: Type[BaseSkill]) -> None:
        """注册一个技能（委托给 SkillManager）"""
        self.skill_manager.register(skill_cls)

    def register_skills(self, *skill_cls_list: Type[BaseSkill]) -> None:
        """批量注册技能"""
        for skill_cls in skill_cls_list:
            self.register_skill(skill_cls)

    def unregister_skill(self, name: str) -> bool:
        """注销一个技能"""
        return self.skill_manager.unregister(name)

    def get_registered_skills(self) -> List[str]:
        return self.skill_manager.get_all()

    def get_skill(self, name: str) -> Optional[Type[BaseSkill]]:
        return self.skill_manager.get(name)

    def _build_tools_for_llm(self) -> List[Dict]:
        return self.skill_manager.get_tools_for_llm()

    def run(self, user_input: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})

        tools = self._build_tools_for_llm() if self.skill_manager.count > 0 else None
        llm_response = self.llm.chat(messages, tools=tools)

        skill_calls = []
        final_content = llm_response.get("content", "")

        if llm_response.get("tool_calls"):
            for call in llm_response["tool_calls"]:
                skill_name = call["name"]
                arguments = call["arguments"]

                # 使用 SkillManager.call() 执行技能（自带输入验证、重试、熔断）
                result = self.skill_manager.call(skill_name, **arguments)

                skill_calls.append(result)

                # 将执行结果追加到消息中
                messages.append({
                    "role": "tool",
                    "tool_call_id": skill_name,
                    "content": json.dumps(
                        result.get("output", result.get("error", "")),
                        ensure_ascii=False
                    )
                })

            # 让 LLM 生成最终回复
            if skill_calls:
                final_llm = self.llm.chat(messages, tools=None)
                final_content = final_llm.get("content", "处理完成。")

        # 更新历史
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": final_content})

        return {
            "response": final_content,
            "skill_calls": skill_calls,
            "history": self._history.copy()
        }

    def reset_history(self) -> None:
        self._history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return self._history.copy()

    def __repr__(self) -> str:
        return f"AgentEngine(skills=[{', '.join(self.skill_manager.get_all()) or '无'}])"