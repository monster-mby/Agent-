"""
LLM 客户端抽象层。

提供多种实现：
1. SimulatedLLM           —— 模拟模式，无需 API
2. OpenAICompatibleLLM    —— 通用 OpenAI 兼容（DeepSeek / 豆包 / OpenAI）
3. create_deepseek()      —— 快捷创建 DeepSeek 客户端
4. create_doubao()        —— 快捷创建豆包客户端
5. create_openai()        —— 快捷创建 OpenAI 客户端
"""
import json
import re
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════
#  抽象基类
# ═══════════════════════════════════════════════

class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        与 LLM 对话。

        返回格式：
        {
            "content": "自然语言回复",
            "tool_calls": [
                {
                    "name": "skill_name",
                    "arguments": {"param1": "value1", ...}
                }
            ]
        }
        """
        ...


# ═══════════════════════════════════════════════
#  通用 OpenAI 兼容客户端
#  ── 支持：OpenAI、DeepSeek、豆包、各种自部署模型
# ═══════════════════════════════════════════════

class OpenAICompatibleLLM(LLMClient):
    """通用的 OpenAI 兼容客户端 —— 一个类覆盖所有 OpenAI-API 兼容服务"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """核心对话方法 —— 懒加载 openai 库，按需 import"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "使用真实 LLM 需要安装 openai 库:\n  pip install openai"
            )

        # ── 构建客户端 ──
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**client_kwargs)

        # ── 构建请求参数 ──
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 工具调用（Function Calling）
        if tools:
            # 转换工具格式：你的 tools 已经是 OpenAI format，直接传入
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # ── 发送请求 ──
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # ── 清洗返回值 → 项目统一格式 ──
        result: Dict[str, Any] = {
            "content": message.content or "",
            "tool_calls": [],
        }

        if message.tool_calls:
            for tc in message.tool_calls:
                # arguments 可能是 str 或 dict（不同服务有差异）
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result["tool_calls"].append({
                    "name": tc.function.name,
                    "arguments": args,
                })

        return result

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleLLM(model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )


# ═══════════════════════════════════════════════
#  便捷工厂函数 —— 一行创建
# ═══════════════════════════════════════════════

def create_deepseek(
    api_key: Optional[str] = None,
    model: str = "deepseek-v4-pro",
) -> OpenAICompatibleLLM:
    """创建 DeepSeek 客户端"""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或直接传入 api_key")
    return OpenAICompatibleLLM(
        model=model,
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def create_doubao(
    api_key: Optional[str] = None,
    model: str = "ep-20260420200124-kddng",
) -> OpenAICompatibleLLM:
    """创建豆包（火山引擎）客户端"""
    key = api_key or os.getenv("DOUBAO_API_KEY")
    if not key:
        raise ValueError("请设置 DOUBAO_API_KEY 环境变量或直接传入 api_key")
    return OpenAICompatibleLLM(
        model=model,
        api_key=key,
        base_url=os.getenv(
            "DOUBAO_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3"
        ),
    )


def create_openai(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> OpenAICompatibleLLM:
    """创建 OpenAI 客户端"""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或直接传入 api_key")
    return OpenAICompatibleLLM(
        model=model,
        api_key=key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


# ═══════════════════════════════════════════════
#  模拟 LLM（保留，开发/测试用）
# ═══════════════════════════════════════════════

# 停用词表
_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'for',
    'of', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'not',
    'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'shall', 'about',
    'into', 'over', 'after', 'before', 'between', 'under', 'above',
    'up', 'down', 'out', 'off', 'just', 'also', 'very', 'too',
}


class SimulatedLLM(LLMClient):
    """
    模拟 LLM：基于 skills_meta 动态匹配技能，无需为每个技能写 if-else。

    适用场景：
    - 本地开发调试
    - 单元测试（不依赖网络和 API Key）
    - 演示
    """

    def __init__(self, skills_meta: List[Dict] = None):
        self.skills_meta = skills_meta or []


    def register_tools(self, tools: List[Dict]) -> None:
        """
        预注册技能列表（供 SkillOrchestrator._sync_tools_to_llm 调用）。

        将 OpenAI Function Calling 格式转为 SimulatedLLM 内部使用的
        skills_meta 格式，这样后续 chat() 调用即使不传 tools 也能匹配。
        """
        self.skills_meta = []
        for tool in tools:
            func = tool.get("function", tool)
            self.skills_meta.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """模拟 LLM 响应"""
        user_msgs = [m for m in messages if m["role"] == "user"]
        if not user_msgs:
            return {"content": "你好！请告诉我你需要什么帮助。", "tool_calls": []}

        last_input = user_msgs[-1]["content"]

        if tools:
            self.skills_meta = []
            for tool in tools:
                func = tool.get("function", tool)
                self.skills_meta.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })

        match_result = self._match_skill(last_input)

        if match_result:
            return {
                "content": f"我决定调用 {match_result['name']} 技能来处理你的请求。",
                "tool_calls": [match_result],
            }

        return {
            "content": (
                f"收到你的消息：「{last_input}」\n"
                f"可用的技能有：{', '.join(self._get_skill_names())}。"
                f"请告诉我你想使用哪个技能？"
            ),
            "tool_calls": [],
        }

    def _get_skill_names(self) -> List[str]:
        return [s["name"] for s in self.skills_meta]

    def _extract_keywords(self, meta: Dict) -> List[str]:
        keywords = set()

        name = meta.get("name", "")
        keywords.add(name)
        for part in name.replace("-", "_").split("_"):
            part = part.strip().lower()
            if part and part not in _STOP_WORDS:
                keywords.add(part)

        desc = meta.get("description", "").lower()

        # 英文：>=3 字母的单词
        english_words = re.findall(r'\b[a-z]{3,}\b', desc)
        for w in english_words:
            if w not in _STOP_WORDS:
                keywords.add(w)

        # ✨ 中文：连续 >=2 汉字的词组 + 2-gram 滑动窗口
        chinese_phrases = re.findall(r'[\u4e00-\u9fff]{2,}', desc)
        for phrase in chinese_phrases:
            keywords.add(phrase)
            for i in range(len(phrase) - 1):
                keywords.add(phrase[i:i + 2])

        params = meta.get("parameters", {}).get("properties", {})
        for param_name in params:
            keywords.add(param_name)
            for part in param_name.replace("-", "_").split("_"):
                part = part.strip().lower()
                if part and part not in _STOP_WORDS and len(part) >= 3:
                    keywords.add(part)

        return list(keywords)

    def _calculate_match_score(self, user_lower: str, keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if kw in user_lower:
                score += min(len(kw), 20)
        return score

    def _match_skill(self, user_input: str) -> Optional[Dict]:
        user_lower = user_input.lower()
        best_meta = None
        best_score = 0

        for meta in self.skills_meta:
            keywords = self._extract_keywords(meta)
            score = self._calculate_match_score(user_lower, keywords)
            if score > best_score:
                best_score = score
                best_meta = meta

        if best_meta and best_score > 0:
            return self._build_call(best_meta, user_input)
        return None

    def _build_call(self, meta: Dict, user_input: str) -> Dict:
        name = meta["name"]
        properties = meta.get("parameters", {}).get("properties", {})
        arguments = {}
        for param_name, param_info in properties.items():
            value = self._extract_param(user_input, param_name, param_info)
            if value is not None:
                arguments[param_name] = value

        # 兜底：为所有缺失的必填字段填值
        required = meta.get("parameters", {}).get("required", [])
        for param_name in required:
            if param_name not in arguments:
                param_info = properties.get(param_name, {})
                ptype = param_info.get("type", "string")
                if ptype == "string":
                    arguments[param_name] = user_input.strip()
                elif ptype == "array":
                    arguments[param_name] = [user_input.strip()]

        return {"name": name, "arguments": arguments}

    def _extract_param(
        self, user_input: str, param_name: str, param_info: Dict
    ) -> Any:
        param_type = param_info.get("type", "string")
        param_label = param_name.lower().replace("_", " ").replace("-", " ")

        value = self._extract_by_pattern(user_input, param_label)
        if value is not None:
            return self._convert_type(value, param_type)

        special_extractors = {
            "code": self._extract_code,
            "name": self._extract_name_param,
            "language": self._extract_language,
        }
        extractor = special_extractors.get(param_name)
        if extractor:
            value = extractor(user_input)
            if value is not None:
                return value

        return param_info.get("default")

    @staticmethod
    def _extract_by_pattern(user_input: str, param_label: str) -> Optional[str]:
        patterns = [
            rf'{re.escape(param_label)}[：:]\s*(.+?)(?:[，,。.\n]|$)',
            rf'{re.escape(param_label)}\s*[=:=]\s*(.+?)(?:[，,。.\n]|$)',
        ]
        for p in patterns:
            match = re.search(p, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_code(user_input: str) -> str:
        if "```" in user_input:
            parts = user_input.split("```")
            if len(parts) >= 3:
                idx = 1
                if parts[1].strip() in (
                    "python", "py", "javascript", "js", "java", "go", ""
                ):
                    idx = 2
                lines = parts[idx].split("\n")
                return "\n".join(lines).strip()
        sample_map = {
            "python": "def hello():\n    print('hello')",
            "javascript": "function hello() {\n    console.log('hello');\n}",
        }
        for lang, code in sample_map.items():
            if lang in user_input.lower():
                return code
        return "# 示例代码\nx = 1"

    @staticmethod
    def _extract_name_param(user_input: str) -> Optional[str]:
        patterns = [
            r"(?:hello|hi|hey)\s*[,，]?\s*([\w\u4e00-\u9fff]+)",
            r"你好\s*[，,，]?\s*([\w\u4e00-\u9fff]+)",
            r"(?:问候|打招呼)\s+([\w\u4e00-\u9fff]+)",
        ]
        for p in patterns:
            match = re.search(p, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_language(user_input: str) -> Optional[str]:
        lang_map = {
            "python": ["python", "py"],
            "javascript": ["javascript", "js", "node"],
            "java": ["java"],
            "go": ["go", "golang"],
        }
        for lang, keywords in lang_map.items():
            if any(kw in user_input.lower() for kw in keywords):
                return lang
        return None

    @staticmethod
    def _convert_type(value: str, param_type: str):
        if param_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return None
        if param_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        if param_type == "boolean":
            return value.lower() in ("true", "yes", "1", "是", "ok")
        return value

# ═══════════════════════════════════════════════
#  LiteLLM 适配器 —— 桥接 LiteLLMClient ↔ LLMClient
# ═══════════════════════════════════════════════

# class LiteLLMAdapter(LLMClient):
#     """
#     将 LiteLLMClient 适配为 LLMClient 接口。
#
#     解决的问题：
#     - LiteLLMClient.chat() → ChatResponse (Pydantic)
#     - LLMClient.chat()       → Dict[str, Any]
#     - 现有 SkillOrchestrator 依赖 dict 格式
#
#     适配器职责：
#     1. 调用 LiteLLMClient.chat()
#     2. ChatResponse → {"content": ..., "tool_calls": [...]}
#     3. register_tools() 存储工具列表供 chat() 使用
#     """
#
#     def __init__(self, config=None, **kwargs):
#         """
#         Args:
#             config: LLMConfig 或 None（自动从 .env 加载）
#             **kwargs: 透传给 LiteLLMClient
#         """
#         # 懒加载，避免强依赖 litellm
#         from src.core.model_client import LiteLLMClient
#
#         if config is not None:
#             self._client = LiteLLMClient(config=config, **kwargs)
#         else:
#             self._client = LiteLLMClient(**kwargs)
#
#         self._tools: List[Dict] = []
#
#     def register_tools(self, tools: List[Dict]) -> None:
#         """
#         注册工具列表。
#
#         供 SkillOrchestrator._sync_tools_to_llm() 调用。
#         """
#         self._tools = tools
#
#     def chat(
#         self,
#         messages: List[Dict[str, str]],
#         tools: Optional[List[Dict]] = None,
#     ) -> Dict[str, Any]:
#         """
#         与 LLM 对话。
#
#         Returns:
#             {"content": str, "tool_calls": [{"name": ..., "arguments": {...}}]}
#         """
#         from src.core.model_client import ChatMessage
#
#         # 1. messages → ChatMessage 列表
#         chat_messages = [
#             m if isinstance(m, ChatMessage) else ChatMessage(**m)
#             for m in messages
#         ]
#
#         # 2. tools：优先用传入的，否则用预注册的
#         effective_tools = tools if tools is not None else self._tools
#
#         # 3. 调用真实 LLM
#         response = self._client.chat(
#             messages=chat_messages,
#             tools=effective_tools if effective_tools else None,
#         )
#
#         # 4. ChatResponse → dict
#         return {
#             "content": response.content,
#             "tool_calls": [
#                 {"name": tc.name, "arguments": tc.arguments}
#                 for tc in response.tool_calls
#             ],
#         }
#
#     def __repr__(self) -> str:
#         return f"LiteLLMAdapter(client={self._client!r})"