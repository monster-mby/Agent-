"""
真实环境客户端 — 让代码能跟真实的大模型API对话

这两个"翻译官"的作用：
1. RealLLMClient：把你的代码习惯翻译成OpenAI API听得懂的格式，再把API返回结果翻译回你代码习惯的格式
2. RealEmbeddingClient：把文字转成向量的真实API调用封装

你的核心代码一行都不用改，以前怎么跟Mock对象说话，现在就怎么跟它们说话。
"""
import os
from typing import Any, Dict, List, Optional

from src.agent.llm_client import LLMClient


class RealLLMClient(LLMClient):
    """
    真实大模型客户端 — 翻译官1号

    职责：
    1. 接收API密钥、地址、模型名
    2. 提供 .chat() 方法：你的代码给它发"用户消息+历史对话"，它去调用真实的OpenAI聊天接口
    3. 把API返回结果整理成你代码里已经在用的格式 {"content": ..., "tool_calls": [...]}
    4. 额外提供 .generate() 方法，兼容旧代码调用习惯
    5. 如果API调用出错，包装成你能看懂的错误信息，不让程序直接崩掉
    """

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.openai.com/v1",
            model: str = "gpt-4o-mini",
            temperature: float = 0.0,
            max_tokens: int = 2048,
    ):
        """
        初始化真实LLM客户端

        Args:
            api_key: OpenAI API密钥（从.env.test读取）
            base_url: API地址（默认OpenAI，可换为DeepSeek/豆包等）
            model: 使用的模型名（默认gpt-4o-mini，省钱又快）
            temperature: 温度参数（0=确定性回答，1=创造性回答）
            max_tokens: 最大生成token数
        """
        if not api_key:
            raise ValueError("API密钥不能为空，请在.env.test中配置LLM_API_KEY")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 懒加载openai库，避免未安装时报错
        self._client = None

    def _get_client(self):
        """获取OpenAI客户端实例（单例模式）"""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "使用真实LLM需要安装openai库:\n  pip install openai"
                )
        return self._client

    def chat(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            **kwargs,  # ✅ 新增：兼容 RagAnswerSkill 传入的 model, temperature 等参数
    ) -> Dict[str, Any]:
        """
        与真实LLM对话

        Args:
            messages: 对话历史，格式 [{"role": "user/assistant/system", "content": "..."}]
            tools: 可选的工具列表（Function Calling）

        Returns:
            {
                "content": "AI的自然语言回复",
                "tool_calls": [
                    {"name": "skill_name", "arguments": {"param1": "value1"}}
                ]
            }

        Raises:
            RuntimeError: API调用失败时抛出（包含详细错误信息）
        """
        try:
            client = self._get_client()

            # 构建请求参数
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            # 如果有工具，添加到请求中
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            # 发送请求到OpenAI API
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # 翻译API返回结果 → 项目统一格式
            result = {
                "content": message.content or "",
                "tool_calls": [],
            }

            # 处理工具调用
            if message.tool_calls:
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    # arguments可能是字符串，需要解析成字典
                    if isinstance(args, str):
                        try:
                            import json
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    result["tool_calls"].append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            return result

        except Exception as e:
            # 捕获所有异常，包装成友好错误信息
            error_msg = f"LLM API调用失败: {str(e)}"

            # 区分不同类型的错误
            if "authentication" in str(e).lower():
                error_msg = "API密钥无效或已过期，请检查.env.test中的LLM_API_KEY配置"
            elif "rate limit" in str(e).lower():
                error_msg = "API配额已用完或触发速率限制，请稍后重试或升级账户"
            elif "connection" in str(e).lower():
                error_msg = "网络连接失败，请检查网络或API地址配置"

            raise RuntimeError(error_msg) from e

    def generate(
            self,
            prompt: str,
            system_message: Optional[str] = None,
    ) -> str:
        """
        简化版对话接口（兼容旧代码）

        Args:
            prompt: 用户输入的问题
            system_message: 可选的系统提示词

        Returns:
            AI的回答文本（纯字符串）
        """
        messages = []

        # 如果有系统提示词，先添加
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # 添加用户问题
        messages.append({"role": "user", "content": prompt})

        # 调用chat方法
        response = self.chat(messages)

        # 只返回content部分
        return response["content"]


class RealEmbeddingClient:
    """
    真实向量客户端 — 翻译官2号

    职责：
    1. 接收API密钥、地址、向量模型名
    2. 提供 .embed() 方法：你的代码给它发一段/几段文字，它去调用真实的向量接口
    3. 把API返回的向量整理成你代码习惯的二维列表格式 [[0.1, 0.2, ...], ...]
    4. 额外提供 .execute() 方法，兼容之前向量检索技能里的调用方式
    5. 提供 .embeddings.create() 方法，兼容 TextEmbedderSkill 的调用方式
    """

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.openai.com/v1",
            model: str = "text-embedding-3-small",
    ):
        """
        初始化真实向量客户端

        Args:
            api_key: OpenAI API密钥（从.env.test读取）
            base_url: API地址（默认OpenAI）
            model: 向量模型名（默认text-embedding-3-small，便宜又快）
        """
        if not api_key:
            raise ValueError("API密钥不能为空，请在.env.test中配置EMBEDDING_API_KEY")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # 懒加载openai库
        self._client = None

        # 兼容 TextEmbedderSkill 的调用方式
        self.embeddings = self  # 让自己成为 embeddings 对象

    def _get_client(self):
        """获取OpenAI客户端实例（单例模式）"""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "使用真实向量模型需要安装openai库:\n  pip install openai"
                )
        return self._client

    def create(self, model: str, input: List[str]):
        """
        兼容 OpenAI API 的 embeddings.create() 方法

        Args:
            model: 模型名称
            input: 文本列表

        Returns:
            模拟 OpenAI API 返回的对象结构
        """
        embeddings = self.embed(input)

        # 构造兼容 OpenAI API 的返回对象
        class EmbeddingData:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class CreateEmbeddingResponse:
            def __init__(self, embeddings_list):
                self.data = [EmbeddingData(i, emb) for i, emb in enumerate(embeddings_list)]

        return CreateEmbeddingResponse(embeddings)

    def embed(
            self,
            texts: List[str],
    ) -> List[List[float]]:
        """
        将文本转换为向量

        Args:
            texts: 待转换的文本列表，如 ["今天天气很好", "我想吃火锅"]

        Returns:
            二维向量列表，如 [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            每个子列表的长度取决于模型（text-embedding-3-small是1536维）

        Raises:
            RuntimeError: API调用失败时抛出
        """
        try:
            client = self._get_client()

            # 调用OpenAI向量接口
            response = client.embeddings.create(
                model=self.model,
                input=texts,
            )

            # 提取向量数据
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)

            return embeddings

        except Exception as e:
            error_msg = f"向量API调用失败: {str(e)}"

            if "authentication" in str(e).lower():
                error_msg = "API密钥无效，请检查.env.test中的EMBEDDING_API_KEY配置"
            elif "rate limit" in str(e).lower():
                error_msg = "API配额已用完，请稍后重试"

            raise RuntimeError(error_msg) from e

    def execute(
            self,
            candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        兼容向量检索技能的调用方式

        Args:
            candidates: 候选列表，每个元素包含 "text" 字段

        Returns:
            添加了 "embedding" 字段的候选列表
        """
        # 提取所有文本
        texts = [c["text"] for c in candidates]

        # 批量转换为向量
        embeddings = self.embed(texts)

        # 将向量注入回candidates
        result = []
        for candidate, embedding in zip(candidates, embeddings):
            enriched = dict(candidate)
            enriched["embedding"] = embedding
            enriched["dimension"] = len(embedding)
            result.append(enriched)

        return result

# ============================================================================
# 兼容性导出 (为了解决 "cannot import name 'generate_code_explanation'" 报错)
# ============================================================================
try:
    from src.skills.preset.technical_development.code_explainer.skill import (
        generate_code_explanation,
    )
except ImportError:
    # 如果导入失败，提供一个占位符，防止整个 real_clients 模块崩溃
    def generate_code_explanation(*args, **kwargs):
        raise RuntimeError(
            "无法导入 generate_code_explanation，请检查 code_explainer/skill.py "
            "是否已正确修复缩进并将该函数移至类外部。"
        )