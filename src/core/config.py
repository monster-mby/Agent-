"""
LLM 配置管理 —— 基于 pydantic-settings，支持 .env、多提供商、自动校验

依赖：pip install pydantic-settings

get_llm_config()（全局单例获取）
  ├→ LLMConfig 初始化
  │   ├→ pydantic-settings 从 .env / 环境变量加载配置
  │   ├→ _validate_provider（校验提供商）
  │   └→ _apply_fallbacks（应用回退机制，补全 api_key/base_url）
  ├→ 输出加载完成的日志（脱敏）
  └→ 返回 LLMConfig 单例

LLMConfig 使用
  ├→ litellm_model（获取 LiteLLM 格式模型）
  ├→ api_key_masked（获取脱敏 API 密钥）
  ├→ model_dump_safe（安全序列化）
  └→ __repr__ / __str__（字符串表示）
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar

from pydantic import (
    AnyHttpUrl,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 已知提供商的默认 base_url
# ═══════════════════════════════════════════════════════════════

_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai":     "https://api.openai.com/v1",
    "deepseek":   "https://api.deepseek.com/v1",
    "doubao":     "https://ark.cn-beijing.volces.com/api/v3",
    "anthropic":  "https://api.anthropic.com",
    "ollama":     "http://localhost:11434/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "together":   "https://api.together.xyz/v1",
}

# ═══════════════════════════════════════════════════════════════
# 配置模型
# ═══════════════════════════════════════════════════════════════

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "dev"
    db_path: str = "data/checkpoints.sqlite"
    api_keys: list[str] = []   # pydantic 自动处理逗号分隔的字符串
    cors_allow_origins: list[str] = ["*"]  # ✅ 新增：CORS 配置

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # ✅ 忽略 .env 中未定义的字段，不报错
        # 若自动分隔不生效，显式打开
        # env_parse_comma_separated_list = True

settings = Settings()   # 模块级，全局可用

class LLMConfig(BaseSettings):
    """统一的 LLM 配置，自动从 .env / 环境变量加载。

    环境变量命名规则：
      - 通用变量：LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, ...
      - 提供商专属回退：DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ...

    优先级：通用 LLM_* 变量 > 提供商专属变量 > 默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_",
        extra="ignore",          # 忽略未知环境变量，不报错
        case_sensitive=False,
    )

    # ── 基础字段 ──
    provider: str = Field(
        default="openai",
        description="LLM 提供商：openai / deepseek / doubao / anthropic / ollama / groq",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="模型名称",
    )
    api_key: str | None = Field(
        default=None,
        description="API 密钥。若不设，自动尝试 {PROVIDER}_API_KEY 环境变量",
    )
    base_url: str | None = Field(
        default=None,
        description="API Base URL。若不设，自动使用已知提供商的默认值，或尝试 {PROVIDER}_BASE_URL",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="采样温度，0.0-2.0",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=131072,
        description="最大输出 token 数",
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="请求超时时间（秒）",
    )

    # ── 可选扩展字段 ──
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="额外的 HTTP 请求头，用于自部署模型等场景",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="核采样参数",
    )

    # ═══════════════════════════════════════════════════════════
    # 验证器
    # ═══════════════════════════════════════════════════════════

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        """校验 provider 是否在已知列表中（不强制，仅警告）"""
        v_lower = v.lower()
        if v_lower not in _DEFAULT_BASE_URLS:
            logger.warning(
                "未知 LLM provider '%s'，将不自动补全 base_url。"
                "已知提供商：%s",
                v, list(_DEFAULT_BASE_URLS.keys()),
            )
        return v_lower

    @model_validator(mode="after")
    def _apply_fallbacks(self) -> "LLMConfig":
        """应用回退机制：通用变量为空时 → 提供商专属变量 → 默认映射"""
        import os

        provider_upper = self.provider.upper()

        # api_key 回退
        if not self.api_key:
            self.api_key = os.getenv(f"{provider_upper}_API_KEY")

        # base_url 回退
        if not self.base_url:
            self.base_url = os.getenv(f"{provider_upper}_BASE_URL") or _DEFAULT_BASE_URLS.get(self.provider)

        # 对于需要 api_key 的云端提供商，做非空校验（ollama 等本地模型除外）
        if self.provider not in ("ollama",) and not self.api_key:
            logger.warning(
                "LLM provider '%s' 的 api_key 未设置。"
                "请设置 LLM_API_KEY 或 %s_API_KEY 环境变量。",
                self.provider, provider_upper,
            )

        return self

    # ═══════════════════════════════════════════════════════════
    # 计算属性
    # ═══════════════════════════════════════════════════════════

    @property
    def litellm_model(self) -> str:
        """LiteLLM 格式：provider/model_name"""
        if "/" in self.model:
            return self.model
        return f"{self.provider}/{self.model}"

    @property
    def api_key_masked(self) -> str:
        """脱敏后的 api_key，用于日志输出"""
        if not self.api_key:
            return "<not set>"
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return self.api_key[:4] + "****" + self.api_key[-4:]

    # ═══════════════════════════════════════════════════════════
    # 序列化控制
    # ═══════════════════════════════════════════════════════════

    def model_dump_safe(self, **kwargs) -> dict:
        """安全序列化，api_key 自动脱敏"""
        data = self.model_dump(**kwargs)
        if "api_key" in data and data["api_key"]:
            data["api_key"] = self.api_key_masked
        return data

    def __repr__(self) -> str:
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={self.api_key_masked!r}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"timeout={self.timeout})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """获取全局 LLM 配置单例（带缓存）"""
    config = LLMConfig()  # type: ignore[call-arg]  # pydantic-settings 自动从 env 加载
    logger.info("LLM 配置加载完成：%s", config)
    return config



# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = LLMConfig()
    print("完整配置（脱敏）:", cfg.model_dump_safe())
    print("litellm_model:", cfg.litellm_model)
    print()

    # 模拟手动构造也能触发验证
    cfg2 = LLMConfig(provider="deepseek", model="deepseek-chat")
    print("手动构造:", cfg2)