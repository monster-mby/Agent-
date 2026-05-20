"""
RerankSkill — 基于阿里云 gte-rerank 模型的多路召回精排技能

依赖：
    pip install dashscope tenacity pydantic
"""

from typing import List, Dict, Any, Optional
import logging
import os

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

# 显式导入以便测试 Mock
try:
    import dashscope
    from dashscope import TextRerank
except ImportError:
    dashscope = None
    TextRerank = None

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pydantic Schema
# ──────────────────────────────────────────────

class RerankCandidate(BaseModel):
    """候选文档结构"""
    content: str = Field(..., description="文档内容")
    source: Optional[str] = Field(None, description="来源标识（向量/ES/KG）")
    doc_id: Optional[str] = Field(None, description="文档唯一 ID")
    original_score: Optional[float] = Field(None, description="原始检索分数")


class RerankInput(BaseModel):
    """RerankSkill 输入模型"""
    query: str = Field(..., description="用户查询问题")
    candidates: List[RerankCandidate] = Field(..., description="待精排的候选文档列表")
    top_n: int = Field(default=5, description="返回的精排结果数量")
    min_relevance_score: Optional[float] = Field(
        default=None, description="最低相关性阈值，不传则使用类默认值"
    )
    api_key: Optional[str] = Field(None, description="DashScope API Key，不传则从环境变量获取")


class RerankOutput(BaseModel):
    """RerankSkill 输出模型"""
    reranked_docs: List[Dict[str, Any]] = Field(..., description="精排后的文档列表")
    skipped: bool = Field(default=False, description="是否跳过了精排逻辑")


# ──────────────────────────────────────────────
#  可重试异常判断（tenacity 用）
# ──────────────────────────────────────────────

def _is_retryable(exception: BaseException) -> bool:
    """仅对网络超时、5xx 做重试；4xx / 认证错误不重试"""
    import requests as _requests

    if isinstance(exception, _requests.exceptions.Timeout):
        return True
    if isinstance(exception, _requests.exceptions.ConnectionError):
        return True
    if isinstance(exception, _requests.exceptions.HTTPError):
        status = (
            exception.response.status_code
            if getattr(exception, "response", None)
            else None
        )
        return status is not None and status >= 500
    # dashscope SDK 抛出的异常通常有 status_code 属性
    if hasattr(exception, "status_code"):
        return getattr(exception, "status_code", 500) >= 500
    return False


# ──────────────────────────────────────────────
#  Skill 实现
# ──────────────────────────────────────────────

class RerankSkill(BaseSkill):
    """基于阿里云 gte-rerank 模型对多路召回结果进行相关性精排"""

    name = "rerank_skill"
    description = "基于阿里云 gte-rerank 模型对多路召回结果进行相关性精排"
    version = "0.2.0"
    author = "monster"
    triggers = ["rerank", "精排", "重排序"]

    input_schema = RerankInput
    output_schema = RerankOutput

    # ── 可配置类常量 ──
    DEFAULT_MODEL = "gte-rerank"
    DEFAULT_MIN_RELEVANCE_SCORE = 0.3       # 低于此分的结果默认丢弃
    REQUEST_TIMEOUT = 30                    # 秒
    RETRY_MAX_ATTEMPTS = 3                  # 含首次，共 3 次
    RETRY_MIN_WAIT = 1                      # 指数退避起始等待秒数
    RETRY_MAX_WAIT = 10                     # 指数退避最长等待秒数

    def _check_skip(self, query: str, candidates: List[RerankCandidate], top_n: int) -> bool:
        """前置校验：判断是否跳过 Rerank 调用

        跳过条件（满足任一即跳过）：
        1. 候选数量不超过 top_n —— 返回全量即可，精排无意义
        2. 查询过短（≤3 字符）—— 语义信息不足，rerank 效果差
        """
        if not candidates:
            return True
        if len(candidates) <= top_n:
            logger.info(
                "跳过精排：候选数(%d) ≤ top_n(%d)", len(candidates), top_n
            )
            return True
        if len(query.strip()) <= 3:
            logger.info("跳过精排：查询过短 '%s'", query)
            return True
        return False

    # ── API 调用（带重试 + 超时） ──

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_dashscope_rerank(
        self, query: str, docs: List[str], api_key: str, top_n: int
    ) -> List[Dict[str, Any]]:
        """调用 DashScope gte-rerank API（带重试与超时）

        优先使用 dashscope SDK；若 SDK 不可用则回退到 requests（均享受重试）。
        """
        # ── 路径 A：dashscope SDK ──
        try:
            import dashscope
            from dashscope import TextRerank

            dashscope.api_key = api_key
            resp = TextRerank.call(
                model=self.DEFAULT_MODEL,
                query=query,
                documents=docs,
                top_n=top_n,
                return_documents=False,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope 返回非 200: status={resp.status_code} "
                    f"message={getattr(resp, 'message', '')}"
                )
            # SDK 返回 results 是 list[dict]，每项含 index / relevance_score
            return resp.output.get("results", [])


        except ImportError:
            pass  # SDK 未安装，回退到 requests
        except Exception as e:
            print(f"DEBUG: SDK 调用失败，准备回退。错误信息: {e}")  # <--- 临时加这行
            # 回退到 requests

        # ── 路径 B：requests 回退 ──
        import requests

        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.DEFAULT_MODEL,
            "query": query,
            "documents": docs,
            "return_documents": False,
            "top_n": top_n,
        }

        logger.debug("Rerank API 请求 | query_len=%d | docs=%d", len(query), len(docs))
        response = requests.post(
            url, json=payload, headers=headers, timeout=self.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("output", {}).get("results", [])

    # ── 主流程 ──

    def execute(self, input_data: RerankInput) -> Dict[str, Any]:
        """执行精排逻辑"""
        api_key = input_data.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY：请在入参传入或设置环境变量")

        min_score = (
            input_data.min_relevance_score
            if input_data.min_relevance_score is not None
            else self.DEFAULT_MIN_RELEVANCE_SCORE
        )

        # ── 0. 空候选 guard ──
        if not input_data.candidates:
            logger.info("候选列表为空，直接返回")
            return {"reranked_docs": [], "skipped": True}

        # ── 1. 跳过校验 ──
        if self._check_skip(input_data.query, input_data.candidates, input_data.top_n):
            return {
                "reranked_docs": [
                    c.model_dump() for c in input_data.candidates[: input_data.top_n]
                ],
                "skipped": True,
            }

        # ── 2. API 调用 ──
        docs_text = [c.content for c in input_data.candidates]
        try:
            rerank_results = self._call_dashscope_rerank(
                query=input_data.query,
                docs=docs_text,
                api_key=api_key,
                top_n=input_data.top_n,
            )
        except Exception as exc:
            logger.warning(
                "Rerank API 调用失败，降级为原始顺序 | error=%s | query=%s",
                exc,
                input_data.query[:50],
            )
            return {
                "reranked_docs": [
                    c.model_dump() for c in input_data.candidates[: input_data.top_n]
                ],
                "skipped": True,
            }

        # ── 3. 结果解析与映射 ──
        final_docs = []
        low_score_docs = []  # 低于阈值的备用

        for item in rerank_results:
            index = item["index"]
            score = item["relevance_score"]

            original_doc = input_data.candidates[index]
            doc_dict = original_doc.model_dump()
            doc_dict["rerank_score"] = score

            if score >= min_score:
                final_docs.append(doc_dict)
            else:
                low_score_docs.append(doc_dict)

        logger.debug(
            "Rerank 结果 | 达标=%d | 低于阈值=%d | threshold=%.2f",
            len(final_docs),
            len(low_score_docs),
            min_score,
        )

        # ── 4. 软过滤：若达标结果不足 top_n，用低分结果补齐 ──
        if len(final_docs) < input_data.top_n and low_score_docs:
            shortage = input_data.top_n - len(final_docs)
            final_docs.extend(low_score_docs[:shortage])
            logger.info(
                "软过滤补回 %d 条低分结果以达到 top_n=%d", shortage, input_data.top_n
            )

        # ── 5. 截断 ──
        final_docs = final_docs[: input_data.top_n]

        logger.info(
            "Rerank 完成 | 最终返回=%d | top_n=%d | 原始候选=%d",
            len(final_docs),
            input_data.top_n,
            len(input_data.candidates),
        )

        result = {"reranked_docs": final_docs, "skipped": False}
        return self.validate_output(result)