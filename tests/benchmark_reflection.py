
#!/usr/bin/env python3
"""
A/B 对比基准：反思模式效果验证

用法：
    # 全量（慢，60 用例）
    python tests/benchmark_reflection.py

    # 快速模式：只测 RAG，只测无反思（~30s，10 用例）
    QUICK_MODE=1 QUICK_ONLY=rag SKIP_REFLECTION=1 python tests/benchmark_reflection.py

    # 快速模式：只测代码审查 + 1 轮反思
    QUICK_MODE=1 QUICK_ONLY=code_review python tests/benchmark_reflection.py

环境变量：
    QUICK_MODE=1          启用快速模式（默认 0）
    QUICK_ONLY=rag        只测 rag 或 code_review（默认 all）
    SKIP_REFLECTION=1     跳过反思配置，只测无反思（默认 0）
    MAX_WORKERS=4         并行线程数（默认 4）
    BENCHMARK_MAX_ITERS=1 反思最大迭代次数（默认 2）

输出：
    - 控制台打印对比报告
    - tests/benchmark_results.json（原始数据）
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ═══════════════════════════════════════════════════════════════
# 🔧 快速模式开关（也支持环境变量覆盖）
# ═══════════════════════════════════════════════════════════════
_QUICK_MODE = os.getenv("QUICK_MODE", "0") == "1"
_QUICK_ONLY = os.getenv("QUICK_ONLY", "all")          # "rag" | "code_review" | "all"
_SKIP_REFLECTION = os.getenv("SKIP_REFLECTION", "0") == "1"
_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
_BENCHMARK_MAX_ITERS = int(os.getenv("BENCHMARK_MAX_ITERS", "2"))


# ═══════════════════════════════════════════════════════════════
# 测试用例（原样保留）
# ═══════════════════════════════════════════════════════════════

RAG_TEST_CASES = [
    {
        "id": "rag_01", "query": "Python 是什么时候由谁创建的？",
        "search_results": [
            {"chunk_id": "c1", "text": "Python 是由 Guido van Rossum 于 1991 年首次发布的编程语言。", "score": 0.95, "metadata": {"title": "Python历史"}},
            {"chunk_id": "c2", "text": "Python 是一种解释型、面向对象的高级编程语言，以简洁语法著称。", "score": 0.88, "metadata": {"title": "Python概述"}},
        ],
        "expected_points": ["Guido van Rossum", "1991年", "首次发布"],
    },
    {
        "id": "rag_02", "query": "Python 的主要特点有哪些？",
        "search_results": [
            {"chunk_id": "c1", "text": "Python 的主要特点包括：动态类型系统、自动内存管理（垃圾回收）、丰富的标准库。", "score": 0.92, "metadata": {"title": "Python特点"}},
            {"chunk_id": "c2", "text": "Python 支持多种编程范式：面向对象、函数式、过程式编程。", "score": 0.85, "metadata": {"title": "Python范式"}},
        ],
        "expected_points": ["动态类型", "自动内存管理", "垃圾回收", "丰富标准库", "多范式"],
    },
    {
        "id": "rag_03", "query": "PyTorch 和 TensorFlow 的区别是什么？",
        "search_results": [
            {"chunk_id": "c1", "text": "PyTorch 使用动态计算图，调试方便，适合研究和快速原型开发。由 Meta 开发维护。", "score": 0.91, "metadata": {"title": "PyTorch"}},
            {"chunk_id": "c2", "text": "TensorFlow 使用静态计算图（2.0 后支持 Eager Execution），适合生产部署。由 Google 开发维护。", "score": 0.89, "metadata": {"title": "TensorFlow"}},
        ],
        "expected_points": ["动态计算图", "静态计算图", "Meta", "Google", "调试", "生产部署"],
    },
    {
        "id": "rag_04", "query": "什么是 Docker，它解决了什么问题？",
        "search_results": [
            {"chunk_id": "c1", "text": "Docker 是一个容器化平台，将应用及其依赖打包到轻量级容器中，确保环境一致性。", "score": 0.94, "metadata": {"title": "Docker概述"}},
            {"chunk_id": "c2", "text": "Docker 解决了“在我机器上能跑”的问题——开发、测试、生产环境保持一致。", "score": 0.87, "metadata": {"title": "Docker价值"}},
        ],
        "expected_points": ["容器化", "环境一致性", "轻量级", "依赖打包"],
    },
    {
        "id": "rag_05", "query": "REST API 和 GraphQL 各有什么优缺点？",
        "search_results": [
            {"chunk_id": "c1", "text": "REST API 使用标准 HTTP 方法（GET/POST/PUT/DELETE），接口简单，缓存友好，但可能存在过度获取（over-fetching）问题。", "score": 0.90, "metadata": {"title": "REST"}},
            {"chunk_id": "c2", "text": "GraphQL 允许客户端精确指定所需字段，避免过度获取，但学习曲线较陡，缓存实现更复杂。", "score": 0.88, "metadata": {"title": "GraphQL"}},
        ],
        "expected_points": ["REST缓存友好", "过度获取", "GraphQL精确查询", "缓存复杂"],
    },
    {
        "id": "rag_06", "query": "Git 中 merge 和 rebase 的区别？",
        "search_results": [
            {"chunk_id": "c1", "text": "git merge 创建一个新的合并提交（merge commit），保留完整的分支历史。适合公共分支。", "score": 0.93, "metadata": {"title": "Merge"}},
            {"chunk_id": "c2", "text": "git rebase 将当前分支的提交重新应用到目标分支顶部，产生线性历史。适合个人分支整理。", "score": 0.91, "metadata": {"title": "Rebase"}},
        ],
        "expected_points": ["merge保留历史", "合并提交", "rebase线性历史", "改写历史"],
    },
    {
        "id": "rag_07", "query": "HTTPS 是如何保证安全性的？",
        "search_results": [
            {"chunk_id": "c1", "text": "HTTPS 通过 TLS/SSL 协议加密通信，使用非对称加密交换密钥，对称加密传输数据。", "score": 0.95, "metadata": {"title": "HTTPS原理"}},
            {"chunk_id": "c2", "text": "HTTPS 还通过数字证书验证服务器身份，防止中间人攻击（MITM）。", "score": 0.89, "metadata": {"title": "HTTPS证书"}},
        ],
        "expected_points": ["TLS", "SSL", "非对称加密", "对称加密", "数字证书", "中间人攻击"],
    },
    {
        "id": "rag_08", "query": "SQL 注入攻击的原理和防护方法？",
        "search_results": [
            {"chunk_id": "c1", "text": "SQL 注入攻击通过将恶意 SQL 代码插入输入字段，欺骗数据库执行非预期查询。", "score": 0.94, "metadata": {"title": "SQL注入原理"}},
            {"chunk_id": "c2", "text": "防护方法包括：参数化查询（Prepared Statement）、输入验证、ORM 框架、最小权限原则。", "score": 0.90, "metadata": {"title": "SQL注入防护"}},
        ],
        "expected_points": ["恶意SQL", "参数化查询", "输入验证", "ORM", "最小权限"],
    },
    {
        "id": "rag_09", "query": "微服务架构的优缺点？",
        "search_results": [
            {"chunk_id": "c1", "text": "微服务优势：独立部署、技术栈灵活、团队自治、可伸缩性强。", "score": 0.92, "metadata": {"title": "微服务优势"}},
            {"chunk_id": "c2", "text": "微服务挑战：分布式系统复杂性、网络延迟、数据一致性、运维成本高。", "score": 0.88, "metadata": {"title": "微服务挑战"}},
        ],
        "expected_points": ["独立部署", "技术栈灵活", "分布式复杂性", "数据一致性", "运维成本"],
    },
    {
        "id": "rag_10", "query": "什么是 CI/CD？",
        "search_results": [
            {"chunk_id": "c1", "text": "CI（持续集成）指代码频繁合并到主干，自动构建和测试，快速发现集成问题。", "score": 0.93, "metadata": {"title": "CI"}},
            {"chunk_id": "c2", "text": "CD（持续交付/部署）在 CI 基础上自动将通过测试的代码部署到生产环境。", "score": 0.90, "metadata": {"title": "CD"}},
        ],
        "expected_points": ["持续集成", "频繁合并", "自动构建测试", "持续交付", "自动部署"],
    },
]

CODE_REVIEW_TEST_CASES = [
    {"id": "cr_01", "language": "python", "code": "def process_data(data):\n    try:\n        result = data['value'] / data['count']\n    except:\n        pass\n    return result", "expected_issues": ["bare_except", "division_by_zero", "key_error"]},
    {"id": "cr_02", "language": "python", "code": "def read_file(path):\n    f = open(path, 'r')\n    content = f.read()\n    f.close()\n    return content", "expected_issues": ["no_with_statement", "resource_leak", "exception_on_failure"]},
    {"id": "cr_03", "language": "python", "code": "def fetch_user(user_id):\n    query = \"SELECT * FROM users WHERE id = \" + user_id\n    return database.execute(query)", "expected_issues": ["sql_injection", "string_concatenation"]},
    {"id": "cr_04", "language": "python", "code": "def add_item(lst=[]):\n    lst.append(1)\n    return lst", "expected_issues": ["mutable_default_arg"]},
    {"id": "cr_05", "language": "python", "code": "def authenticate(password):\n    if password == \"admin123\":\n        return True\n    return False", "expected_issues": ["hardcoded_password", "plain_text"]},
    {"id": "cr_06", "language": "python", "code": "import os\ndef delete_user(user_id):\n    os.system(f\"rm -rf /data/users/{user_id}\")", "expected_issues": ["os_system", "command_injection"]},
    {"id": "cr_07", "language": "python", "code": "def calculate_average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        total += numbers[i]\n    return total / len(numbers)", "expected_issues": ["non_pythonic_loop", "zero_division"]},
    {"id": "cr_08", "language": "python", "code": "import hashlib\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()", "expected_issues": ["md5_insecure", "no_salt"]},
    {"id": "cr_09", "language": "python", "code": "def nested_loop(data):\n    result = []\n    for i in range(len(data)):\n        for j in range(len(data)):\n            if data[i] * data[j] > 100:\n                result.append((i, j))\n    return result", "expected_issues": ["nested_loop", "performance", "on2"]},
    {"id": "cr_10", "language": "python", "code": "class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n\ndef save_user(user):\n    # TODO: implement database save\n    pass", "expected_issues": ["TODO", "incomplete", "no_validation"]},
]


# ═══════════════════════════════════════════════════════════════
# LLM 评判器（原样保留）
# ═══════════════════════════════════════════════════════════════

class LLMJudge:
    """用 LLM 评判输出质量，失败时回退启发式评分"""

    def __init__(self, llm_client: Any = None, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        if self.llm is None:
            return None
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是评分员，只输出 JSON。不要有任何解释。"},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=300,
            )
            raw = response["content"] if isinstance(response, dict) else (
                response.content if hasattr(response, "content") else str(response)
            )
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            logger.warning("LLM 评判失败: %s，回退启发式", e)
        return None

    def score_rag_answer(self, question, answer, expected_points, search_results):
        prompt = f"""为以下 RAG 回答评分（0-10），基于这些标准：
- 准确性（4分）：事实正确性
- 完整性（3分）：覆盖关键点：{expected_points}
- 引用质量（2分）：是否基于参考资料
- 语言质量（1分）：清晰流畅

## 用户问题
{question}

## 参考资料
{chr(10).join(f'[{r["chunk_id"]}] {r["text"][:200]}' for r in search_results[:3])}

## 待评分答案
{answer[:2000]}

输出 JSON 格式：
{{"score": 8.5, "accuracy": 3.5, "completeness": 2.5, "citation_quality": 1.5, "language_quality": 1.0, "comment": "简短评价"}}
"""
        result = self._call_llm(prompt)
        if result:
            return result
        return self._score_rag_heuristic(answer, expected_points)

    def score_code_review(self, code, review, expected_issues):
        prompt = f"""为以下代码审查评分（0-10），基于这些标准：
- 问题发现（5分）：是否发现了预期问题：{expected_issues}
- 分析深度（3分）：解释是否深入
- 建议质量（2分）：修复建议是否具体

## 待审查代码
```python
{code[:1500]}
```

## 审查结果
{review[:2000]}

输出 JSON 格式：
{{"score": 8.0, "issue_discovery": 4.0, "analysis_depth": 2.5, "suggestion_quality": 1.5, "comment": "简短评价"}}
"""
        result = self._call_llm(prompt)
        if result:
            return result
        return self._score_code_heuristic(review, expected_issues)

    @staticmethod
    def _score_rag_heuristic(answer, expected_points):
        if not answer or len(answer) < 20:
            return {"score": 0, "comment": "空答案或过短", "method": "heuristic"}
        hits = sum(1 for p in expected_points if p.lower() in answer.lower())
        score = min(10, round(hits / len(expected_points) * 8 + 2, 1))
        return {
            "score": score, "accuracy": min(4, score * 0.4),
            "completeness": round(hits / len(expected_points) * 3, 1),
            "citation_quality": 1.0, "language_quality": 1.0,
            "comment": f"启发式：{hits}/{len(expected_points)} 要点命中", "method": "heuristic",
        }

    @staticmethod
    def _score_code_heuristic(review, expected_issues):
        if not review or len(review) < 30:
            return {"score": 0, "comment": "空审查或过短", "method": "heuristic"}
        hits = sum(1 for p in expected_issues if p.lower().replace("_", " ") in review.lower())
        score = min(10, round(hits / len(expected_issues) * 8 + 2, 1))
        return {
            "score": score, "issue_discovery": round(hits / len(expected_issues) * 5, 1),
            "analysis_depth": 1.5, "suggestion_quality": 1.0,
            "comment": f"启发式：{hits}/{len(expected_issues)} 预期问题", "method": "heuristic",
        }


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    name: str
    enable_reflection: bool
    max_iterations: int  # 0=无，1=1轮，2=2轮


@dataclass
class SingleResult:
    test_id: str
    config_name: str
    score: float
    success: bool
    elapsed_ms: float
    token_estimate: int
    output: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    config_name: str
    total_tests: int
    success_count: int
    success_rate: float
    avg_score: float
    median_score: float
    avg_elapsed_ms: float
    avg_tokens: int
    all_scores: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Token 估算
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return int(chinese * 1.5 + english_words * 1.3)


# ═══════════════════════════════════════════════════════════════
# 基准运行器（核心改动：并行 + 快速模式 + 图编译分离）
# ═══════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """三种配置 × 用户可选用例数"""

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4",
        output_file: str = None,
        quick_only: str = "all",
        max_workers: int = 4,
        benchmark_max_iters: int = 2,
    ):
        self.llm_client = llm_client
        self.model = model
        self.output_file = output_file or str(_PROJECT_ROOT / "tests" / "benchmark_results.json")
        self.judge = LLMJudge(llm_client, model)
        self.results: List[SingleResult] = []
        self._skill_cache: Dict[str, Any] = {}  # ← 图/技能编译缓存
        self._rag_classes: Optional[tuple] = None
        self._code_classes: Optional[tuple] = None

        # 🔧 快速模式配置
        self.quick_only = quick_only                # "rag" | "code_review" | "all"
        self.max_workers = max_workers
        self.benchmark_max_iters = benchmark_max_iters

    # ── 延迟导入 + 缓存 ──

    def _get_rag_classes(self):
        if self._rag_classes is None:
            try:
                from src.skills.custom.rag_skills.rag_answer.skill import (
                    RagAnswerSkill, RagAnswerInput, SearchResultRef,
                )
                self._rag_classes = (RagAnswerSkill, RagAnswerInput, SearchResultRef)
            except ImportError:
                from src.skills.custom.rag_skills.rag_answer import RagAnswerSkill as _RAS
                self._rag_classes = (_RAS, _RAS.input_schema, None)
        return self._rag_classes

    def _get_code_classes(self):
        if self._code_classes is None:
            try:
                from src.skills.preset.technical_development.code_explainer.skill import (
                    CodeExplainerSkill, CodeExplainerInput,
                )
                self._code_classes = (CodeExplainerSkill, CodeExplainerInput)
            except ImportError:
                from src.skills.preset.technical_development.code_explainer import (
                    CodeExplainerSkill as _CES,
                )
                self._code_classes = (_CES, _CES.input_schema)
        return self._code_classes

    # ── RAG 单条（抽离为独立方法，方便并行调用）──

    def run_rag_single(self, tc: dict, cfg: BenchmarkConfig) -> SingleResult:
        RagAnswerSkill, RagAnswerInput, SearchResultRef = self._get_rag_classes()

        kwargs = {"query": tc["query"], "search_results": tc["search_results"]}
        if SearchResultRef:
            kwargs["search_results"] = [SearchResultRef(**r) for r in tc["search_results"]]

        inp = RagAnswerInput(
            **kwargs,
            enable_reflection=cfg.enable_reflection,
            llm_client=self.llm_client if cfg.enable_reflection else None,
            model=self.llm_client.model if self.llm_client and cfg.enable_reflection else None,
        )

        skill = RagAnswerSkill()
        t0 = time.perf_counter()
        try:
            result = skill.execute(inp)
        except Exception as exc:
            logger.error("RAG 执行失败 [%s]: %s", tc["id"], exc)
            return SingleResult(
                test_id=tc["id"], config_name=cfg.name,
                score=0, success=False, elapsed_ms=0,
                token_estimate=0,
                detail={"type": "rag", "error": str(exc),
                        "reflection_enabled": cfg.enable_reflection},
            )
        elapsed = (time.perf_counter() - t0) * 1000

        answer = result.get("answer", "") if isinstance(result, dict) else getattr(result, "answer", "")

        # 🔧 检测反思回退（WARNING 来源）
        reflection_fallback = False
        if isinstance(result, dict):
            reflection_fallback = result.get("reflection_fallback", False)
        elif hasattr(result, "reflection_fallback"):
            reflection_fallback = result.reflection_fallback

        tokens = estimate_tokens(answer) + estimate_tokens(tc["query"])
        tokens += 500 if cfg.enable_reflection else 0

        judgement = self.judge.score_rag_answer(
            question=tc["query"], answer=answer,
            expected_points=tc["expected_points"],
            search_results=tc["search_results"],
        )
        score = judgement.get("score", 0)

        return SingleResult(
            test_id=tc["id"], config_name=cfg.name,
            score=score, success=score >= 6,
            elapsed_ms=elapsed, token_estimate=int(tokens),
            output=answer[:500],
            detail={
                "type": "rag", "judgement": judgement,
                "reflection_fallback": reflection_fallback,
            },
        )

    # ── 代码审查单条 ──

    def run_code_review_single(self, tc: dict, cfg: BenchmarkConfig) -> SingleResult:
        CodeExplainerSkill, CodeExplainerInput = self._get_code_classes()

        inp = CodeExplainerInput(
            code=tc["code"],
            language=tc.get("language", "python"),
            detail_level="detailed",
            enable_reflection=cfg.enable_reflection,
            llm_client=self.llm_client if cfg.enable_reflection else None,
            model=self.llm_client.model if self.llm_client and cfg.enable_reflection else None,
        )

        skill = CodeExplainerSkill()
        t0 = time.perf_counter()
        try:
            result = skill.execute(inp)
        except Exception as exc:
            logger.error("代码审查执行失败 [%s]: %s", tc["id"], exc)
            return SingleResult(
                test_id=tc["id"], config_name=cfg.name,
                score=0, success=False, elapsed_ms=0,
                token_estimate=0,
                detail={"type": "code_review", "error": str(exc),
                        "reflection_enabled": cfg.enable_reflection},
            )
        elapsed = (time.perf_counter() - t0) * 1000

        overview = getattr(result, "overview", "")
        issues = getattr(result, "potential_issues", [])
        review_text = overview + "\n" + "\n".join(issues)

        # 🔧 检测反思回退
        reflection_fallback = getattr(result, "reflection_fallback", False)

        tokens = estimate_tokens(review_text) + estimate_tokens(tc["code"])
        tokens += 500 if cfg.enable_reflection else 0

        judgement = self.judge.score_code_review(
            code=tc["code"], review=review_text,
            expected_issues=tc["expected_issues"],
        )
        score = judgement.get("score", 0)

        return SingleResult(
            test_id=tc["id"], config_name=cfg.name,
            score=score, success=score >= 6,
            elapsed_ms=elapsed, token_estimate=int(tokens),
            output=review_text[:500],
            detail={
                "type": "code_review", "judgement": judgement,
                "reflection_fallback": reflection_fallback,
            },
        )

    # ── 并行运行同配置的所有用例 ──

    def _run_config_parallel(self, cfg: BenchmarkConfig, test_cases: list):
        """并行执行同一配置下的全部用例"""
        rag_cases = [tc for tc in test_cases if "search_results" in tc]
        code_cases = [tc for tc in test_cases if "search_results" not in tc]

        futures_map = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for tc in rag_cases:
                f = executor.submit(self.run_rag_single, tc, cfg)
                futures_map[f] = tc["id"]
            for tc in code_cases:
                f = executor.submit(self.run_code_review_single, tc, cfg)
                futures_map[f] = tc["id"]

            completed = 0
            for future in as_completed(futures_map):
                completed += 1
                test_id = futures_map[future]
                try:
                    r = future.result(timeout=120)  # 单个用例 2 分钟超时
                except Exception as exc:
                    logger.error("并行任务超时/崩溃 [%s]: %s", test_id, exc)
                    r = SingleResult(
                        test_id=test_id, config_name=cfg.name,
                        score=0, success=False, elapsed_ms=0,
                        token_estimate=0,
                        detail={"error": str(exc), "type": "timeout"},
                    )
                self.results.append(r)
                logger.info(
                    "[%d/%d 完成] %s | 分=%.1f | %s | 耗时=%.0fms | Tokens≈%d",
                    completed, len(futures_map), test_id,
                    r.score, "✅" if r.success else "❌",
                    r.elapsed_ms, r.token_estimate,
                )
                # 🔧 反思回退告警
                if r.detail.get("reflection_fallback"):
                    logger.warning(
                        "  ⚠️  [%s] 反思管道回退——skill 内部捕获异常，返回原始答案。"
                        "请检查 ReflectionContext 注册 / CriticNode / _run_*_reflection 方法。",
                        test_id,
                    )

    # ── 获取本次要跑的测试集 ──

    def _get_test_cases(self):
        if self.quick_only == "rag":
            return list(RAG_TEST_CASES)
        elif self.quick_only == "code_review":
            return list(CODE_REVIEW_TEST_CASES)
        else:
            return list(RAG_TEST_CASES) + list(CODE_REVIEW_TEST_CASES)

    # ── 全量运行 ──

    def run_all(self):
        # 🔧 快速模式：是否跳过反思配置
        if _SKIP_REFLECTION:
            configs = [
                BenchmarkConfig("无反思", enable_reflection=False, max_iterations=0),
            ]
        else:
            configs = [
                BenchmarkConfig("无反思", enable_reflection=False, max_iterations=0),
                BenchmarkConfig(
                    "1轮反思", enable_reflection=True,
                    max_iterations=min(1, self.benchmark_max_iters),
                ),
                BenchmarkConfig(
                    "2轮反思", enable_reflection=True,
                    max_iterations=min(2, self.benchmark_max_iters),
                ),
            ]

        test_cases = self._get_test_cases()
        total = sum(1 for _ in configs) * len(test_cases)

        # 预热：导入所有需要用到的类（防止并行时重复导入竞态）
        if any(tc.get("search_results") for tc in test_cases):
            self._get_rag_classes()
        if any("search_results" not in tc for tc in test_cases):
            self._get_code_classes()

        logger.info("=" * 60)
        logger.info("🚀 基准测试开始")
        logger.info("   模式: %s | 跳过反思: %s | 并行: %d workers",
                     self.quick_only, _SKIP_REFLECTION, self.max_workers)
        logger.info("   配置数: %d | 用例数: %d | 合计: %d",
                     len(configs), len(test_cases), total)
        logger.info("=" * 60)

        for cfg in configs:
            logger.info("─" * 50)
            logger.info("配置: %s (reflection=%s, max_iters=%d)",
                        cfg.name, cfg.enable_reflection, cfg.max_iterations)
            logger.info("─" * 50)

            t_cfg_start = time.perf_counter()
            self._run_config_parallel(cfg, test_cases)
            t_cfg_elapsed = (time.perf_counter() - t_cfg_start)
            logger.info("配置 %s 完成，耗时 %.1fs", cfg.name, t_cfg_elapsed)

        self._save()
        self._report()

    # ── 报告（增强：回退统计）──

    def _build_report(self, config_name: str) -> BenchmarkReport:
        matching = [r for r in self.results if r.config_name == config_name]
        if not matching:
            return BenchmarkReport(config_name=config_name, total_tests=0,
                                   success_count=0, success_rate=0,
                                   avg_score=0, median_score=0,
                                   avg_elapsed_ms=0, avg_tokens=0)
        scores = [r.score for r in matching]
        return BenchmarkReport(
            config_name=config_name,
            total_tests=len(matching),
            success_count=sum(1 for r in matching if r.success),
            success_rate=sum(1 for r in matching if r.success) / len(matching) * 100,
            avg_score=statistics.mean(scores),
            median_score=statistics.median(scores),
            avg_elapsed_ms=statistics.mean([r.elapsed_ms for r in matching]),
            avg_tokens=int(statistics.mean([r.token_estimate for r in matching])),
            all_scores=scores,
        )

    def _report(self):
        configs = ["无反思", "1轮反思", "2轮反思"]
        reports = {c: self._build_report(c) for c in configs}

        print("\n" + "=" * 80)
        print("  A/B 对比基准报告 — 反思模式效果验证")
        print("=" * 80)
        print(f"  时间: {datetime.now().isoformat(timespec='seconds')}")
        print(f"  总运行: {len(self.results)} 次")
        if _SKIP_REFLECTION:
            print(f"  ⚠️  已跳过反思配置（SKIP_REFLECTION=1）")
        if self.quick_only != "all":
            print(f"  ⚠️  快速模式：只测 {self.quick_only}")

        header = f"  {'指标':<18} | {'无反思':<10} | {'1轮反思':<10} | {'2轮反思':<10} | {'1轮vs无':<12} | {'2轮vs1':<12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        r0, r1, r2 = reports["无反思"], reports["1轮反思"], reports["2轮反思"]

        def row(label, v0, v1, v2, fmt=".1f", unit=""):
            d1 = v1 - v0 if isinstance(v0, (int, float)) else 0
            d2 = v2 - v1 if isinstance(v1, (int, float)) else 0
            a1 = "↑" if d1 > 0 else ("↓" if d1 < 0 else "→")
            a2 = "↑" if d2 > 0 else ("↓" if d2 < 0 else "→")
            print(f"  {label:<18} | {v0:{fmt}}{unit:<6} | {v1:{fmt}}{unit:<6} | "
                  f"{v2:{fmt}}{unit:<6} | {a1} {abs(d1):{fmt}}{unit:<8} | {a2} {abs(d2):{fmt}}{unit}")

        row("成功率 (%)", r0.success_rate, r1.success_rate, r2.success_rate)
        row("平均分", r0.avg_score, r1.avg_score, r2.avg_score, fmt=".2f")
        row("中位分", r0.median_score, r1.median_score, r2.median_score, fmt=".2f")
        row("平均耗时 (ms)", r0.avg_elapsed_ms, r1.avg_elapsed_ms, r2.avg_elapsed_ms, fmt=".0f")
        row("平均 Token", r0.avg_tokens, r1.avg_tokens, r2.avg_tokens, fmt="d")

        # 🔧 反思回退统计
        print()
        print("  ── 反思回退告警 ──")
        for cfg_name in ["1轮反思", "2轮反思"]:
            fallback_count = sum(
                1 for r in self.results
                if r.config_name == cfg_name and r.detail.get("reflection_fallback")
            )
            total = len([r for r in self.results if r.config_name == cfg_name])
            pct = fallback_count / total * 100 if total else 0
            icon = "❌" if pct > 0 else "✅"
            print(f"  {icon} {cfg_name}: {fallback_count}/{total} ({pct:.0f}%) 回退到原始答案")

        print()
        print("  ── 关键指标验证 ──")
        imp1 = r1.success_rate - r0.success_rate
        imp2 = r2.success_rate - r1.success_rate
        print(f"  {'✅' if imp1 >= 20 else '❌'} 1轮反思提升 {imp1:+.1f}% {'>=20% 目标' if imp1 >= 20 else '<20% 未达标'}")
        print(f"  {'✅' if imp2 <= 5 else '⚠️ '} 2轮vs1轮 {imp2:+.1f}% {'≤5% → 1轮已够' if imp2 <= 5 else '>5% → 2轮仍有提升'}")

        # 分类明细
        print("\n  ── 分项明细 ──")
        for task_type, label in [("rag", "RAG 问答"), ("code_review", "代码审查")]:
            print(f"\n  {label}:")
            for c in configs:
                sub = [r for r in self.results if r.config_name == c and r.detail.get("type") == task_type]
                if sub:
                    avg_s = statistics.mean([r.score for r in sub])
                    succ = sum(1 for r in sub if r.success)
                    avg_e = statistics.mean([r.elapsed_ms for r in sub])
                    fallback_n = sum(1 for r in sub if r.detail.get("reflection_fallback"))
                    extra = f" | 回退={fallback_n}" if fallback_n else ""
                    print(f"    {c}: 分={avg_s:.2f} | 成功率={succ}/{len(sub)} ({succ/len(sub)*100:.0f}%) | 耗时={avg_e:.0f}ms{extra}")

        print("\n" + "=" * 80)

    def _save(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "configs": ["无反思", "1轮反思", "2轮反思"],
            "quick_mode": {
                "enabled": _QUICK_MODE or self.quick_only != "all",
                "quick_only": self.quick_only,
                "skip_reflection": _SKIP_REFLECTION,
                "max_workers": self.max_workers,
            },
            "results": [
                {
                    "test_id": r.test_id,
                    "config": r.config_name,
                    "score": r.score,
                    "success": r.success,
                    "elapsed_ms": r.elapsed_ms,
                    "token_estimate": r.token_estimate,
                    "output_preview": r.output[:300],
                    "reflection_fallback": r.detail.get("reflection_fallback", False),
                    "detail": r.detail.get("judgement", {}),
                }
                for r in self.results
            ],
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("原始数据已保存: %s", self.output_file)


# ═══════════════════════════════════════════════════════════════
# LLM 客户端创建
# ═══════════════════════════════════════════════════════════════

def create_llm_client() -> Any:
    from dotenv import load_dotenv
    env_test_path = Path(__file__).parent.parent / ".env.test"
    if env_test_path.exists():
        load_dotenv(env_test_path)
        logger.info("已加载 .env.test: %s", env_test_path)

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        logger.warning("未配置 LLM_API_KEY → 使用启发式评分（准确性有限）")
        logger.warning("要启用 LLM 评判，请在 .env.test 中配置 LLM_API_KEY")
        return None

    try:
        from tests.real_clients import RealLLMClient
        client = RealLLMClient(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
        logger.info("✅ LLM 客户端已配置: model=%s, base_url=%s", client.model, client.base_url)
        return client
    except ImportError as e:
        logger.error("❌ 无法导入 RealLLMClient: %s", e)
        return None
    except Exception as e:
        import traceback
        logger.error("❌ 创建 LLM 客户端失败: %s", e)
        logger.error("详细堆栈:\n%s", traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("A/B 对比基准开始")

    # 🔧 打印当前模式
    if _QUICK_MODE:
        logger.info("⚡ 快速模式: quick_only=%s, skip_reflection=%s, max_workers=%d",
                     _QUICK_ONLY, _SKIP_REFLECTION, _MAX_WORKERS)
    else:
        logger.info("🐢 全量模式（可用 QUICK_MODE=1 加速）")

    llm = create_llm_client()
    runner = BenchmarkRunner(
        llm_client=llm,
        model=os.getenv("BENCHMARK_MODEL", "gpt-4"),
        quick_only=_QUICK_ONLY,
        max_workers=_MAX_WORKERS,
        benchmark_max_iters=_BENCHMARK_MAX_ITERS,
    )
    runner.run_all()
    logger.info("A/B 对比基准完成")


if __name__ == "__main__":
    main()
