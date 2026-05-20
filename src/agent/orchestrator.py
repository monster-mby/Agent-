"""
这是一个企业级技能编排与 Agent 执行框架，能自动发现并注册技能，通过自然语言分析意图，自动路由到单技能、预定义流水线（顺序 / 并行）或多轮 Agent 模式，最终聚合结果统一返回。下面从功能概览、核心结构设计、类 / 方法详解、方法调用链路、代码质量评价五个维度深度解析。
一、功能概览
这个工具的核心能力是：
自动发现与注册：启动时自动扫描并注册所有预设（preset）和自定义（custom）技能，新增技能只需按规范添加。
自然语言意图路由：用户输入自然语言 → 关键词匹配预定义流水线 → 无匹配则 LLM 匹配单技能。
多技能流水线编排：
顺序执行：技能 A 的输出通过input_mapper映射为技能 B 的输入，形成 A→B→C 的处理链。
并行执行：多个技能同时接收相同输入，执行后合并结果。
预定义流水线开箱即用：内置 6 条高频办公流水线（总结后发邮件、解释代码并生成测试等）。
多轮 Agent 模式（新增）：支持 LLM 多轮调用工具，自动维护对话历史，有熔断机制（连续失败终止）。
结果统一聚合：所有执行结果封装为统一的OrchestratorResult，包含步骤详情、最终输出、执行摘要、耗时统计。
二、核心结构设计
代码采用分层架构，核心结构如下：
plaintext
┌─────────────────────────────────────────────────────────────┐
│  1. 数据结构层（dataclass/Enum）                             │
│     - PipelineType（流水线类型枚举）                          │
│     - PipelineStep/Pipeline（流水线定义）                     │
│     - StepResult/OrchestratorResult（结果定义）               │
│     - MessageHistory（对话历史管理器）                        │
├─────────────────────────────────────────────────────────────┤
│  2. 预定义流水线层（PREDEFINED_PIPELINES）                   │
│     - 内置6条高频办公流水线（总结后发邮件、解释代码并生成测试等）│
├─────────────────────────────────────────────────────────────┤
│  3. 主编排器类（SkillOrchestrator）                          │
│     - 初始化：自动发现注册技能、注册预定义流水线、同步工具到LLM │
│     - 主入口：process（自然语言处理）、run_agent（多轮Agent）  │
│     - 核心逻辑：流水线匹配、单技能/流水线/Agent执行、结果聚合  │
│     - 管理接口：注册流水线、列出技能/流水线、获取状态          │
├─────────────────────────────────────────────────────────────┤
│  4. 便捷函数层                                                │
│     - get_orchestrator（全局单例获取）                        │
│     - 命令行测试入口（__main__）                              │
└─────────────────────────────────────────────────────────────┘

### 链路 1：自然语言处理（process）
```
SkillOrchestrator.process()
  [自然语言处理主入口] 协调流水线匹配或单技能执行，计算总耗时并返回统一结果
  ├→ _match_pipeline(user_input)  [关键词匹配流水线] 基于触发词匹配预定义流水线，返回得分最高的流水线或None
  │   ├─ 匹配成功 → _execute_pipeline_from_user_input()
  │   │   [从用户输入提取初始参数] 调用LLM解析输入参数，无参数则默认填充，启动流水线
  │   │   └→ run_pipeline()
  │   │       [显式执行指定流水线] 根据流水线类型选择顺序或并行执行模式
  │   │       ├→ _execute_sequential() [顺序执行] 按序执行步骤，上一步输出通过input_mapper映射为下一步输入，遇错即停
  │   │       │   └→ 遍历步骤 → _execute_single_tool()
  │   │       │       [执行单个技能] 调用SkillManager执行技能，记录输入/输出/耗时，返回StepResult和工具字符串
  │   │       └→ _execute_parallel() [并行执行] 所有步骤共享相同初始输入依次执行（当前为串行模拟，未来可改为真并行）
  │   │           └→ 遍历步骤 → _execute_single_tool()
  │   │               [执行单个技能] 同上
  │   └─ 匹配失败 → _execute_single_skill() [LLM匹配单技能] 调用LLM分析意图并匹配单个技能，执行后返回结果
  │       ├→ llm_client.chat() [LLM分析意图] 传入用户输入与工具列表，获取LLM生成的tool_call
  │       └→ _execute_single_tool() [执行单个技能] 同上
  ├→ _build_result_summary() [构建结果摘要] 遍历步骤结果，用✅/❌标记状态，显示步骤描述与耗时
  └→ 封装为 OrchestratorResult 返回
       [聚合统一结果] 将步骤详情、最终输出、摘要、耗时封装为结构化对象
```


### 链路 2：多轮 Agent（run_agent）
```
SkillOrchestrator.run_agent()
  [多轮Agent模式主入口] LLM自动调用工具，维护对话历史，支持连续失败熔断
  ├→ 初始化 MessageHistory，添加用户输入
  │   [初始化对话历史] 创建历史管理器，存入系统提示（若有）和用户初始输入
  └→ 循环（最多max_turns轮）：
      ├→ llm_client.chat(messages=history, tools=_tools_cache)
      │   [LLM决策] 传入完整对话历史与工具列表，获取LLM的下一步决策（直接回复或工具调用）
      ├─ 无tool_calls → 封装结果返回
      │   [任务完成] LLM认为无需调用工具，直接返回最终回复与执行摘要
      └─ 有tool_calls：
          ├→ 遍历tool_calls：
          │   ├→ _execute_single_tool()
          │   │   [执行指定技能] 执行LLM选中的技能，记录输入、输出、耗时与成功状态
          │   ├→ 记录StepResult
          │   │   [保存步骤详情] 将当前工具执行的完整信息存入步骤结果列表
          │   ├→ history.add_tool_result()
          │   │   [更新对话历史] 将工具执行结果（自动截断）添加到历史，供LLM下一轮参考
          │   └─ 连续失败≥max_consecutive_failures → 熔断终止
          │       [安全熔断] 触发连续失败阈值，停止Agent循环，避免无限错误
          └→ 达到max_turns → 封装结果返回
              [强制终止] 达到最大对话轮数，返回已执行的结果摘要与提示
```
"""
from __future__ import annotations
import os
import sys
import re
import importlib.util
import inspect
import logging

# 导入基础设施模块（带容错）
try:
    from src.infrastructure.logger import set_trace_id, get_trace_id
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    def set_trace_id(trace_id=None): return ""
    def get_trace_id(): return ""

# 初始化日志，方便调试导入过程
logger = logging.getLogger(__name__)


import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

# 现有组件
from src.skills.base.skill_manager import SkillManager

try:
    from src.core.model_client import LiteLLMClient
except ImportError:
    LiteLLMClient = None
from src.skills.base.base_skill import BaseSkill
from src.agent.llm_client import SimulatedLLM
from src.skills.custom.rag_skills.query_rewrite_skill.skill import QueryRewriteSkill, QueryRewriteInput, QueryHistoryItem
from src.skills.custom.rag_skills.rerank_skill.skill import RerankSkill, RerankInput, RerankCandidate



# ═══════════════════════════════════════════
# 流水线类型枚举
# ═══════════════════════════════════════════

class PipelineType(Enum):
    """流水线类型"""
    SEQUENTIAL = "sequential"   # 顺序执行：A → B → C
    PARALLEL = "parallel"       # 并行执行：A | B | C

# ═══════════════════════════════════════════
# 流水线单个步骤定义
# ═══════════════════════════════════════════
@dataclass
class PipelineStep:
    """
    流水线中的一个步骤

    Attributes:
        skill_name: 技能名称（必须已在 SkillManager 中注册）
        input_mapper: 函数，接收上一步输出，返回本步输入 dict。
                      若为 None，则使用原始用户输入。
        description: 步骤描述（用于日志/调试）
    """
    skill_name: str
    input_mapper: Optional[Callable[..., Dict[str, Any]]] = None
    description: str = ""

# ═══════════════════════════════════════════
# 完整流水线定义
# ═══════════════════════════════════════════
@dataclass
class Pipeline:
    """
    作用：定义完整的流水线。

    Attributes:
        name: 流水线名称
        steps: 步骤列表
        pipeline_type: SEQUENTIAL 或 PARALLEL
        description: 流水线描述
        triggers: 触发关键词（用于 LLM 匹配流水线）
    """
    name: str
    steps: List[PipelineStep]
    pipeline_type: PipelineType = PipelineType.SEQUENTIAL
    description: str = ""
    triggers: List[str] = field(default_factory=list)

# ═══════════════════════════════════════════
# 单步骤执行结果
# ═══════════════════════════════════════════
@dataclass
class StepResult:
    """单个步骤的执行结果"""
    step_description: str # 步骤描述
    skill_name: str # 技能名称
    input_data: Dict[str, Any] # 输入数据
    output: Any # 技能输出
    success: bool # 是否成功
    error: Optional[str] = None # 错误信息
    elapsed_ms: float = 0.0 # 执行耗时

# ═══════════════════════════════════════════
# 	编排器统一返回结果
# ═══════════════════════════════════════════
@dataclass
class OrchestratorResult:
    """
    编排器统一返回结果

    Attributes:
        success: 是否全部成功
        pipeline_type: 使用的流水线类型（单技能时为 "single"）
        pipeline_name: 流水线名称（单技能时为技能名）
        step_results: 各步骤执行结果
        final_output: 最终聚合输出（单技能时为技能输出）
        summary: 人类可读的结果摘要
        elapsed_ms: 总耗时
    """
    success: bool
    pipeline_type: str
    pipeline_name: str
    step_results: List[StepResult]
    final_output: Any
    summary: str = ""
    elapsed_ms: float = 0.0


# ═══════════════════════════════════════════
# 预定义流水线：聚焦知识库构建与检索，与项目"企业学习助手"定位一致
# ═══════════════════════════════════════════

PREDEFINED_PIPELINES: List[Pipeline] = [
    # ────────────────────────────────────────────
    # 1. 单文档索引 → 检索
    # ────────────────────────────────────────────
    Pipeline(
        name="index_then_search",
        description="先对文档构建知识图谱索引，再对索引进行问答检索",
        triggers=[
            # 有明确"先索引再检索"动作链
            "索引并查询",
            "索引后搜索",
            "索引并检索",
            "索引然后查询",
            "先索引再查询",
            "先索引再检索",
            # 明确提到"知识图谱"
            "构建知识图谱并查询",
            "构建索引并提问",
            "构建索引并查询",
            # 英文
            "index and search",
            "index then search",
            "build index and search",
            "index and query",
        ],
        steps=[
            PipelineStep(
                skill_name="graphrag_indexer",
                description="第一步：构建知识图谱索引",
            ),
            PipelineStep(
                skill_name="graphrag_searcher",
                description="第二步：对索引进行问答检索",
            ),
        ],
    ),

    # ────────────────────────────────────────────
    # 2. 多文件索引 → 检索
    # ────────────────────────────────────────────
    Pipeline(
        name="multi_file_index_then_search",
        description="索引多个文件后统一检索",
        triggers=[
            # 必须同时包含"批量/多文件" + "索引"两个信号
            "批量索引这些文件然后搜索",
            "批量索引文件并搜索",
            "批量索引并查询",
            "批量索引后检索",
            "索引多个文件并搜索",
            "索引多个文件然后查询",
            "多文件索引并检索",
            "多文件索引后搜索",
            "多个文件索引后查询",
            "索引这些文件然后搜索",
            "索引这批文件并查询",
            # 英文
            "index multiple files and search",
            "index multiple files then search",
            "batch index files and search",
            "batch index and search",
            "multiple files index and search",
            "index multiple documents",
        ],
        steps=[
            PipelineStep(
                skill_name="graphrag_indexer",
                description="第一步：索引多个文件",
            ),
            PipelineStep(
                skill_name="graphrag_searcher",
                description="第二步：统一检索",
            ),
        ],
    ),
]

# ... existing code ...

# ═══════════════════════════════════════════
# 作用：管理 Agent 模式的对话历史，避免外部修改内部状态
# ═══════════════════════════════════════════
class MessageHistory:
    """对话历史管理器"""
    MAX_TOOL_CONTENT_LENGTH = 4000

    def __init__(self, system_prompt: str | None = None):
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self.add_system(system_prompt)

    def add_system(self, content: str) -> None:
        self._messages.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content[:self.MAX_TOOL_CONTENT_LENGTH],
        })

    def to_list(self) -> list[dict[str, Any]]:
        """返回副本，避免外部修改影响内部状态"""
        return [dict(m) for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

# ═══════════════════════════════════════════
# 整个编排器的核心，负责初始化、意图匹配、技能 / 流水线执行、结果聚合。
# ═══════════════════════════════════════════
class SkillOrchestrator:
    """
    技能编排器 — 统一入口

    使用方式：
        orch = SkillOrchestrator()
        result = orch.process("帮我总结这段文本：...")

        # 或者显式执行预定义流水线
        result = orch.run_pipeline("summarize_then_email", initial_input={...})
    """
    def __init__(self, llm_client=None):
        """
        初始化编排器。

        Args:
            llm_client: LLM 客户端。支持：
                - LiteLLMAdapter（真实 LLM，推荐）
                - SimulatedLLM（模拟，测试用）
                - 任何实现 LLMClient 的对象
                默认：优先 LiteLLMAdapter → 回退 SimulatedLLM
        """
        self.skill_manager = SkillManager()
        self._pipelines: Dict[str, Pipeline] = {}
        self._tools_cache: list[dict[str, Any]] = []  # 新增

        # LLM 客户端：优先真实 LLM → 回退模拟
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = self._create_default_llm()

        # 启动时自动完成所有初始化
        self._auto_discover_and_register_skills()
        self._register_predefined_pipelines()
        self._sync_tools_to_llm()

    # ═══════════════════════════════════════════
    # 作用：执行多轮 Agent 模式，LLM 自动调用工具，维护对话历史，有熔断机制
    # ═══════════════════════════════════════════
    def run_agent(
            self,
            user_input: str,
            max_turns: int = 5,
            system_prompt: str | None = None,
            max_consecutive_failures: int = 3,
    ) -> OrchestratorResult:
        start_time = time.time()
        step_results: list[StepResult] = []

        history = MessageHistory(system_prompt=system_prompt)
        history.add_user(user_input)

        consecutive_failures = 0

        for turn in range(max_turns):
            try:
                llm_response = self.llm_client.chat(
                    messages=history.to_list(), tools=self._tools_cache
                )
            except Exception as e:
                return OrchestratorResult(
                    success=False,
                    pipeline_type="agent",
                    pipeline_name=f"agent_turn_{turn}",
                    step_results=step_results,
                    final_output=None,
                    summary=f"❌ LLM 调用失败（第 {turn + 1} 轮）：{e}",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            if isinstance(llm_response, dict):
                content = llm_response.get("content", "")
                tool_calls_raw = llm_response.get("tool_calls", [])
            else:
                content = llm_response.content
                tool_calls_raw = llm_response.tool_calls

            if not tool_calls_raw:
                summary = self._build_result_summary(step_results)
                # 只要没触发熔断且最后一步成功就算成功
                is_success = (
                        len(step_results) > 0
                        and step_results[-1].success
                        and consecutive_failures < max_consecutive_failures
                )
                return OrchestratorResult(
                    success=is_success,
                    pipeline_type="agent",
                    pipeline_name=f"agent_{len(step_results)}_tools",
                    step_results=step_results,
                    final_output=content,
                    summary=(
                        summary + f"\n 最终回复：{content[:200]}"
                        if content else summary
                    ),
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            history.add_assistant(
                content=content,
                tool_calls=[
                    tc if isinstance(tc, dict) else tc.model_dump()
                    for tc in tool_calls_raw
                ],
            )

            for i, tc in enumerate(tool_calls_raw):
                skill_name = tc["name"] if isinstance(tc, dict) else tc.name
                arguments = tc["arguments"] if isinstance(tc, dict) else tc.arguments
                tool_call_id = (
                    tc.get("id", f"call_{turn}_{i}")
                    if isinstance(tc, dict)
                    else (tc.id or f"call_{turn}_{i}")
                )
                step_result, tool_result = self._execute_single_tool(
                    skill_name=skill_name,
                    arguments=arguments,
                    turn=turn,
                )

                step_results.append(step_result)
                history.add_tool_result(tool_call_id, tool_result)

                if step_result.success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        summary = self._build_result_summary(step_results)
                        return OrchestratorResult(
                            success=False,
                            pipeline_type="agent",
                            pipeline_name=f"agent_fail_{turn}",
                            step_results=step_results,
                            final_output=None,
                            summary=summary + f"\n⚠️ 连续 {consecutive_failures} 次工具执行失败，Agent 终止",
                            elapsed_ms=(time.time() - start_time) * 1000,
                        )

        summary = self._build_result_summary(step_results)
        # 只要没触发熔断且最后一步成功就算成功
        is_success = (
                len(step_results) > 0
                and step_results[-1].success
                and consecutive_failures < max_consecutive_failures
        )
        return OrchestratorResult(
            success=is_success,
            pipeline_type="agent",
            pipeline_name="agent_max_turns",
            step_results=step_results,
            final_output=None,
            summary=summary + f"\n⚠️ 达到最大轮数（{max_turns}），Agent 循环终止",
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    @staticmethod
    def _create_default_llm():
        """创建默认 LLM 客户端：优先真实 LLM，不可用则回退模拟"""
        import logging
        logger = logging.getLogger(__name__)

        if LiteLLMClient is not None:
            try:
                client = LiteLLMClient()
                logger.info("使用 LiteLLMClient（真实 LLM）：%s", client.litellm_model)
                return client
            except Exception as e:
                logger.warning("LiteLLMClient 初始化失败（%s），回退 SimulatedLLM", e)
        else:
            logger.warning("LiteLLMClient 不可用，回退 SimulatedLLM")
        return SimulatedLLM()

    # ═══════════════════════════════════════════
    #作用：自动发现并注册所有 preset 和 custom 技能（当前为静态导入，未来可改为动态扫描）。
    #逻辑：导入所有技能类，遍历调用skill_manager.register()注册。
    # ═══════════════════════════════════════════

    def _auto_discover_and_register_skills(self):
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(src_dir)
        skills_root_dir = os.path.join(src_dir, "skills")

        # ★ 加到末尾而非开头，避免覆盖标准库
        if project_root not in sys.path:
            sys.path.append(project_root)

        target_scan_dirs = [
            os.path.join(skills_root_dir, "preset"),
            os.path.join(skills_root_dir, "custom"),
        ]

        registered_count = 0
        failed_list = []

        for base_dir in target_scan_dirs:
            if not os.path.exists(base_dir):
                logger.warning("技能目录不存在，已跳过: %s", base_dir)
                continue

            for root, _, files in os.walk(base_dir):
                if "skill.py" not in files:
                    continue

                skill_file_path = os.path.join(root, "skill.py")

                # ★ 修复：Windows 跨盘符兼容（pytest tmp_path 在 C:，项目在 D:）
                try:
                    relative_path = os.path.relpath(skill_file_path, src_dir)
                except ValueError:
                    # Windows 跨盘符回退（pytest tmp_path 与项目不在同一盘符）
                    # skill_file_path 来自 os.walk(base_dir)，一定在 base_dir 之下
                    relative_path = os.path.relpath(skill_file_path, base_dir)

                module_name = relative_path.replace(os.sep, ".")[:-3]

                try:
                    spec = importlib.util.spec_from_file_location(module_name, skill_file_path)
                    if not spec or not spec.loader:
                        failed_list.append(f"{skill_file_path} | 原因：无法创建模块规范")
                        continue

                    skill_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = skill_module

                    try:
                        spec.loader.exec_module(skill_module)
                    except Exception:
                        sys.modules.pop(module_name, None)  # ★ 清理残骸
                        raise

                    for class_name, class_obj in inspect.getmembers(skill_module, inspect.isclass):
                        try:
                            if issubclass(class_obj, BaseSkill) and class_obj is not BaseSkill:
                                self.skill_manager.register(class_obj)
                                registered_count += 1
                                logger.info("✅ 成功注册技能: %s | 文件: %s", class_name, skill_file_path)
                        except TypeError:
                            logger.debug("跳过非兼容类型: %s.%s", module_name, class_name)

                except Exception as e:
                    error_msg = f"{skill_file_path} | 原因：{str(e)}"
                    failed_list.append(error_msg)
                    logger.error("❌ 技能导入失败: %s", error_msg, exc_info=True)

        logger.info("=" * 80)
        logger.info("📦 技能自动扫描注册完成")
        logger.info(f"   ✅ 成功注册：{registered_count} 个技能")
        if failed_list:
            logger.warning(f"   ❌ 导入失败：{len(failed_list)} 个技能")
            for fail in failed_list:
                logger.warning(f"      - {fail}")
        logger.info("=" * 80)

    def _register_predefined_pipelines(self):
        """注册预定义的 6 条流水线"""
        for pipeline in PREDEFINED_PIPELINES:
            self._pipelines[pipeline.name] = pipeline

    def _sync_tools_to_llm(self):
        """缓存工具列表"""
        self._tools_cache = self.skill_manager.generate_tool_descriptions()

    # ═══════════════════════════════════════════
    # 自然语言处理主入口
    # ═══════════════════════════════════════════

    def process(self, user_input: str, **kwargs) -> OrchestratorResult:
        """
        处理用户自然语言输入（主入口）

        流程：
          1. LLM 意图分类 → 映射流水线
          2. 若匹配流水线 → 执行流水线
          3. 否则 → LLM tool_calls 匹配单技能 → 执行单技能

        Args:
            user_input: 用户自然语言输入
            **kwargs: 传递给技能的额外参数

        Returns:
            OrchestratorResult: 统一结果对象
        """
        # 设置 trace_id（每个请求的唯一标识）
        if INFRASTRUCTURE_AVAILABLE:
            try:
                set_trace_id()  # 自动生成 UUID
            except Exception as e:
                logger.warning(f"设置 trace_id 失败: {e}")


        start_time = time.time()

        # ── 第 1 步：检测是否匹配预定义流水线 ──
        matched_pipeline = self._match_pipeline(user_input)
        if matched_pipeline:
            result = self._execute_pipeline_from_user_input(
                matched_pipeline, user_input
            )
            result.elapsed_ms = (time.time() - start_time) * 1000
            return result

        # ── 第 2 步：LLM 匹配单技能 ──
        result = self._execute_single_skill(user_input)
        result.elapsed_ms = (time.time() - start_time) * 1000
        return result

    # ═══════════════════════════════════════════
    # 流水线匹配
    # ═══════════════════════════════════════════

    def _match_pipeline(self, user_input: str) -> Optional[Pipeline]:
        """基于 LLM 意图分类匹配预定义流水线（已移除关键词匹配）"""
        intent = self._classify_intent_with_llm(user_input)
        if intent and intent.get("pipeline"):
            matched = self._pipelines.get(intent["pipeline"])
            if matched:
                logger.info(
                    "LLM 意图分类命中流水线 | input='%s' | pipeline=%s",
                    user_input[:50], intent["pipeline"],
                )
                return matched
        return None

    # ═══════════════════════════════════════════
    # LLM 意图分类
    # ═══════════════════════════════════════════

    def _classify_intent_with_llm(self, user_input: str) -> Optional[dict]:
        """
        一次 LLM 调用同时返回流水线分类 + RAG 检索模式（合并版）。

        Returns:
            {"pipeline": "xxx", "rag_mode": "vector"|"graphrag", "intent": "..."}
            或 None（分类失败时降级）
        """
        try:
            if not self.llm_client:
                return None

            pipeline_names = list(self._pipelines.keys()) if self._pipelines else []
            pipeline_desc_list = []
            for name, p in (self._pipelines or {}).items():
                desc = f"- {name}: {'→'.join(s.skill_name for s in p.steps)}"
                pipeline_desc_list.append(desc)

            prompt = f"""你是一个意图分类助手。一次输出同时完成两项判断：

已知流水线：
{pipeline_desc_list}

用户输入：{user_input}

输出 JSON（不要 Markdown 代码块）：
{{
  "pipeline": "流水线名称或null",
  "rag_mode": "vector 或 graphrag",
  "intent": "简短意图描述"
}}

判断 rag_mode 规则：
- 问"整体架构/模块关系/全局总结/人物关联/对比分析" → graphrag
- 其他日常问题 → vector"""

            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.content if hasattr(response, 'content') else str(response)
            import json as _json
            data = _json.loads(content.strip().strip('`').strip())
            data.setdefault("rag_mode", "vector")
            return data
        except Exception:
            return None

    # ═══════════════════════════════════════════
    # 单技能执行
    # ═══════════════════════════════════════════

    def _execute_single_skill(self, user_input: str) -> OrchestratorResult:
        step_results = []
        try:
            # LLM tool_calls 统一路由
            routed = self._route_to_skill(user_input)  # 已移除关键词，直接返回 None
            if routed:
                skill_name, arguments = routed
                if not arguments:
                    arguments = {}
                step_result, _ = self._execute_single_tool(
                    skill_name, arguments, turn=0
                )
                step_results.append(step_result)
                return OrchestratorResult(
                    success=step_result.success,
                    pipeline_type="single",
                    pipeline_name=skill_name,
                    step_results=step_results,
                    final_output=step_result.output,
                    summary=self._build_result_summary(step_results),
                )

            #  原有 LLM 路由逻辑
            llm_response = self.llm_client.chat(
                messages=[{"role": "user", "content": user_input}],
                tools=self._tools_cache,
            )
            # 兼容 dict（SimulatedLLM）和 ChatResponse（LiteLLMClient）
            if isinstance(llm_response, dict):
                tool_calls_raw = llm_response.get("tool_calls", [])
            else:
                tool_calls_raw = llm_response.tool_calls

            if not tool_calls_raw:
                return OrchestratorResult(
                    success=False,
                    pipeline_type="single",
                    pipeline_name="unknown",
                    step_results=[],
                    final_output=None,
                    summary="️ 未能识别您的意图，请提供更多信息或换个说法。",
                )

            tc = tool_calls_raw[0]
            skill_name = tc["name"] if isinstance(tc, dict) else tc.name
            arguments = tc["arguments"] if isinstance(tc, dict) else tc.arguments

            #  新增：如果 LLM 没有提取参数，尝试从用户输入中智能推断
            if not arguments:
                arguments = self._infer_arguments_from_input(skill_name, user_input)
                if arguments:
                    import logging
                    logger.info(
                        "从用户输入中推断参数：%s → %s", user_input, arguments
                    )

            step_result, _ = self._execute_single_tool(skill_name, arguments, turn=0)
            step_results.append(step_result)

            return OrchestratorResult(
                success=step_result.success,
                pipeline_type="single",
                pipeline_name=skill_name,
                step_results=step_results,
                final_output=step_result.output,
                summary=self._build_result_summary(step_results),
            )
        except Exception as e:
            return OrchestratorResult(
                success=False,
                pipeline_type="single",
                pipeline_name="error",
                step_results=step_results,
                final_output=None,
                summary=f"❌ 执行失败：{str(e)}",
            )

    def execute_smart_rag(self, query: str, mode: str = "auto") -> OrchestratorResult:
        """智能路由 RAG：根据查询意图自动选择检索链路"""

        query = (query or "").strip()
        if not query:
            return self.execute_rag_pipeline(query="")

        # 1. 强制路由
        if mode == "vector":
            return self.execute_rag_pipeline(query=query)

        if mode == "graphrag":
            res = self.skill_manager.call("graphrag_searcher", query=query, mode="global")
            return OrchestratorResult(
                success=res.get("status") == "success",
                pipeline_type="macro_rag",
                pipeline_name="graphrag_global_search",
                step_results=[],
                final_output=res.get("output"),
                summary="✅ GraphRAG 全局检索完成"
            )

        if mode == "hybrid":
            # 双路并行召回，合并结果
            vec_result = self.execute_rag_pipeline(query=query)
            graph_res = self.skill_manager.call("graphrag_searcher", query=query, mode="global")
            graph_output = graph_res.get("output")
            graph_sources = []
            if isinstance(graph_output, dict):
                graph_sources = graph_output.get("results", graph_output.get("documents", []))
            elif isinstance(graph_output, list):
                graph_sources = graph_output
            graph_result = {"sources": graph_sources}
            return self._merge_results(vec_result.final_output, graph_result)

        # 2. Auto 模式：LLM 意图判断（一次调用同时返回 pipeline + rag_mode）
        if mode == "auto":
            intent = self._classify_intent_with_llm(query)
            if intent and intent.get("rag_mode") == "graphrag":
                res = self.skill_manager.call("graphrag_searcher", query=query, mode="global")
                return OrchestratorResult(
                    success=res.get("status") == "success",
                    pipeline_type="macro_rag",
                    pipeline_name="graphrag_auto_search",
                    step_results=[],
                    final_output=res.get("output"),
                    summary="✅ LLM 判定为宏观/关系问题 -> GraphRAG"
                )

        # 3. 默认走 Vector + Rerank 链路
        return self.execute_rag_pipeline(query=query)


    def execute_rag_pipeline(
            self,
            query: str,
            history: list = None,
            top_k: int = 10,
            top_n: int = 5,
            pre_rerank_cap: int = 30,
    ) -> OrchestratorResult:
        """执行增强型 RAG 全链路（同步版：改写 -> 召回 -> 精排）"""
        start_time = time.time()

        # 0. 安全构建 history
        safe_history = []
        for h in (history or []):
            if isinstance(h, dict):
                safe_history.append(QueryHistoryItem(**h))
            elif isinstance(h, QueryHistoryItem):
                safe_history.append(h)

        # 1. Query Rewrite (查询改写)
        try:
            rewrite_input = QueryRewriteInput(original_query=query, history=safe_history)
            # 【修正】使用 call 方法，并从 output 字段取值
            res = self.skill_manager.call("query_rewrite_skill", **rewrite_input.model_dump())
            rewrite_result = res.get("output") if res.get("status") == "success" else None
            queries = rewrite_result.get("rewritten_queries", [query]) if rewrite_result else [query]
        except Exception as e:
            logger.warning(f"QueryRewrite 失败，使用原始查询: {e}")
            queries = [query]

        all_candidates = []

        # 2. Retrieval (并行召回) - 选用 vector_search
        for q in queries:
            try:
                # 【修正】使用 call 方法，技能名为 vector_search
                res = self.skill_manager.call("vector_search", query=q, top_k=top_k)
                vec_res = res.get("output") if res.get("status") == "success" else None

                for doc in (vec_res.get("results", []) if vec_res else []):
                    content = doc.get("content")
                    if content and isinstance(content, str):
                        all_candidates.append(RerankCandidate(
                            content=content,
                            source=doc.get("source", "vector"),
                            doc_id=str(doc.get("id")),
                            original_score=doc.get("score")
                        ))
            except Exception as e:
                logger.error(f"Vector Search failed for '{q}': {e}")

        # 3. Merge & Deduplicate (合并去重)
        seen_ids = set()
        unique_candidates = []
        for c in all_candidates:
            if c.doc_id and c.doc_id not in seen_ids:
                seen_ids.add(c.doc_id)
                unique_candidates.append(c)

        unique_candidates.sort(key=lambda x: x.original_score or 0.0, reverse=True)
        unique_candidates = unique_candidates[:pre_rerank_cap]

        # 4. Rerank (相关性精排)
        final_docs = []
        if unique_candidates:
            try:
                rerank_input = RerankInput(query=query, candidates=unique_candidates, top_n=top_n)
                res = self.skill_manager.call("rerank_skill", **rerank_input.model_dump())
                rerank_result = res.get("output") if res.get("status") == "success" else None
                final_docs = rerank_result.get("reranked_docs", []) if rerank_result else []
            except Exception as e:
                logger.warning(f"Rerank 失败，降级为原始顺序: {e}")
                final_docs = [c.model_dump() for c in unique_candidates[:top_n]]

        # 5. LLM 生成真实答案 (调用 RagAnswerSkill)
        final_answer = ""
        if final_docs:
            try:
                from src.skills.custom.rag_skills.rag_answer.skill import SearchResultRef

                # 将精排结果转换为 RagAnswerSkill 需要的格式
                search_results = [
                    SearchResultRef(
                        chunk_id=doc.get("doc_id", f"doc_{i}"),
                        text=doc.get("content", ""),
                        score=doc.get("rerank_score", doc.get("original_score", 0.5)),
                        metadata=doc.get("metadata", {})
                    )
                    for i, doc in enumerate(final_docs)
                ]

                rag_input = {
                    "query": query,
                    "search_results": [r.model_dump() for r in search_results],
                    "include_citations": True,
                    "estimate_confidence": True,
                    "max_context_chunks": top_n
                }

                res = self.skill_manager.call("rag_answer", **rag_input)
                if res.get("status") == "success":
                    output = res.get("output", {})
                    final_answer = output.get("answer", "")
            except Exception as e:
                logger.warning(f"RagAnswer 生成失败，使用占位符: {e}")
                final_answer = "已检索到相关文档，但生成答案时出现错误。"

        elapsed_ms = (time.time() - start_time) * 1000

        return OrchestratorResult(
            success=True,
            pipeline_type="rag_enhanced",
            pipeline_name="enhanced_rag_pipeline",
            step_results=[],
            final_output={"answer": final_answer, "sources": final_docs},
            summary=f"✅ RAG 完成 | 召回: {len(unique_candidates)} | 精排后: {len(final_docs)}",
            elapsed_ms=elapsed_ms
        )

    def _merge_results(self, vec_result: dict, graph_result: dict) -> OrchestratorResult:
        """简易双路合并：优先 vector 的 answer，补充 graph 的来源"""
        sources = list(vec_result.get("sources", []))
        # 追加 graph 来源（去重）
        seen_ids = {s.get("doc_id") for s in sources if s.get("doc_id")}
        for src in graph_result.get("sources", []):
            if src.get("doc_id") not in seen_ids:
                sources.append(src)
                seen_ids.add(src.get("doc_id", ""))
        return OrchestratorResult(
            success=True,
            pipeline_type="hybrid_rag",
            pipeline_name="vector_graphrag_merged",
            step_results=[],
            final_output={
                "answer": vec_result.get("answer", graph_result.get("answer", "")),
                "sources": sources,
            },
            summary=f"✅ 双路融合完成 | 来源数: {len(sources)}"
        )

    def _infer_arguments_from_input(self, skill_name: str, user_input: str) -> dict:
        """
        从用户输入中智能推断技能参数（当 LLM 未提取时）
        """
        import re

        arguments = {}

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # graphrag_indexer：提取文件路径 或 文档文本
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if skill_name == "graphrag_indexer":
            # 提取文件路径（注意：用非捕获组 (?:...) 否则 findall 只返回扩展名）
            file_matches = re.findall(
                r'[\w./\\-]+\.(?:txt|md|json|csv|py|log)',
                user_input, re.IGNORECASE
            )
            if file_matches:
                arguments["input_files"] = [fp.strip() for fp in file_matches]

            # 如果没有文件，尝试提取被引号包裹的长文本作为 documents
            if not file_matches:
                quoted = re.findall(r'[""]([^""]{15,})[""]', user_input)
                if quoted:
                    arguments["documents"] = [
                        {"id": f"doc_{i}", "text": t.strip()}
                        for i, t in enumerate(quoted, 1)
                    ]
                else:
                    # 去掉常见指令前缀，剩余部分当做文本
                    clean = re.sub(
                        r'(请|帮我|帮忙|给我|索引|构建|创建|建立)(一下|一个)?(知识图谱|索引)?',
                        '', user_input
                    ).strip("，。、；：！？,.;:!? ")
                    if len(clean) > 15:
                        arguments["documents"] = [{"id": "input_text", "text": clean}]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # graphrag_searcher：提取 query、mode
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif skill_name == "graphrag_searcher":
            input_lower = user_input.lower()

            # 推断 mode
            if any(kw in input_lower for kw in ["全局", "整体", "global", "overview", "总览"]):
                arguments["mode"] = "global"
            elif any(kw in input_lower for kw in ["混合", "综合", "hybrid", "combined"]):
                arguments["mode"] = "hybrid"
            else:
                arguments["mode"] = "local"

            # 提取 query（去掉指令前缀）
            query = user_input
            for pat in [
                r'(请|帮我|帮忙|给我|用|使用|通过).{0,10}?(搜索|检索|查询|查找|查一下|问一下)',
                r'(在|从)(知识)?图谱中.{0,5}?(搜索|检索|查询|查)',
            ]:
                query = re.sub(pat, '', query, flags=re.IGNORECASE)
            query = query.strip("，。、；：！？,.;:!? ")
            arguments["query"] = query or user_input

            # 提取 top_k
            m = re.search(r'(top[_ ]?k|前|取前)\s*(\d+)', user_input, re.IGNORECASE)
            if m:
                k = int(m.group(2))
                arguments["top_k_entities"] = min(k, 20)
                arguments["top_k_communities"] = min(k, 10)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # document_loader（保留原逻辑）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif skill_name == "document_loader":
            file_pattern = r'[\w./\\-]+\.(?:txt|md|json|csv|py|js|html|xml|yaml|yml|log|ini|cfg|toml)'
            match = re.search(file_pattern, user_input, re.IGNORECASE)
            if match:
                arguments['file_path'] = match.group(0)

        return arguments

    # ... existing code ...

    def _route_to_skill(self, user_input: str) -> Optional[tuple[str, dict]]:
        """已移除关键词路由，交由 LLM tool_calls 统一处理"""
        return None

    # ═══════════════════════════════════════════
    # 流水线执行
    # ═══════════════════════════════════════════

    def run_pipeline(self, pipeline_name: str,
                     initial_input: Dict[str, Any]) -> OrchestratorResult:
        """
        显式执行指定流水线

        Args:
            pipeline_name: 流水线名称
            initial_input: 第一步的输入参数

        Returns:
            OrchestratorResult
        """
        pipeline = self._pipelines.get(pipeline_name)
        if pipeline is None:
            return OrchestratorResult(
                success=False,
                pipeline_type="unknown",
                pipeline_name=pipeline_name,
                step_results=[],
                final_output=None,
                summary=f"❌ 流水线 '{pipeline_name}' 不存在。可用流水线：{list(self._pipelines.keys())}",
            )

        start_time = time.time()
        step_results: List[StepResult] = []

        try:
            if pipeline.pipeline_type == PipelineType.SEQUENTIAL:
                step_results = self._execute_sequential(pipeline, initial_input)
            else:
                step_results = self._execute_parallel(pipeline, initial_input)

            final_output = step_results[-1].output if step_results else None
            summary = self._build_result_summary(step_results)

            return OrchestratorResult(
                success=all(s.success for s in step_results),
                pipeline_type=pipeline.pipeline_type.value,
                pipeline_name=pipeline_name,
                step_results=step_results,
                final_output=final_output,
                summary=summary,
                elapsed_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return OrchestratorResult(
                success=False,
                pipeline_type=pipeline.pipeline_type.value,
                pipeline_name=pipeline_name,
                step_results=step_results,
                final_output=None,
                summary=f"❌ 流水线执行失败：{str(e)}",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def _execute_pipeline_from_user_input(self, pipeline, user_input):
        llm_response = self.llm_client.chat(
            messages=[{"role": "user", "content": user_input}],
            tools=self._tools_cache,
        )
        if isinstance(llm_response, dict):
            tool_calls_raw = llm_response.get("tool_calls", [])
        else:
            tool_calls_raw = llm_response.tool_calls

        initial_input = {}
        if tool_calls_raw:
            tc0 = tool_calls_raw[0]
            initial_input = dict(
                tc0["arguments"] if isinstance(tc0, dict) else tc0.arguments
            )
        if not initial_input:
            initial_input = {"text": user_input}

        # ─── 就加这一段（≤6行） ───
        # LLM 经常把 code 字段写成 function_code / source_code 等别名
        ALIASES = {
            "function_code": "code",
            "source_code": "code",
            "content": "text",
            "input_text": "text",
        }
        for alias, canonical in ALIASES.items():
            if alias in initial_input and canonical not in initial_input:
                initial_input[canonical] = initial_input.pop(alias)
        # ────────────────────────────

        return self.run_pipeline(pipeline.name, initial_input)

    def _execute_single_tool(
            self,
            skill_name: str,
            arguments: dict[str, Any],
            turn: int,
    ) -> tuple[StepResult, str]:
        t0 = time.time()
        try:
            output = self.skill_manager.invoke(skill_name, **arguments)
            elapsed = (time.time() - t0) * 1000
            step_result = StepResult(
                step_description=f"Agent 第 {turn + 1} 轮：{skill_name}",
                skill_name=skill_name,
                input_data=arguments,
                output=output,
                success=True,
                elapsed_ms=elapsed,
            )
            tool_result = output if isinstance(output, str) else str(output)
            return step_result, tool_result
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            step_result = StepResult(
                step_description=f"Agent 第 {turn + 1} 轮：{skill_name}",
                skill_name=skill_name,
                input_data=arguments,
                output=None,
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )
            return step_result, f"错误：{e}"

    def _execute_sequential(
        self, pipeline: Pipeline, initial_input: Dict[str, Any]
    ) -> List[StepResult]:
        """
        顺序执行流水线步骤

        每个步骤的输出通过 input_mapper 映射为下一步的输入
        """
        step_results = []
        current_input = initial_input

        for i, step in enumerate(pipeline.steps):
            # 如果有 input_mapper，用上一步输出映射输入
            if i > 0 and step.input_mapper is not None:
                prev_output = step_results[-1].output
                mapped_input = step.input_mapper(prev_output, current_input)
                if mapped_input:
                    current_input = {**current_input, **mapped_input}

            t0 = time.time()
            try:
                output = self.skill_manager.invoke(
                    step.skill_name, **current_input
                )
                elapsed = (time.time() - t0) * 1000
                step_results.append(StepResult(
                    step_description=step.description,
                    skill_name=step.skill_name,
                    input_data=current_input.copy(),
                    output=output,
                    success=True,
                    elapsed_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                step_results.append(StepResult(
                    step_description=step.description,
                    skill_name=step.skill_name,
                    input_data=current_input.copy(),
                    output=None,
                    success=False,
                    error=str(e),
                    elapsed_ms=elapsed,
                ))
                # 顺序流水线遇错即停
                break

        return step_results

    def _execute_parallel(
        self, pipeline: Pipeline, initial_input: Dict[str, Any]
    ) -> List[StepResult]:
        """
        并行执行流水线步骤

        所有步骤使用相同的初始输入，同时执行
        """
        step_results = []

        for step in pipeline.steps:
            t0 = time.time()
            try:
                output = self.skill_manager.invoke(
                    step.skill_name, **initial_input
                )
                elapsed = (time.time() - t0) * 1000
                step_results.append(StepResult(
                    step_description=step.description,
                    skill_name=step.skill_name,
                    input_data=initial_input.copy(),
                    output=output,
                    success=True,
                    elapsed_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                step_results.append(StepResult(
                    step_description=step.description,
                    skill_name=step.skill_name,
                    input_data=initial_input.copy(),
                    output=None,
                    success=False,
                    error=str(e),
                    elapsed_ms=elapsed,
                ))

        return step_results

    # ═══════════════════════════════════════════
    # 结果格式化
    # ═══════════════════════════════════════════

    def _build_result_summary(self, step_results: List[StepResult]) -> str:
        """构建人类可读的结果摘要"""
        if not step_results:
            return "⚠️ 无结果"

        lines = []
        for i, sr in enumerate(step_results, 1):
            icon = "✅" if sr.success else "❌"
            time_str = f" ({sr.elapsed_ms:.0f}ms)"
            lines.append(f"{icon} 步骤 {i}：{sr.step_description}{time_str}")
            if sr.error:
                lines.append(f"   错误：{sr.error}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 管理接口
    # ═══════════════════════════════════════════

    def register_pipeline(self, pipeline: Pipeline) -> None:
        """动态注册一条新的流水线"""
        self._pipelines[pipeline.name] = pipeline

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """列出所有已注册的流水线"""
        return [
            {
                "name": p.name,
                "description": p.description,
                "type": p.pipeline_type.value,
                "steps": [s.skill_name for s in p.steps],
                "triggers": p.triggers,
            }
            for p in self._pipelines.values()
        ]

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有已注册的技能"""
        return self.skill_manager.list_all()

    def get_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        return {
            "registered_skills": len(self.skill_manager.list_all()),
            "registered_pipelines": len(self._pipelines),
            "pipeline_names": list(self._pipelines.keys()),
            "llm_type": type(self.llm_client).__name__,
        }


# ═══════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════

# 全局单例（可选）
_default_orchestrator: Optional[SkillOrchestrator] = None


def get_orchestrator() -> SkillOrchestrator:
    """获取全局编排器单例"""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = SkillOrchestrator()
    return _default_orchestrator


# ═══════════════════════════════════════════
# 命令行测试入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SkillOrchestrator — 技能编排器测试")
    print("=" * 60)

    orch = SkillOrchestrator()

    # 查看状态
    status = orch.get_status()
    print(f"\n📊 编排器状态：")
    print(f"   已注册技能：{status['registered_skills']} 个")
    print(f"   已注册流水线：{status['registered_pipelines']} 条")
    print(f"   流水线列表：{status['pipeline_names']}")
    print(f"   LLM 类型：{status['llm_type']}")

    # 测试 1：单技能
    print("\n" + "─" * 60)
    print("  🧪 测试 1：单技能 — 文本摘要")
    print("─" * 60)
    result = orch.process(
        "帮我总结一下这段文本：人工智能是计算机科学的一个分支，"
        "致力于创建能够执行通常需要人类智能的任务的系统。"
        "机器学习是AI的核心驱动力，深度学习则是机器学习的重要子领域。"
    )
    print(f"   成功：{result.success}")
    print(f"   流水线：{result.pipeline_name}")
    print(f"   耗时：{result.elapsed_ms:.0f}ms")
    print(f"   摘要：{result.summary}")

    # 测试 2：流水线
    print("\n" + "─" * 60)
    print("  🧪 测试 2：流水线 — 解释代码 + 生成测试")
    print("─" * 60)
    result2 = orch.process(
        "分析这段代码然后生成测试：def multiply(a, b): return a * b"
    )
    print(f"   成功：{result2.success}")
    print(f"   流水线：{result2.pipeline_name}")
    print(f"   步骤数：{len(result2.step_results)}")
    print(f"   耗时：{result2.elapsed_ms:.0f}ms")
    print(f"   摘要：{result2.summary}")

    # 测试 3：列出流水线
    print("\n" + "─" * 60)
    print("  📋 可用流水线")
    print("─" * 60)
    for p in orch.list_pipelines():
        print(f"   🔗 {p['name']}")
        print(f"      描述：{p['description']}")
        print(f"      步骤：{' → '.join(p['steps'])}")
        print(f"      触发词：{', '.join(p['triggers'][:3])}...")
        print()

    print("=" * 60)
    print("  ✅ 所有测试完成")
    print("=" * 60)