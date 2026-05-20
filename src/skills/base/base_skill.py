"""
技能标准基类 - 所有技能必须继承此类
遵循吴恩达技能优先架构规范
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

# 导入基础设施模块（带容错）
try:
    from src.infrastructure.logger import set_trace_id, get_trace_id, set_skill_name, log_info, log_error
    from src.infrastructure.metrics import record_skill_metric
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    # 提供降级函数
    def set_trace_id(trace_id=None): return ""
    def get_trace_id(): return ""
    def set_skill_name(name): pass
    def log_info(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass
    def record_skill_metric(*args, **kwargs): pass

class BaseSkill(ABC):
    """所有技能的标准基类"""

    # ============ 元数据（子类必须覆写） ============
    name: str = ""                 # 技能名称，如 "code_review"
    description: str = ""          # 技能描述
    version: str = "0.1.0"         # 语义化版本号
    author: str = ""               # 作者
    triggers: list[str] = []       # 触发关键词（当用户输入包含这些词时自动匹配）
    changelog: list[Dict[str, str]] = []  # 更新日志
    # ============ 输入输出 Schema（子类可选覆写） ============
    # 用类变量定义，子类可以覆写为具体的 Pydantic 模型
    input_schema: type = None  # 例如: SomeInputModel
    output_schema: type = None  # 例如: SomeOutputModel

    def __init__(self):
        """初始化时自动校验元数据"""
        if not self.name:
            raise ValueError(f"技能必须定义 name，当前类: {self.__class__.__name__}")
        if not self.description:
            raise ValueError(f"技能必须定义 description，当前类: {self.__class__.__name__}")

    @abstractmethod
    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        执行技能核心逻辑（子类必须实现）

        推荐模式A：def execute(self, input_data: XxxInput) -> Dict[str, Any]
        兼容模式B：def execute(self, **kwargs) -> Dict[str, Any]
        """
        pass



    def validate_input(self, **kwargs) -> None:
        """验证输入参数（第一层容错：在错误发生前拦截）"""
        if self.input_schema is not None:
            try:
                self.input_schema(**kwargs)
            except Exception as e:
                raise ValueError(f"[{self.name}] 输入验证失败: {e}")

    def validate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证输出结果"""
        if self.output_schema is not None:
            try:
                self.output_schema(**result)
            except Exception as e:
                raise ValueError(f"[{self.name}] 输出验证失败: {e}")
        return result

    def _execute_with_logging(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        带日志和指标记录的执行包装器（可选使用）
        如果基础设施不可用，则回退到原始 execute 方法
        """
        if not INFRASTRUCTURE_AVAILABLE:
            # 如果基础设施不可用，直接调用原方法
            return self.execute(input_data, **kwargs)

        try:
            # 设置技能名称上下文
            set_skill_name(self.name)

            # 如果没有trace_id，则生成一个新的
            if not get_trace_id():
                set_trace_id()

            trace_id = get_trace_id()
            start_time = time.perf_counter()

            # 记录技能开始事件
            log_info("skill.start", f"Skill {self.name} started",
                     {"input_data": str(input_data)[:200]})

            # 执行原始的技能逻辑
            result = self.execute(input_data, **kwargs)

            # 计算耗时
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # 记录技能成功结束事件
            log_info("skill.end", f"Skill {self.name} completed successfully",
                     {"result_preview": str(result)[:200]}, elapsed_ms)

            # 记录技能指标
            record_skill_metric(
                skill_name=self.name,
                trace_id=trace_id,
                status="success",
                elapsed_ms=elapsed_ms
            )

            return result

        except Exception as e:
            # 计算耗时
            elapsed_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else 0

            # 记录技能错误事件
            log_error("skill.error", f"Skill {self.name} failed with error: {str(e)}",
                      {"error_type": type(e).__name__, "error_message": str(e)}, elapsed_ms)

            # 记录技能指标（失败情况）
            record_skill_metric(
                skill_name=self.name,
                trace_id=trace_id if 'trace_id' in locals() else "",
                status="error",
                elapsed_ms=elapsed_ms,
                error_type=type(e).__name__
            )

            # 重新抛出异常
            raise
