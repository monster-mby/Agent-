"""
第三层容错：熔断与降级机制。
核心定位
第三层容错：防止系统雪崩、故障隔离、自动降级
专门解决：某个技能 / 接口彻底挂了、一直报错 的场景。
一、核心三种状态（最关键）
CLOSED 关闭（正常）
一切正常，所有请求正常放行，随便调用技能。
OPEN 熔断（断路）
某个技能连续失败次数达标（比如连续 5 次报错）
→ 直接禁止调用这个技能
→ 不再发送请求，避免疯狂报错、卡死、拖垮整个 Agent。
HALF_OPEN 半开（试探恢复）
熔断冷却时间到了（比如 30 秒）
→ 放少量试探请求进去试试
试探成功 → 自动恢复正常（关闭熔断）
试探失败 → 立刻再次熔断
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"          # 正常状态
    OPEN = "open"              # 熔断状态：连续失败太多，直接禁止请求
    HALF_OPEN = "half_open"    # 半开状态（试探）：熔断冷却后，放少量请求试试水。


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5           # 连续失败多少次触发熔断（默认 5 次）
    recovery_timeout: float = 30.0       # 熔断后等多久进入半开状态（默认 30 秒）。
    half_open_max_calls: int = 3         # 半开状态最多放几个试探请求（默认 3 个）。
    monitored_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
        ValueError,
    )                                   # 哪些异常算 “失败”（比如网络超时、连接错误）。
    fallback_return: Any = None          # 熔断时的降级返回值（不抛错，直接返回这个，保证程序不崩）。
    name: str = "default"                # 熔断器名称（用于打日志，区分不同技能）。


class CircuitBreaker:
    """
    熔断器。

    用法：
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        @breaker.protect
        def risky_function():
            ...

        # 手动控制
        if breaker.allow_request():
            try:
                result = risky_function()
                breaker.on_success()
            except Exception:
                breaker.on_failure()
    """


    #创建一个熔断器实例，初始化所有内部状态和计数器。
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    # ==================== 属性 ====================

    @property
    def state(self) -> CircuitState:
        """当前状态"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"🔓 [{self.config.name}] 熔断器进入半开状态")
        return self._state

    @property
    def is_available(self) -> bool:
        """熔断器是否可用（允许请求通过）"""
        return self.state != CircuitState.OPEN

    @property
    def failure_rate(self) -> float:
        """
        计算历史总失败率（总失败数 / 总请求数）
        """
        total = self._total_failures + self._total_successes
        if total == 0:
            return 0.0
        return self._total_failures / total

    @property
    def failure_count(self) -> int:
        """当前连续失败次数"""
        return self._failure_count


    # ==================== 核心方法 ====================

    def allow_request(self) -> bool:
        """
        allow_request (是否允许请求通过)
        做什么：
        这是你手动控制熔断器时用的核心方法。
        根据当前状态，决定放不放行这个请求：
        CLOSED：直接放行。
        OPEN：直接拒绝。
        HALF_OPEN：只放行前 N 个试探请求（N 由 half_open_max_calls 定义）。
        输入：
        无。
        输出：
        bool：True 允许通过，False 拒绝。
        """
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            logger.warning(f"⛔ [{self.config.name}] 熔断器开启，请求被拒绝")
            return False

        # HALF_OPEN: 限制试探调用数
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                logger.info(f"🔍 [{self.config.name}] 半开状态，允许试探调用 ({self._half_open_calls}/{self.config.half_open_max_calls})")
                return True
            return False

        return False

    def on_success(self) -> None:
        """
        做什么：
        请求成功后必须调用这个方法。
        逻辑：
        如果是 HALF_OPEN 状态且成功了 → 太好了，恢复正常！ 切换回 CLOSED，清空所有失败计数。
        如果是 CLOSED 状态 → 重置连续失败计数。
        """
        self._total_successes += 1

        current_state = self.state  # 使用属性以触发状态自动转换

        if current_state == CircuitState.HALF_OPEN:
            # 半开状态成功 → 关闭熔断器
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info(f"🔒 [{self.config.name}] 熔断器恢复关闭状态")
        elif current_state == CircuitState.CLOSED:
            # 正常状态，重置失败计数
            self._failure_count = 0

    def on_failure(self) -> None:
        """
        做什么：
        请求失败后必须调用这个方法。
        逻辑：
        增加连续失败计数，记录失败时间。
        如果是 CLOSED 状态且连续失败达标 → 触发熔断！ 切换到 OPEN。
        如果是 HALF_OPEN 状态且失败了 → 试探失败，继续熔断！ 切回 OPEN，重置冷却时间。
        """
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"🔥 [{self.config.name}] 熔断器开启！"
                    f"连续失败 {self._failure_count}/{self.config.failure_threshold}"
                )
        elif self._state == CircuitState.HALF_OPEN:
            # 半开状态失败 → 重新熔断
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            logger.warning(f"🔥 [{self.config.name}] 半开状态试探失败，重新熔断")

    # ==================== 装饰器 ====================

    def protect(self, func: Callable) -> Callable:
        """
        做什么：
        这是最常用的方式，一行代码给函数加上自动熔断。
        它会自动：
        检查 allow_request()。
        如果熔断中 → 直接返回 fallback_return（降级值），不执行原函数。
        如果放行 → 执行原函数。
        成功 → 自动调用 on_success()。
        失败（且是监控的异常）→ 自动调用 on_failure()，然后抛出异常。
        输入：
        func：你要加保护的函数。
        输出：
        wrapper：包装后的新函数（包含自动熔断逻辑）。

        用法：
            @breaker.protect
            def my_func():
                ...
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not self.allow_request():
                # 熔断中，返回降级值
                logger.warning(f"⛔ [{self.config.name}] 熔断中，使用降级返回值")
                if isinstance(self.config.fallback_return, Exception):
                    raise self.config.fallback_return
                return self.config.fallback_return

            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except self.config.monitored_exceptions as e:
                self.on_failure()
                raise e

        return wrapper

    # ==================== 统计 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取熔断器统计信息
        做什么：
        返回一个字典，包含熔断器的所有状态数据（当前状态、失败次数、失败率、是否可用等）。
        用于监控、调试、打日志。
        输入：
        无。
        输出：
        Dict[str, Any]：包含所有统计信息的字典。

        """
        return {
            "name": self.config.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.config.failure_threshold,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "failure_rate": f"{self.failure_rate:.1%}",
            "last_failure_time": self._last_failure_time,
            "is_available": self.is_available,
        }

    def reset(self) -> None:
        """手动重置熔断器"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        logger.info(f"🔄 [{self.config.name}] 熔断器已手动重置")

    def __repr__(self) -> str:
        """
        定义打印熔断器对象时显示的内容，方便调试。
        """
        return (
            f"CircuitBreaker(name='{self.config.name}', "
            f"state={self._state.value}, "
            f"failures={self._failure_count}/{self.config.failure_threshold})"
        )


# 便捷装饰器工厂函数
def circuit_breaker(config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    创建并返回一个熔断器实例（可直接作为装饰器使用）。

    用法：
        @circuit_breaker(CircuitBreakerConfig(failure_threshold=3))
        def risky_function():
            ...
    """
    breaker = CircuitBreaker(config)
    return breaker