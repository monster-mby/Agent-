"""
第二层容错：指数退避自动重试。
核心定位
第二层容错：事中自愈，针对偶发、临时的故障，自动重试恢复，不用人工干预，大幅提升系统稳定性，是系统的第二道补救防线。
具体干了什么事
自动重试：技能 / 接口执行失败时，不会直接抛出错误，而是按照配置自动重复执行，解决偶发故障。
指数退避：重试的等待时间会指数级递增（1s→2s→4s→8s），避免疯狂重试把下游服务（比如 LLM API）打崩。
抖动防雪崩：给等待时间加随机偏移，避免大量请求同时重试造成的「惊群效应」，进一步保护下游服务。
精准重试：只对「可重试的异常」生效（比如网络超时、连接错误、临时限流），像参数错误这种不可重试的问题，不会瞎试浪费资源。
全场景支持：同时支持同步函数和异步函数，适配你项目里的各种执行场景。

"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3              # 最大重试次数
    base_delay: float = 1.0           # 第一次重试等多久（默认 1 秒）
    max_delay: float = 60.0           # 最多等多久（防止无限变长，默认 60 秒）。
    exponential_base: float = 2.0     # 指数基数
    jitter: bool = True               # 是否加随机抖动（防止大量请求同时重试打崩服务，默认开）
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        ValueError,
        RuntimeError,
    )                                 #设定哪些异常才重试（比如网络超时、连接错误，参数错误这种不重试）。
    on_retry_callback: Optional[Callable] = None  # 每次重试前可以执行的自定义函数（可选）


def retry(
    config: Optional[RetryConfig] = None,
    name: Optional[str] = None,
) -> Callable:
    """
    做什么：
    这是一个装饰器工厂。
    你给它一个配置，它返回一个装饰器。
    把这个装饰器加在任何不稳定的函数上，函数失败了就会自动等一会儿 → 再试一次。
    输入：
    config：可选，一个 RetryConfig 对象（不传就用默认配置）。
    name：可选，操作名称（用于打日志，不传就用函数名）。
    输出：
    decorator：一个装饰器函数。
    重试装饰器。

    用法：
        @retry()
        def unstable_function():
            ...

        @retry(RetryConfig(max_retries=5))
        def another_function():
            ...

    Args:
        config: 重试配置，默认为 RetryConfig()
        name: 可选的操作名称（用于日志）
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        """
        做什么：
        包裹你的原函数。
        循环尝试执行原函数。
        成功了就直接返回结果。
        失败了：
        算一下要等多久（指数退避 + 抖动）。
        打日志告诉你 “第几次重试、等多久”。
        执行回调（如果有）。
        time.sleep(delay) 等待。
        重试次数用完还失败，就抛出 RuntimeError。
        输入：
        *args, **kwargs：原函数的所有参数。
        输出：
        成功：原函数的返回值。
        失败：抛出 RuntimeError（包含最后一次异常信息）。
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            operation_name = name or func.__name__

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        # 计算等待时间：指数退避 + 可选抖动
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        if config.jitter:
                            delay = delay * (0.5 + random.random() * 0.5)

                        logger.warning(
                            f"🔄 [{operation_name}] 第 {attempt + 1}/{config.max_retries} 次重试 "
                            f"（等待 {delay:.1f}s）: {e}"
                        )

                        # 回调
                        if config.on_retry_callback:
                            config.on_retry_callback(attempt, delay, e)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ [{operation_name}] 重试 {config.max_retries} 次后仍失败: {e}"
                        )

                except Exception as e:
                    # 非可重试异常，直接抛出
                    raise e

            # 所有重试都失败
            raise RuntimeError(
                f"[{operation_name}] 执行失败（已重试 {config.max_retries} 次）: {last_exception}"
            ) from last_exception

        return wrapper

    return decorator


class AsyncRetryMixin:
    """异步重试混入类
    做什么：
    专门给 async def 异步函数用的重试工具。
    逻辑和上面的 retry 装饰器完全一样，唯一区别是：
    把 time.sleep(delay) 换成了 await asyncio.sleep(delay)。
    不会阻塞事件循环。
    """

    @staticmethod
    async def retry_async(
        coro,
        config: Optional[RetryConfig] = None,
        name: Optional[str] = None,
    ):
        """异步重试执行
        做什么：
        异步版的重试执行器。
        接收一个协程对象（coroutine），自动重试。
        输入：
        coro：要执行的协程对象（比如 your_async_function()）。
        config：可选，RetryConfig 配置。
        name：可选，操作名称。
        输出：
        成功：协程的返回值。
        失败：抛出 RuntimeError。

        """
        if config is None:
            config = RetryConfig()

        last_exception = None
        operation_name = name or getattr(coro, "__name__", "unknown")

        for attempt in range(config.max_retries + 1):
            try:
                return await coro

            except config.retryable_exceptions as e:
                last_exception = e

                if attempt < config.max_retries:
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    if config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"🔄 [{operation_name}] 第 {attempt + 1}/{config.max_retries} 次重试 "
                        f"（等待 {delay:.1f}s）: {e}"
                    )

                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"[{operation_name}] 异步执行失败（已重试 {config.max_retries} 次）: {last_exception}"
        ) from last_exception