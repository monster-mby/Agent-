"""
测试三层容错架构
"""

import time
import pytest
from src.skills.base.fault_tolerance import (
    InputValidator,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from pydantic import BaseModel, Field


# ==================== 测试 InputValidator ====================

class TestSchema(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(default="test@example.com")


class TestInputValidator:

    def setup_method(self):
        self.validator = InputValidator(TestSchema)

    def test_valid_input(self):
        """测试有效输入"""
        result = self.validator.validate(name="张三", age=25)
        assert result["name"] == "张三"
        assert result["age"] == 25

    def test_invalid_input_missing_field(self):
        """测试缺少必填字段"""
        with pytest.raises(ValueError, match="输入验证失败"):
            self.validator.validate(age=25)

    def test_invalid_input_wrong_type(self):
        """测试类型错误"""
        with pytest.raises(ValueError, match="输入验证失败"):
            self.validator.validate(name="张三", age="二十五")

    def test_invalid_input_out_of_range(self):
        """测试范围越界"""
        with pytest.raises(ValueError, match="输入验证失败"):
            self.validator.validate(name="张三", age=200)

    def test_no_schema(self):
        """测试无 schema 时不验证"""
        v = InputValidator()
        result = v.validate(anything="goes")
        assert result == {"anything": "goes"}

    def test_validate_output_valid(self):
        """测试有效输出验证"""
        data = {"name": "李四", "age": 30}
        result = self.validator.validate_output(data)
        assert result["name"] == "李四"

    def test_validate_output_invalid(self):
        """测试无效输出验证"""
        with pytest.raises(ValueError, match="输出验证失败"):
            self.validator.validate_output({"name": "李四", "age": -1})

    def test_decorator(self):
        """测试装饰器模式"""
        @self.validator.decorate
        def greet(name: str, age: int):
            return f"{name} is {age} years old"

        result = greet(name="测试", age=20)
        assert result == "测试 is 20 years old"

        with pytest.raises(ValueError):
            greet(name="", age=20)


# ==================== 测试重试装饰器 ====================

class TestRetry:

    def test_retry_success_on_first_try(self):
        """测试一次成功"""
        call_count = 0

        from src.skills.base.fault_tolerance import retry

        @retry(RetryConfig(max_retries=3))
        def always_success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = always_success()
        assert result == "ok"
        assert call_count == 1

    def test_retry_eventually_success(self):
        """测试重试后成功"""
        call_count = 0

        from src.skills.base.fault_tolerance import retry

        @retry(RetryConfig(max_retries=3, base_delay=0.01))
        def eventually_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temp error")
            return "success"

        result = eventually_success()
        assert result == "success"
        assert call_count == 3

    def test_retry_all_fail(self):
        """测试全部失败"""
        call_count = 0

        from src.skills.base.fault_tolerance import retry

        @retry(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fail")

        with pytest.raises(RuntimeError, match="执行失败"):
            always_fail()
        assert call_count == 3  # 初始 + 2次重试

    def test_retry_non_retryable_exception(self):
        """测试不可重试异常"""
        from src.skills.base.fault_tolerance import retry

        @retry(RetryConfig(max_retries=3))
        def raises_key_error():
            raise KeyError("not retryable")

        with pytest.raises(KeyError):
            raises_key_error()

    def test_retry_with_callback(self):
        """测试重试回调"""
        callback_called = False

        def on_retry(attempt, delay, exc):
            nonlocal callback_called
            callback_called = True

        from src.skills.base.fault_tolerance import retry

        call_count = 0

        @retry(RetryConfig(
            max_retries=1,
            base_delay=0.01,
            on_retry_callback=on_retry
        ))
        def fail_once():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(RuntimeError):
            fail_once()
        assert callback_called


# ==================== 测试 CircuitBreaker ====================

class TestCircuitBreaker:

    def setup_method(self):
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.1,  # 快速恢复用于测试
            half_open_max_calls=2,
            name="test_breaker"
        )
        self.breaker = CircuitBreaker(self.config)

    def test_initial_state_closed(self):
        """测试初始状态为关闭"""
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.is_available is True

    def test_open_after_failures(self):
        """测试达到阈值后熔断"""
        for _ in range(3):
            self.breaker.on_failure()
        assert self.breaker.state == CircuitState.OPEN
        assert self.breaker.is_available is False

    def test_allow_request_when_closed(self):
        """测试关闭时允许请求"""
        assert self.breaker.allow_request() is True

    def test_deny_request_when_open(self):
        """测试熔断时拒绝请求"""
        for _ in range(3):
            self.breaker.on_failure()
        assert self.breaker.allow_request() is False

    def test_half_open_after_timeout(self):
        """测试超时后进入半开状态"""
        for _ in range(3):
            self.breaker.on_failure()
        assert self.breaker.state == CircuitState.OPEN

        time.sleep(0.15)  # 等待超时
        assert self.breaker.state == CircuitState.HALF_OPEN
        assert self.breaker.allow_request() is True

    def test_close_after_half_open_success(self):
        """测试半开成功后关闭"""
        for _ in range(3):
            self.breaker.on_failure()
        time.sleep(0.15)

        # 半开状态，成功
        self.breaker.on_success()
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.is_available is True

    def test_reopen_after_half_open_failure(self):
        """测试半开失败后重新熔断"""
        for _ in range(3):
            self.breaker.on_failure()
        time.sleep(0.15)

        # 半开状态，失败
        self.breaker.on_failure()
        assert self.breaker.state == CircuitState.OPEN

    def test_protect_decorator_success(self):
        """测试保护装饰器 - 成功"""
        call_count = 0

        @self.breaker.protect
        def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_func()
        assert result == "ok"
        assert call_count == 1

    def test_protect_decorator_fallback(self):
        """测试保护装饰器 - 熔断降级"""
        self.breaker.config.fallback_return = "fallback_value"

        @self.breaker.protect
        def fail_func():
            raise ConnectionError("fail")

        # 触发熔断
        for _ in range(3):
            try:
                fail_func()
            except ConnectionError:
                pass

        # 熔断后返回降级值
        result = fail_func()
        assert result == "fallback_value"

    def test_reset(self):
        """测试手动重置"""
        for _ in range(3):
            self.breaker.on_failure()
        assert self.breaker.state == CircuitState.OPEN

        self.breaker.reset()
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.failure_count == 0

    def test_stats(self):
        """测试统计信息"""
        stats = self.breaker.get_stats()
        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["failure_rate"] == "0.0%"

        self.breaker.on_failure()
        stats = self.breaker.get_stats()
        assert stats["total_failures"] == 1


# ==================== 集成测试 ====================

class TestFaultToleranceIntegration:

    def test_input_validator_plus_retry(self):
        """测试验证器 + 重试集成"""
        from src.skills.base.fault_tolerance import retry, RetryConfig
        from pydantic import BaseModel, Field

        class InputSchema(BaseModel):
            value: int = Field(..., ge=0)

        validator = InputValidator(InputSchema)
        call_count = 0

        @retry(RetryConfig(max_retries=2, base_delay=0.01))
        def process(value: int):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("temp error")
            return value * 2

        # 先验证，再执行
        validated = validator.validate(value=5)
        result = process(**validated)
        assert result == 10
        assert call_count == 2

    def test_full_pipeline(self):
        """测试完整容错管道：验证 → 重试 → 熔断"""
        from src.skills.base.fault_tolerance import retry, RetryConfig

        call_count = 0

        breaker_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            fallback_return={"error": "service unavailable"}
        )
        breaker = CircuitBreaker(breaker_config)

        @breaker.protect
        @retry(RetryConfig(max_retries=1, base_delay=0.01))
        def full_pipeline():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("service down")

        # 第一次：重试后失败
        for _ in range(2):
            try:
                full_pipeline()
            except (RuntimeError, ConnectionError):
                pass

        # 熔断后返回降级值
        result = full_pipeline()
        assert result == {"error": "service unavailable"}

if __name__ == "__main__":
    pytest.main()