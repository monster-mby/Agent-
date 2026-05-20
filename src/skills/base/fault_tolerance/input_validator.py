"""
第一层容错：输入输出验证。
第一层容错：事前防御，在代码执行前就拦截无效 / 错误的输入输出，从源头掐灭崩溃风险，是整个系统的第一道安全闸门。
具体干了什么事
输入校验：基于你定义的 Pydantic 数据模型，严格检查传入参数的类型、格式、必填项、取值范围是否合规。
比如你的code_review技能，要求必须传code参数、language必须是预设的几种语言，不符合规则直接拦截，根本不会让技能执行。
输出校验：技能执行完后，检查返回结果是否符合约定的格式，避免脏数据、畸形结果流向下游的 LLM 或前端，引发连锁错误。
友好报错：把 Pydantic 原生的复杂报错，转换成人能一眼看懂的清晰提示，直接告诉你「哪个字段错了、为什么错」，而不是抛一堆看不懂的堆栈日志。
便捷接入：支持装饰器模式，一行代码就能给任意函数加上校验，不用改业务逻辑。
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class InputValidator:
    """
    输入验证器。

    用法：
        validator = InputValidator(SchemaModel)
        data = validator.validate(name="test", value=123)
    """

    def __init__(self, schema_model: Optional[Type[BaseModel]] = None):
        """
        Args:
            schema_model: Pydantic BaseModel 类，用于验证输入
        """
        self.schema_model = schema_model

    def validate(self, **kwargs) -> Dict[str, Any]:
        """
        验证输入参数。

        Args:
            **kwargs: 要验证的参数

        Returns:
            验证通过后的参数字典

        Raises:
            ValueError: 验证失败时抛出，包含详细错误信息
        """
        if self.schema_model is None:
            return kwargs

        try:
            validated = self.schema_model(**kwargs)
            return validated.model_dump()#将验证后的数据转换为 Python 字典并返回
        except ValidationError as e:#抛出 ValidationError 异常，这里将其捕获并赋值给 e
            # 格式化错误信息，方便定位具体字段
            error_details = []
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                msg = err["msg"]
                error_details.append(f"  - {field}: {msg}")

            error_msg = (
                f"❌ 输入验证失败:\n"
                + "\n".join(error_details)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_output(self, data: Dict[str, Any], schema_model: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """
        验证输出结果。

        Args:
            data: 输出数据字典
            schema_model: 可选的输出 schema，默认使用输入 schema

        Returns:
            验证通过后的输出字典

        Raises:
            ValueError: 验证失败时抛出
        """
        model = schema_model or self.schema_model
        if model is None:
            return data

        try:
            validated = model(**data)
            return validated.model_dump()
        except ValidationError as e:
            error_details = []
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                msg = err["msg"]
                error_details.append(f"  - {field}: {msg}")

            error_msg = (
                f"❌ 输出验证失败:\n"
                + "\n".join(error_details)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def decorate(self, func: Callable) -> Callable:
        """
        装饰器模式：自动验证函数参数。

        用法：
            @validator.decorate
            def my_func(name: str, value: int):
                ...
        """

        def wrapper(*args, **kwargs) -> Any:
            # 获取函数签名
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 验证
            validated = self.validate(**bound.arguments)

            # 只保留函数签名中定义的参数
            filtered_kwargs = {
                key: validated[key]
                for key in sig.parameters.keys()
                if key in validated
            }

            return func(*args, **filtered_kwargs)

        return wrapper

