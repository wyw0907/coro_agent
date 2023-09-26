
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain.pydantic_v1 import (
    BaseModel,
)
from langchain.tools.base import StructuredTool, create_schema_from_function

class StructuredToolWithCtx(StructuredTool):

    context_key: str = 'context'

    @property
    def args(self) -> dict:
        """The tool's input arguments without context_key"""

        if not self.context_key:
            raise ValueError('context_key must set in StructuredToolWithCtx')

        args_dict = self.args_schema.schema()["properties"]
        return {k:v for k, v in args_dict.items() if k != self.context_key}

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        context_key: str = 'context',
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ):

        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            raise ValueError("Function and/or coroutine must be provided")
        name = name or source_function.__name__
        description = description or source_function.__doc__
        if description is None:
            raise ValueError(
                "Function must have a docstring if description not provided."
            )

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        sig = signature(source_function)
        description = f"{name}{sig} - {description.strip()}"
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", source_function)
        return cls(
            name=name,
            context_key=context_key,
            func=func,
            coroutine=coroutine,
            args_schema=_args_schema,
            description=description,
            return_direct=return_direct,
            **kwargs,
        )