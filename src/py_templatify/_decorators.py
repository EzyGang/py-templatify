import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from py_templatify._types import Wrapped


class templatify[CTX]:
    def __init__(self, description: str | None = None, escape_symbols: str | None = None) -> None:
        self._description = description
        self._escape_symbols = escape_symbols

    def __call__[**_P, _R](
        self,
        _func: Callable[_P, _R],
    ) -> Wrapped[_P, CTX]:
        signature = self._get_typed_signature(_func)

        if _func.__doc__ is None:
            raise RuntimeError('Template string is not provided')

        wrapped = Wrapped[_P, CTX](func=_func, escape=self._escape_symbols, tpl=_func.__doc__, signature=signature)
        wrapped.__doc__ = self._description

        return wrapped

    def _get_typed_signature(self, _func: Callable[..., Any]) -> inspect.Signature:
        signature = inspect.signature(_func)
        type_hints = get_type_hints(_func, include_extras=True)
        typed_params = [
            inspect.Parameter(
                name=param.name,
                kind=param.kind,
                default=param.default,
                annotation=type_hints.get(param.name, Any),
            )
            for param in signature.parameters.values()
        ]

        return inspect.Signature(typed_params)
