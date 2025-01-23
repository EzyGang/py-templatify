import inspect
from collections.abc import Callable
from typing import Annotated, Any, ParamSpec, Protocol, Self, TypeVar, get_origin, get_type_hints


# from pydantic import BaseModel


_PS = ParamSpec('_PS')
CTX = TypeVar('CTX')  # , bound=BaseModel)


class WrappedProto(Protocol[_PS, CTX]):
    __tpl: str
    __signature: inspect.Signature
    __templaterr: 'templatify[CTX]'
    __ctx: CTX | None
    __wrapped__: Callable[_PS, str]

    def ctx(self: Self, context: CTX) -> Self: ...

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str: ...


class Wrapped(WrappedProto[_PS, CTX]):
    __wrapped__: Callable[_PS, str]

    def __init__(self, templatify: 'templatify[CTX]', signature: inspect.Signature, func: Callable[_PS, Any], tpl: str):
        self.__ctx = None
        self.__templaterr = templatify
        self.__tpl = tpl
        self.__signature = signature
        self.__func = func

    def ctx(self: Self, context: CTX) -> Self:
        self._ctx = context
        return self

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str:
        arguments, kwd_args = self._get_format_kwargs(bound_args=self.__signature.bind(*args, **kwargs))

        for kwd, value in kwd_args.items():
            parameter = self.__signature.parameters.get(kwd, None)
            if not parameter or get_origin(annotation := parameter.annotation) is not Annotated:
                continue

            new_value = value
            for meta in annotation.__metadata__:
                if not callable(meta):
                    continue

                new_value = meta(value)

            kwd_args[kwd] = new_value

        print(f'{arguments=}, {kwd_args=}')
        return self.__tpl.format(*arguments, **kwd_args)

    def _get_format_kwargs(self, bound_args: inspect.BoundArguments) -> tuple[tuple[Any, ...], dict[str, Any]]:
        bound_args.apply_defaults()
        args_dict = bound_args.arguments
        args: tuple[Any, ...] = args_dict.pop('args', ())
        kwargs: dict[str, Any] = args_dict.pop('kwargs', {})
        kwargs.update(args_dict)

        return args, kwargs


class templatify[CTX]:
    def __init__(self, description: str | None = None) -> None:
        self._description = description

    def __call__[**_P, _R](
        self,
        _func: Callable[_P, _R],
    ) -> Wrapped[_P, CTX]:
        signature = self._get_typed_signature(_func)

        if _func.__doc__ is None:
            raise RuntimeError('Template string is missing')

        wrapped = Wrapped[_P, CTX](templatify=self, func=_func, tpl=_func.__doc__, signature=signature)
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
