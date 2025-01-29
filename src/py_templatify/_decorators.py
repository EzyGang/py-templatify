import inspect
import re
import sys
from collections.abc import Callable
from copy import copy
from typing import Annotated, Any, Protocol, Self, get_origin, get_type_hints


if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

from py_templatify._tags._base import Option, TagBase


_regex = re.compile(r'{(.+\..+)}')


class WrappedProto[**_PS, CTX](Protocol):
    __tpl: str
    __signature: inspect.Signature
    __templatify: 'templatify[CTX]'
    __ctx: CTX | None
    __wrapped__: Callable[_PS, str]

    def ctx(self: Self, context: CTX) -> Self: ...

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str: ...


def is_option(v: object) -> TypeIs[type[Option[Any]] | Option[Any]]:
    return isinstance(v, Option) or (inspect.isclass(v) and issubclass(v, Option))


def is_tag(v: object) -> TypeIs[type[TagBase[Any]] | TagBase[Any]]:
    return isinstance(v, TagBase) or (inspect.isclass(v) and issubclass(v, TagBase))


class Wrapped[**_PS, CTX](WrappedProto[_PS, CTX]):
    __wrapped__: Callable[_PS, str]

    def __init__(self, templatify: 'templatify[CTX]', signature: inspect.Signature, func: Callable[_PS, Any], tpl: str):
        self.__ctx: CTX | None = None
        self.__templatify = templatify

        self.__tpl = tpl
        self.__signature = signature
        self.__func = func

        self.__used_attributes = re.findall(_regex, self.__tpl)

    def ctx(self: Self, context: CTX) -> Self:
        self.__ctx = context
        return self

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str:
        arguments, kwd_args = self._get_format_kwargs(bound_args=self.__signature.bind(*args, **kwargs))
        _tpl = copy(self.__tpl)

        if self.__used_attributes:
            param_values = self._get_parameter_values_from_objs_for_fields(kwd_args=kwd_args)
            for field, (param, value) in param_values.items():
                if not param:
                    continue

                annotation = self._get_annotation_from_parameter(parameter=param)
                kwd_args[param.name] = (
                    self._get_parameter_value_after_transforms(
                        value=value, annotation=self._get_annotation_from_parameter(parameter=param)
                    )
                    if annotation
                    else value
                )

                _tpl = _tpl.replace(f'{{{field}}}', f'{{{param.name}}}')

        self._update_kwd_args_with_annotations(kwd_args=kwd_args)

        return _tpl.format(*arguments, **kwd_args)

    def _update_kwd_args_with_annotations(self, kwd_args: dict[str, Any]) -> None:
        for kwd, value in kwd_args.items():
            parameter = self.__signature.parameters.get(kwd, None)
            if not parameter:
                continue

            annotation = self._get_annotation_from_parameter(parameter=parameter)
            if not annotation:
                continue

            kwd_args[kwd] = self._get_parameter_value_after_transforms(value=value, annotation=annotation)

    def _get_parameter_value_after_transforms(self, value: Any, annotation: Any) -> Any:
        new_value: Any = value
        for meta in annotation.__metadata__:
            if not (inspect.isfunction(meta) or is_option(meta) or is_tag(meta)):
                continue

            if is_option(meta):
                _opt_instance = meta() if not isinstance(meta, Option) else meta
                new_value = _opt_instance(value)

                is_do_break = _opt_instance.is_empty and not _opt_instance.resume

                if is_do_break:
                    break

                continue

            new_value = meta(new_value)

            # If it is still an instance of a TagBase,
            # then type annotation was of a class type and not an instance or a function
            if isinstance(new_value, TagBase):
                new_value = new_value()

        return new_value

    def _get_annotation_from_parameter(self, parameter: inspect.Parameter) -> Any | None:
        # handle type alias annotation
        type_alias_origin = self._get_type_alias_origin(parameter.annotation)
        if type_alias_origin is not None:
            return type_alias_origin

        # handle annotated as is
        if get_origin(annotation := parameter.annotation) is Annotated:
            return annotation

        return None

    @staticmethod
    def _get_type_alias_origin(param_annotation: Any) -> None | Any:
        try:
            return alias_original if get_origin(alias_original := param_annotation.__value__) is Annotated else None
        except Exception:
            return None

    def _get_format_kwargs(self, bound_args: inspect.BoundArguments) -> tuple[tuple[Any, ...], dict[str, Any]]:
        bound_args.apply_defaults()
        args_dict = bound_args.arguments
        args: tuple[Any, ...] = args_dict.pop('args', ())
        kwargs: dict[str, Any] = args_dict.pop('kwargs', {})
        kwargs.update(args_dict)

        return args, kwargs

    def _get_parameter_values_from_objs_for_fields(
        self, kwd_args: dict[str, Any]
    ) -> dict[str, tuple[inspect.Parameter | None, Any | None]]:
        annotations: dict[str, tuple[inspect.Parameter | None, Any | None]] = {}

        # Going through all the attribute accesses that are used in the template
        for field in self.__used_attributes:
            if '.' not in field:
                continue

            parts = field.split('.')
            obj = kwd_args.get(parts[0])
            if not obj:
                annotations[field] = (None, None)
                continue

            current_annotation = obj.__class__
            current_val = obj
            for part in parts[1:]:
                current_val = getattr(current_val, part, None)
                current_annotation = current_annotation.__annotations__.get(part)

            # At this point, current_annotation should be the type we want
            param = inspect.Parameter(
                name=field.replace('.', '_'),
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=current_annotation,
            )
            annotations[field] = (param, current_val)

        return annotations


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
