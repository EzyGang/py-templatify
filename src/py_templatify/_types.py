import inspect
import logging
import random
import re
import string
import sys
from collections.abc import Callable, Iterable
from copy import copy
from functools import partial
from typing import Annotated, Any, Protocol, get_origin

from py_templatify._tags._base import UNSET, Option, TagBase


if sys.version_info >= (3, 13):
    from typing import TypeIs  # pragma: no cover
else:
    from typing_extensions import TypeIs  # pragma: no cover


logger = logging.getLogger('py-templatify')
_regex = re.compile(r'{(.+?\..+?)}')
_placehold_regex = re.compile(r'(\{[^\}]+?\})')


class WrappedProto[**_PS, CTX](Protocol):
    _tpl: str
    _signature: inspect.Signature
    _ctx: CTX | None
    __wrapped__: Callable[_PS, str]

    # def ctx(self: Self, context: CTX) -> Self: ...  # pragma: no cover

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str: ...  # pragma: no cover


def is_option(v: object) -> TypeIs[type[Option[Any]] | Option[Any]]:
    return isinstance(v, Option) or (inspect.isclass(v) and issubclass(v, Option))


def is_tag(v: object) -> TypeIs[type[TagBase[Any]] | TagBase[Any]]:
    return isinstance(v, TagBase) or (inspect.isclass(v) and issubclass(v, TagBase))


class Wrapped[**_PS, CTX](WrappedProto[_PS, CTX]):
    __wrapped__: Callable[_PS, str]

    def __init__(self, signature: inspect.Signature, escape: str | None, func: Callable[_PS, Any], tpl: str):
        self._ctx: CTX | None = None

        self._tpl = tpl
        self._signature = signature
        self._func = func

        self._escape_symbols = set(escape) if escape else set()
        self._rand_from = list(set(string.punctuation + string.ascii_letters) - self._escape_symbols)
        self._escape_regex = re.compile(f'({"|".join(f"\\{symbol}" for symbol in self._escape_symbols)})')
        self._escape_func = staticmethod(self._escape_func_factory())

        self._used_attributes = re.findall(_regex, self._tpl)

    # def ctx(self: Self, context: CTX) -> Self:
    #     self._ctx = context
    #     return self

    def __call__(self, *args: _PS.args, **kwargs: _PS.kwargs) -> str:
        arguments, kwd_args = self._get_format_kwargs(bound_args=self._signature.bind(*args, **kwargs))

        _tpl = self._escape_tpl()
        _tpl = self._update_kwd_args_from_attributes(kwd_args=kwd_args, tpl=_tpl)
        self._update_kwd_args_with_annotations(kwd_args=kwd_args)

        return _tpl.format(*arguments, **kwd_args)

    def _update_kwd_args_from_attributes(self, kwd_args: dict[str, Any], tpl: str) -> str:
        if not self._used_attributes:
            return tpl

        param_values = self._get_parameter_values_from_objs_for_fields(kwd_args=kwd_args)
        for field, (param, value) in param_values.items():
            if not param:
                continue

            annotation = self._get_annotation_from_parameter(parameter=param)
            kwd_args[param.name] = self._get_parameter_value_after_transforms(value=value, annotation=annotation)

            tpl = tpl.replace(f'{{{field}}}', f'{{{param.name}}}')

        return tpl

    def _update_kwd_args_with_annotations(self, kwd_args: dict[str, Any]) -> None:
        for kwd, value in kwd_args.items():
            parameter = self._signature.parameters.get(kwd, None)
            if not parameter:
                continue

            annotation = self._get_annotation_from_parameter(parameter=parameter)
            kwd_args[kwd] = self._get_parameter_value_after_transforms(value=value, annotation=annotation)

    def _get_parameter_value_after_transforms(self, value: Any, annotation: Any | None) -> Any:
        new_value: Any = value

        _is_escaped = False
        if annotation is not None and annotation.__metadata__:
            _is_escaped, new_value = self._process_annotation_metadata(
                new_value=new_value, metadata=annotation.__metadata__
            )

        if not _is_escaped:
            new_value = self._escape_func(str(new_value))

        return new_value

    def _process_annotation_metadata(self, new_value: Any, metadata: Iterable[Any]) -> tuple[bool, Any]:
        _is_escaped = False
        for meta in metadata:
            escape_func = self._escape_func if not _is_escaped else UNSET
            if not (inspect.isfunction(meta) or is_option(meta) or is_tag(meta)):
                continue

            if is_option(meta):
                _opt_instance = meta() if not isinstance(meta, Option) else meta
                new_value = _opt_instance(new_value, escape=escape_func)
                _is_escaped = True

                if _opt_instance.is_empty and not _opt_instance.resume:
                    break

                continue

            if is_tag(meta):
                _tag_instance = meta() if not isinstance(meta, TagBase) else meta
                new_value = _tag_instance(new_value, escape=escape_func)

                _is_escaped = True
                continue

            new_value = meta(new_value)

        return _is_escaped, new_value

    def _get_annotation_from_parameter(self, parameter: inspect.Parameter) -> Any | None:
        # handle type alias annotation
        type_alias_origin = self._get_type_alias_origin(parameter.annotation)
        if type_alias_origin is not None:
            return type_alias_origin

        # handle annotated as is
        if get_origin(annotation := parameter.annotation) is Annotated:
            return annotation

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
        for field in self._used_attributes:
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

    def _escape_func_factory(self) -> Callable[[str], str]:
        def _escape_str(s: str, regex: re.Pattern[str]) -> str:
            return regex.sub(r'\\\1', s)

        if not self._escape_symbols:
            return lambda x: str(x)

        return partial(_escape_str, regex=self._escape_regex)

    def _escape_tpl(self) -> str:
        _tpl = copy(self._tpl)
        if not self._escape_symbols:
            return _tpl

        _placeholders: dict[str, int] = {}
        for i, match in enumerate(_placehold_regex.finditer(_tpl), start=0):
            _placeholders[match.group(0)] = i

        _prefix = ''.join(list(random.sample(self._rand_from, 3)))
        for placeholder, i in _placeholders.items():
            _tpl = _tpl.replace(placeholder, f'{_prefix}{i}')

        _tpl = self._escape_func(_tpl)
        for placeholder, i in _placeholders.items():
            _tpl = _tpl.replace(f'{_prefix}{i}', placeholder)

        return _tpl

    @staticmethod
    def _get_type_alias_origin(param_annotation: Any) -> None | Any:
        try:
            return alias_original if get_origin(alias_original := param_annotation.__value__) is Annotated else None
        except Exception:
            return None
