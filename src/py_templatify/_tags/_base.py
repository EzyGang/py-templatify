import sys
from collections.abc import Sequence
from typing import cast


if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

from py_templatify._tags._types import SupportsBool, SupportsIter


class Unset:
    def __bool__(self) -> bool:
        return False


UNSET = Unset()


def is_unset(val: object | Unset) -> TypeIs[Unset]:
    return isinstance(val, Unset) or val == UNSET


class TagBase[T]:
    pre: str = ''
    post: str = ''

    def __init__(self, val: T | Unset = UNSET, pre: str | Unset = UNSET, post: str | Unset = UNSET) -> None:
        self.pre = pre if not is_unset(pre) else self.pre
        self.post = post if not is_unset(post) else self.post
        self._val: T | Unset = val

    def __call__(self, val: T | Unset = UNSET, pre: str | Unset = UNSET, post: str | Unset = UNSET) -> str:
        self._val = val if not is_unset(val) else self._val
        self.pre = pre if not is_unset(pre) else self.pre
        self.post = post if not is_unset(post) else self.post

        return str(self)

    def __repr__(self) -> str:
        if is_unset(self._val):
            return super().__repr__()

        return str(self)

    def __str__(self) -> str:
        if is_unset(self._val):
            raise ValueError('Value must be set')

        return f'{self.pre}{self._val}{self.post}'


class IterableTagBase[T: SupportsIter](TagBase[Sequence[T]]):
    elem_tag: type[TagBase[T]] = TagBase[T]

    pre_elem: str = ''
    post_elem: str = ''

    def __init__(
        self,
        val: Sequence[T] | Unset = UNSET,
        pre: str | Unset = UNSET,
        post: str | Unset = UNSET,
        pre_elem: str | Unset = UNSET,
        post_elem: str | Unset = UNSET,
    ) -> None:
        super().__init__(val=val, pre=pre, post=post)
        self.pre_elem = pre_elem if not is_unset(pre_elem) else self.pre_elem
        self.post_elem = post_elem if not is_unset(post_elem) else self.post_elem

    def __call__(
        self,
        val: Sequence[T] | Unset = UNSET,
        pre: str | Unset = UNSET,
        post: str | Unset = UNSET,
        pre_elem: str | Unset = UNSET,
        post_elem: str | Unset = UNSET,
    ) -> str:
        super().__call__(val=val, pre=pre, post=post)
        self.pre_elem = pre_elem if not is_unset(pre_elem) else self.pre_elem
        self.post_elem = post_elem if not is_unset(post_elem) else self.post_elem

        return str(self)

    def __str__(self) -> str:
        if is_unset(self._val):
            raise ValueError('Value must be set')

        elems = [str(self.elem_tag(elem, pre=self.pre_elem, post=self.post_elem)) for elem in self._val]
        return f'{self.pre}{"".join(elems)}{self.post}'


class Option[T]:
    if_none: str = ''
    resume: bool = False

    def __init__(self, val: T | None | Unset = UNSET, /, if_none: str = '', resume: bool = False) -> None:
        self._val: T | None | Unset = val
        self.if_none = if_none or self.if_none
        self.resume = resume

    @property
    def is_empty(self) -> bool:
        return True if is_unset(self._val) or self._val is None else False

    def __call__(self, val: T | Unset | None = UNSET) -> T | str:
        self._val = val

        return self.if_none if self.is_empty else cast(T, self._val)


class Boolean[T: SupportsBool](TagBase[T]):
    if_true: str = '+'
    if_false: str = '-'

    def __init__(
        self,
        val: T | Unset = UNSET,
        pre: str | Unset = UNSET,
        post: str | Unset = UNSET,
        /,
        if_true: Unset | str = UNSET,
        if_false: Unset | str = UNSET,
    ) -> None:
        super().__init__(val, pre, post)
        self.if_true = if_true if not is_unset(if_true) else self.if_true
        self.if_false = if_false if not is_unset(if_false) else self.if_false

    def __str__(self) -> str:
        return self.if_true if bool(self._val) else self.if_false
