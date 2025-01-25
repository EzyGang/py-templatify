from py_templatify._tags._base import UNSET, IterableTagBase, TagBase, Unset, is_unset
from py_templatify._tags._types import SupportsIter


class H1[T](TagBase[T]):
    pre = '# '


class H2[T](TagBase[T]):
    pre = '## '


class H3[T](TagBase[T]):
    pre = '### '


class Bold[T](TagBase[T]):
    pre = '**'
    post = '**'


class Italic[T](TagBase[T]):
    pre = '*'
    post = '*'


class Quote[T](TagBase[T]):
    pre = '> '


class Code[T](TagBase[T]):
    pre = '`'
    post = '`'


class LineAfter[T](TagBase[T]):
    post = '\n---'


class LineBefore[T](TagBase[T]):
    pre = '---\n'


class Strike[T](TagBase[T]):
    pre = '~~'
    post = '~~'


class Highlight[T](TagBase[T]):
    pre = '=='
    post = '=='


class Underline[T](TagBase[T]):
    pre = '__'
    post = '__'


class Spoiler[T](TagBase[T]):
    pre = '||'
    post = '||'


class CodeBlock[T](TagBase[T]):
    pre = '```'
    post = '\n```'

    def __init__(
        self, val: T | Unset = UNSET, pre: str | Unset = UNSET, post: str | Unset = UNSET, /, code: str = ''
    ) -> None:
        super().__init__(val, pre, post)
        self.code = code

    def __str__(self) -> str:
        if is_unset(self._val):
            raise ValueError('Value must be set')

        if self.code and '\n' not in self.code:
            self.code += '\n'

        return f'{self.pre}{self.code}{self._val}{self.post}'


class Link[T](TagBase[tuple[T, T]]):
    def __str__(self) -> str:
        if is_unset(self._val):
            raise ValueError('Value must be set')

        if len(self._val) != 2:
            raise ValueError('Tuple length should be 2')

        return f'[{self._val[0]}]({self._val[1]})'


class Image[T](Link[T]):
    def __str__(self) -> str:
        super().__str__()

        return f'![{self._val[0]}]({self._val[1]})'  # type: ignore[index]


class DotList[T: SupportsIter](IterableTagBase[T]):
    pre_elem: str = '- '
    post_elem: str = '\n'


class OrderedList[T: SupportsIter](IterableTagBase[T]):
    post_elem: str = '\n'

    def __str__(self) -> str:
        if is_unset(self._val):
            raise ValueError('Value must be set')

        elems = [
            str(self.elem_tag(elem, pre=f'{i}. ', post=self.post_elem)) for i, elem in enumerate(self._val, start=1)
        ]
        return f'{self.pre}{"".join(elems)}{self.post}'
