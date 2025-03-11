from ._decorators import templatify as templatify
from ._tags._base import Boolean as Boolean
from ._tags._base import IterableTagBase as IterableTagBase
from ._tags._base import Option as Option
from ._tags._base import TagBase as TagBase
from . import markdown as markdown
from . import shortcuts as shortcuts


__all__ = [
    'templatify',
    'Boolean',
    'IterableTagBase',
    'Option',
    'TagBase',
    'markdown',
    'shortcuts',
]


try:
    from importlib.metadata import version

    __version__ = version('py-templatify')
except ModuleNotFoundError:  # pragma: no cover
    __version__ = f'No version available for {__name__}'  # pragma: no cover
