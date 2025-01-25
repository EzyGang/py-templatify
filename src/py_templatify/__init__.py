from ._decorators import templatify as templatify
from ._tags._base import Boolean as Boolean
from ._tags._base import IterableTagBase as IterableTagBase
from ._tags._base import Option as Option
from ._tags._base import TagBase as TagBase


try:
    from importlib.metadata import version

    __version__ = version('py-templatify')
except ModuleNotFoundError:
    __version__ = f'No version available for {__name__}'
