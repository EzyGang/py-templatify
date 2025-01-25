from typing import Any, Protocol


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...


class SupportsStr(Protocol):
    def __str__(self) -> str: ...


class SupportsIter(Protocol):
    def __iter__(self) -> Any: ...
