from typing import Any, Protocol


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...  # pragma: no cover


class SupportsStr(Protocol):
    def __str__(self) -> str: ...  # pragma: no cover


class SupportsIter(Protocol):
    def __iter__(self) -> Any: ...  # pragma: no cover
