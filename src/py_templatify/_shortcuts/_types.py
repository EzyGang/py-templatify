from typing import Annotated

from py_templatify._tags._markdown import (
    H1,
    H2,
    H3,
    Bold,
    Code,
    CodeBlock,
    DotList,
    Highlight,
    Image,
    Italic,
    LineAfter,
    LineBefore,
    Link,
    OrderedList,
    Quote,
    Spoiler,
    Strike,
    Underline,
)
from py_templatify._tags._types import SupportsIter, SupportsStr


type H1Tag[T] = Annotated[T, H1]
type H2Tag[T] = Annotated[T, H2]
type H3Tag[T] = Annotated[T, H3]
type BoldTag[T] = Annotated[T, Bold]
type ItalicTag[T] = Annotated[T, Italic]
type QuoteTag[T] = Annotated[T, Quote]
type CodeTag[T] = Annotated[T, Code]
type LineAfterTag[T] = Annotated[T, LineAfter]
type LineBeforeTag[T] = Annotated[T, LineBefore]
type StrikeTag[T] = Annotated[T, Strike]
type HighlightTag[T] = Annotated[T, Highlight]
type UnderlineTag[T] = Annotated[T, Underline]
type SpoilerTag[T] = Annotated[T, Spoiler]
type CodeBlockTag[T] = Annotated[T, CodeBlock]
type LinkTag[T: tuple[SupportsStr, SupportsStr]] = Annotated[T, Link]
type ImageTag[T: tuple[SupportsStr, SupportsStr]] = Annotated[T, Image]
type DotListTag[T: SupportsIter] = Annotated[T, DotList]
type OrderedListTag[T: SupportsIter] = Annotated[T, OrderedList]
