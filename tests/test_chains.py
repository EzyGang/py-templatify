import inspect

import pytest

from py_templatify._types import Wrapped
from py_templatify.markdown.tags import H1, H2, Bold, CodeBlock, DotList, Italic, Link, Quote


@pytest.fixture
def wrapped():
    return Wrapped(inspect.signature(lambda x: x), None, None, 'test')


def test_h1_bold(wrapped):
    tags = [H1(), Bold()]

    is_escaped, new_value = wrapped._process_annotation_metadata('John Doe', tags)

    assert new_value == '**# John Doe**'


def test_h1_italic(wrapped):
    tags = [H1(), Italic()]

    is_escaped, new_value = wrapped._process_annotation_metadata('John Doe', tags)

    assert new_value == '*# John Doe*'


def test_h2_quote(wrapped):
    tags = [H2(), Quote()]

    is_escaped, new_value = wrapped._process_annotation_metadata('John Doe', tags)

    assert new_value == '> ## John Doe'


def test_codeblock_bold(wrapped):
    tags = [CodeBlock(code='test'), Bold()]

    is_escaped, new_value = wrapped._process_annotation_metadata('John Doe', tags)

    assert new_value == '**```test\nJohn Doe\n```**'


def test_link_image(wrapped):
    tags = [Link(), Bold()]

    is_escaped, new_value = wrapped._process_annotation_metadata(('John', 'https://example.com'), tags)

    assert new_value == '**[John](https://example.com)**'


def test_combined_tags(wrapped):
    tags = [H1(), Bold(), Italic()]

    is_escaped, new_value = wrapped._process_annotation_metadata('John Doe', tags)

    assert new_value == '***# John Doe***'


def test_combined_list(wrapped):
    tags = [DotList(), CodeBlock(code='test')]

    is_escaped, new_value = wrapped._process_annotation_metadata(['john', 'doe'], tags)

    assert new_value == '```test\n- john\n- doe\n\n```'
