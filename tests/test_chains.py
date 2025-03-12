import pytest

from py_templatify._types import ParamValueTransformer
from py_templatify.markdown.tags import H1, H2, Bold, CodeBlock, DotList, Italic, Link, Quote


@pytest.fixture
def transformer():
    return ParamValueTransformer(lambda x: x)


def test_h1_bold(transformer):
    tags = [H1(), Bold()]

    is_escaped, new_value = transformer._process_annotation_metadata('John Doe', tags)

    assert new_value == '**# John Doe**'


def test_h1_italic(transformer):
    tags = [H1(), Italic()]

    is_escaped, new_value = transformer._process_annotation_metadata('John Doe', tags)

    assert new_value == '*# John Doe*'


def test_h2_quote(transformer):
    tags = [H2(), Quote()]

    is_escaped, new_value = transformer._process_annotation_metadata('John Doe', tags)

    assert new_value == '> ## John Doe'


def test_codeblock_bold(transformer):
    tags = [CodeBlock(code='test'), Bold()]

    is_escaped, new_value = transformer._process_annotation_metadata('John Doe', tags)

    assert new_value == '**```test\nJohn Doe\n```**'


def test_link_image(transformer):
    tags = [Link(), Bold()]

    is_escaped, new_value = transformer._process_annotation_metadata(('John', 'https://example.com'), tags)

    assert new_value == '**[John](https://example.com)**'


def test_combined_tags(transformer):
    tags = [H1(), Bold(), Italic()]

    is_escaped, new_value = transformer._process_annotation_metadata('John Doe', tags)

    assert new_value == '***# John Doe***'


def test_combined_list(transformer):
    tags = [DotList(), CodeBlock(code='test')]

    is_escaped, new_value = transformer._process_annotation_metadata(['john', 'doe'], tags)

    assert new_value == '```test\n- john\n- doe\n\n```'
