
import pytest

from py_templatify._tags._base import UNSET
from py_templatify._tags._markdown import CodeBlock, DotList, Image, Link, OrderedList


def test_link_tag_validation():
    # Test unset value
    with pytest.raises(ValueError, match='Value must be set'):
        str(Link(UNSET))

    # Test invalid value type
    with pytest.raises(ValueError, match='Value must be set'):
        str(Link())


def test_image_tag():
    # Test basic image tag
    img = Image[str](('alt text', 'image.jpg'))
    assert str(img) == '![alt text](image.jpg)'

    # Test with URLs
    img = Image[str](('profile pic', 'https://example.com/pic.png'))
    assert str(img) == '![profile pic](https://example.com/pic.png)'


def test_code_block_with_language():
    # Test code block with language specification
    python_code = CodeBlock[str]("print('hello')", code='python')
    assert str(python_code) == "```python\nprint('hello')\n```"

    # Test code block with empty language
    no_lang_code = CodeBlock[str]('var x = 5;', code='')
    assert str(no_lang_code) == '```\nvar x = 5;\n```'

    # Test multiline code
    multiline = CodeBlock[str]('def func():\n    pass', code='python')
    assert str(multiline) == '```python\ndef func():\n    pass\n```'


def test_dot_list_with_different_iterables():
    # Test with list
    assert str(DotList[str](['a', 'b', 'c'])) == '- a\n- b\n- c\n'

    # Test with tuple
    assert str(DotList[str](('x', 'y'))) == '- x\n- y\n'

    # Test with list of strings containing numbers
    assert str(DotList[str](['1', '2', '3'])) == '- 1\n- 2\n- 3\n'

    # Test with list comprehension
    numbers: list[str] = [str(x * 2) for x in range(1, 3)]
    assert str(DotList[str](numbers)) == '- 2\n- 4\n'


def test_ordered_list_with_different_iterables():
    # Test with list
    assert str(OrderedList[str](['a', 'b'])) == '1. a\n2. b\n'

    # Test with tuple
    assert str(OrderedList[str](('x', 'y', 'z'))) == '1. x\n2. y\n3. z\n'

    # Test with list of string numbers
    assert str(OrderedList[str](['2', '4'])) == '1. 2\n2. 4\n'

    # Test empty list
    assert str(OrderedList[str]([])) == ''
