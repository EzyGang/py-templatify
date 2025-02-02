import pytest

from py_templatify._tags._base import UNSET, Boolean, IterableTagBase, Option, TagBase


def escape_func(value: str) -> str:
    return f'escaped({value})'


@pytest.mark.parametrize(
    'pre, post, value, expected',
    [
        ('<p>', '</p>', 'Hello', '<p>escaped(Hello)</p>'),
        ('', '', 'World', 'escaped(World)'),
        ('<b>', '</b>', UNSET, 'Value must be set'),
    ],
)
def test_tagbase_str(pre, post, value, expected):
    tag = TagBase[str](pre=pre, post=post)

    if value == UNSET:
        with pytest.raises(ValueError, match=expected):
            str(tag)  # Trigger ValueError for unset value
    else:
        assert str(tag(val=value, escape=escape_func)) == expected


@pytest.mark.parametrize(
    'elements, pre_elem, post_elem, pre, post, expected',
    [
        (
            ['item1', 'item2'],
            '<li>',
            '</li>',
            '<ul>',
            '</ul>',
            '<ul><li>escaped(item1)</li><li>escaped(item2)</li></ul>',
        ),
        ([], '', '', '', '', 'Value must be set'),
    ],
)
def test_iterabletagbase_str(elements, pre_elem, post_elem, pre, post, expected):
    it_tag = IterableTagBase(pre_elem=pre_elem, post_elem=post_elem, pre=pre, post=post)

    if not elements:
        with pytest.raises(ValueError, match=expected):
            str(it_tag)  # Trigger ValueError for unset value
    else:
        assert str(it_tag(val=elements, escape=escape_func)) == expected


@pytest.mark.parametrize(
    'value, escape, expected, is_empty',
    [
        ('Hello', UNSET, 'Hello', False),
        (UNSET, UNSET, 'default', True),
        (None, UNSET, 'default', True),
    ],
)
def test_option(value, escape, expected, is_empty):
    opt = Option(val=value, escape=escape, if_none='default')
    assert opt.is_empty == is_empty

    assert opt() == expected


@pytest.mark.parametrize(
    'value, expected',
    [
        (True, '+'),
        (False, '-'),
        (UNSET, '-'),
    ],
)
def test_boolean_str(value, expected):
    bool_tag = Boolean(value)
    assert str(bool_tag) == expected


@pytest.mark.parametrize(
    'val, expected',
    [
        (True, 'escaped(+)'),
        (False, 'escaped(-)'),
        (UNSET, 'escaped(-)'),
    ],
)
def test_boolean_str_with_escape(val, expected):
    bool_tag = Boolean(val=val, escape=escape_func)
    assert str(bool_tag) == expected


@pytest.mark.parametrize(
    'val,expected,func',
    [
        ('Hello', 'escaped(Hello)', escape_func),
        (UNSET, UNSET, escape_func),
        (123, '123', UNSET),
    ],
)
def test_tagbase_escape(val, expected, func):
    tag = TagBase(val=val, escape=func)

    assert tag.escape(val) == expected
