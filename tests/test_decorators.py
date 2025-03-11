# ruff: noqa: E501
from typing import Annotated

import pytest

from py_templatify import templatify
from py_templatify._tags._markdown import Code
from py_templatify._types import Wrapped


# Example function to be used for testing
def example_function(x: int, y: str = 'default') -> None:
    """Template string for the example function. {x} {y}"""
    ...


def no_docstring_function(x: int) -> int:
    return x * 2


def test_templatify_initialization():
    # Test initialization with valid parameters
    deco = templatify(description='Example description', escape_symbols='&')
    assert deco._description == 'Example description'
    assert deco._escape_symbols == '&'

    # Test initialization with None parameters
    deco = templatify()
    assert deco._description is None
    assert deco._escape_symbols is None


def test_templatify_with_valid_function():
    deco = templatify(description='Example description')

    wrapped = deco(example_function)

    assert isinstance(wrapped, Wrapped)


def test_templatify_without_docstring():
    deco = templatify()

    with pytest.raises(RuntimeError, match='Template string is not provided'):
        deco(no_docstring_function)


def test_templatify_signature_retrieval():
    deco = templatify()

    wrapped = deco(example_function)

    assert wrapped._signature.parameters['x'].annotation is int
    assert wrapped._signature.parameters['y'].annotation is str
    assert wrapped._signature.parameters['y'].default == 'default'
    assert len(wrapped._signature.parameters) == 2


@pytest.mark.parametrize('description,expected', [(None, None), ('Custom description', 'Custom description')])
def test_templatify_description_param(description, expected):
    deco = templatify(description=description)
    wrapped = deco(example_function)
    assert wrapped.__doc__ == expected


def test_index_access_list():
    @templatify()
    def list_template(items: list[str]) -> None:
        """First item: {items[0]}, Second item: {items[1]}, Missing item: {items[999]}"""

    result = list_template(items=['apple', 'banana'])
    assert result == 'First item: apple, Second item: banana, Missing item: None'


def test_index_access_dict():
    @templatify()
    def dict_template(data: dict[str, int]) -> None:
        """Value a: {data['a']}, Value b: {data["b"]}, Missing value: {data['missing']}"""

    result = dict_template(data={'a': 1, 'b': 2})
    assert result == 'Value a: 1, Value b: 2, Missing value: None'


def test_index_access_dict_annotated():
    @templatify()
    def dict_template(data: Annotated[dict[str, int], Code]) -> None:
        """Value a: {data['a']}, Value b: {data["b"]}, Missing value: {data['missing']}"""

    result = dict_template(data={'a': 1, 'b': 2})
    assert result == 'Value a: `1`, Value b: `2`, Missing value: `None`'


def test_multi_level_index_access():
    @templatify()
    def nested_template(matrix: list[list[int]], nested_dict: dict[str, dict[str, str]]) -> None:
        """
        Matrix values: {matrix[0][1]}, {matrix[1][0]}, Invalid first level: {matrix[99][0]},
        Nested dict: {nested_dict['outer']['inner']}, Invalid first level: {nested_dict['missing']['key']},
        Invalid second level: {nested_dict['outer']['missing']}
        """

    result = nested_template(matrix=[[1, 2], [3, 4]], nested_dict={'outer': {'inner': 'value'}})
    assert (
        result
        == '\nMatrix values: 2, 3, Invalid first level: None,\nNested dict: value, Invalid first level: None,\nInvalid second level: None\n'
    )


def test_mixed_attribute_and_index_access():
    class Container:
        def __init__(self, items: list[str]):
            self.items = items

    @templatify()
    def mixed_template(container: Container) -> None:
        """Items: {container.items[0]}, {container.items[1]}"""

    result = mixed_template(container=Container(['first', 'second']))
    assert result == 'Items: first, second'


def test_invalid_index_types():
    @templatify()
    def invalid_template(items: list[int], data: dict[str, int]) -> None:
        """
        List with str index: {items['key']},
        Dict with int index: {data[0]},
        Invalid syntax: {items[]}
        """

    with pytest.raises(ValueError):
        invalid_template(items=[1, 2, 3], data={'a': 1})
