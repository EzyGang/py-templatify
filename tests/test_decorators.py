import pytest

from py_templatify import templatify
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
