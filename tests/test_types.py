import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import pytest
from pytest_mock import MockerFixture

from py_templatify import Option, TagBase, templatify
from py_templatify._types import Wrapped, is_option, is_tag


class CustomOption(Option):
    pass


class CustomTag(TagBase):
    pass


@pytest.mark.parametrize(
    'value, expected',
    [
        (Option(), True),  # Instance of Option
        (CustomOption(), True),  # Subclass of Option
        ('not an option', False),  # Not an Option instance
        (Option, True),  # Class of Option
        (CustomOption, True),  # Class of CustomOption
        (None, False),  # None value
        (123, False),  # Not an Option instance
        (object, False),  # Random object
    ],
)
def test_is_option(value, expected):
    assert is_option(value) == expected, f'Expected {value} to return {expected} for is_option'


@pytest.mark.parametrize(
    'value, expected',
    [
        (TagBase(''), True),  # Instance of TagBase
        (CustomTag(''), True),  # Subclass of TagBase
        (123, False),  # Not a TagBase instance
        (TagBase, True),  # Class of TagBase
        (CustomTag, True),  # Class of CustomTag
        (None, False),  # None value
        (object, False),  # Random object
    ],
)
def test_is_tag(value, expected):
    assert is_tag(value) == expected, f'Expected {value} to return {expected} for is_tag'


@dataclass
class User:
    name: str
    age: int
    info: str


def sample_function(user: User) -> None:
    """Hello @{user.name}, you are {user.age} years old! Info: {user.info}"""


def sample_function_plain(username: str) -> None:
    """Hello {username}!"""


template_decorator = templatify(escape_symbols='@!')


@pytest.fixture
def wrapped_instance():
    return template_decorator(sample_function)


@pytest.fixture
def wrapped_instance_plain():
    return template_decorator(sample_function_plain)


@pytest.fixture
def wrapped_instance_with_annotations():
    def _function(user: User, number: int, option_param: Annotated[str | None, Option[str](if_none='No')]) -> None:
        """{user.name} has number {number} and the option is {option_param}"""

    return templatify(escape_symbols='@!')(_function)


def test_init(wrapped_instance):
    assert wrapped_instance._tpl == 'Hello @{user.name}, you are {user.age} years old! Info: {user.info}'
    assert wrapped_instance._escape_symbols == {'@', '!'}
    assert isinstance(wrapped_instance._escape_func, Callable)


def test_call(wrapped_instance):
    result = wrapped_instance(user=User(name='Alice', age=30, info='Some info'))

    expected_result = 'Hello \\@Alice, you are 30 years old\\! Info: Some info'
    assert result == expected_result

    with pytest.raises(TypeError):
        wrapped_instance(age=25)


def test_update_kwd_args_from_attributes(wrapped_instance):
    user = User(name='Alice', age=30, info='Some info')
    kwd_args = {'user': user}
    tpl = wrapped_instance._tpl

    updated_tpl = wrapped_instance._update_kwd_args_from_attributes(kwd_args, tpl)

    assert '{user_name}' in updated_tpl
    assert '{user_age}' in updated_tpl
    assert '{user_info}' in updated_tpl


def test_update_kwd_args_from_attributes_no_param(wrapped_instance, mocker: MockerFixture):
    user = User(name='Alice', age=30, info='Some info')
    kwd_args = {}
    tpl = wrapped_instance._tpl

    mocker.patch.object(
        wrapped_instance, '_get_parameter_values_from_objs_for_fields', return_value={'user': (None, user)}
    )
    wrapped_instance._update_kwd_args_from_attributes(kwd_args, tpl)


def test_update_kwd_args_from_attributes_plain(wrapped_instance_plain, mocker: MockerFixture):
    kwd_args = {'username': 'test'}
    tpl = wrapped_instance_plain._tpl

    spy = mocker.spy(wrapped_instance_plain, '_get_parameter_values_from_objs_for_fields')
    updated_tpl = wrapped_instance_plain._update_kwd_args_from_attributes(kwd_args, tpl)

    assert '{username}' in updated_tpl
    spy.assert_not_called()


def test_update_kwd_args_with_annotations(wrapped_instance_with_annotations):
    kwd_args = {'user': User(name='Bob', age=25, info='Some info'), 'number': 42, 'option_param': None}

    wrapped_instance_with_annotations._update_kwd_args_with_annotations(kwd_args)

    assert kwd_args['number'] == '42'
    assert kwd_args['option_param'] == 'No'


def test_get_parameter_value_after_transforms(wrapped_instance_with_annotations):
    value = 'Hello @Wrld!'
    transformed_value = wrapped_instance_with_annotations._get_parameter_value_after_transforms(value, None)

    assert transformed_value == 'Hello \\@Wrld\\!'


def test_process_annotation_metadata(wrapped_instance_with_annotations):
    def mock_escape_func_factory():
        return lambda s: s.replace('Hello', 'Hi')

    wrapped_instance_with_annotations._escape_func = mock_escape_func_factory()

    metadata = [TagBase(pre='<T>', post='</T>')]
    is_escaped, new_value = wrapped_instance_with_annotations._process_annotation_metadata('Hello there!', metadata)

    assert is_escaped
    assert new_value == '<T>Hi there!</T>'


def test_process_annotation_metadata_no_callable(wrapped_instance_with_annotations):
    metadata = ['not a callable']
    is_escaped, new_value = wrapped_instance_with_annotations._process_annotation_metadata('Hello there!', metadata)

    assert not is_escaped
    assert new_value == 'Hello there!'


def test_process_annotation_metadata_callable(wrapped_instance_with_annotations):
    metadata = [lambda x: 'replacedvalue']
    is_escaped, new_value = wrapped_instance_with_annotations._process_annotation_metadata('Hello there!', metadata)

    assert not is_escaped
    assert new_value == 'replacedvalue'


def test_process_annotation_metadata_option_resume(wrapped_instance_with_annotations):
    metadata = [Option(if_none='No', resume=True), TagBase(pre='<T>', post='</T>')]
    is_escaped, new_value = wrapped_instance_with_annotations._process_annotation_metadata(None, metadata)

    assert is_escaped
    assert new_value == '<T>No</T>'

    metadata = [Option(if_none='No', resume=False), TagBase(pre='<T>', post='</T>')]
    is_escaped, new_value = wrapped_instance_with_annotations._process_annotation_metadata(None, metadata)

    assert is_escaped
    assert new_value == 'No'


def test_get_format_kwargs(wrapped_instance):
    bound_args = inspect.signature(sample_function).bind(User(name='Alice', age=30, info='Some info'))

    args, kwargs = wrapped_instance._get_format_kwargs(bound_args)

    assert args == ()
    assert 'user' in kwargs


def test_get_parameter_values_from_objs_for_fields(wrapped_instance):
    user = User(name='Alice', age=30, info='Some info')
    kwd_args = {'user': user}

    wrapped_instance._used_attributes.extend(['test_attr_no_dot', 'test_obj.non_existent'])
    values = wrapped_instance._get_parameter_values_from_objs_for_fields(kwd_args)

    assert 'user.name' in values
    assert 'test_attr_no_dot' not in values
    assert values['test_obj.non_existent'] == (None, None)
    assert values['user.name'][1] == 'Alice'
    assert values['user.age'][1] == 30
    assert values['user.info'][1] == 'Some info'


def test_escape_func_factory(wrapped_instance):
    escape_func = wrapped_instance._escape_func_factory()
    escaped_string = escape_func('Hello @! World!')

    assert escaped_string == 'Hello \\@\\! World\\!'


def test_escape_tpl(wrapped_instance):
    escaped_tpl = wrapped_instance._escape_tpl()

    assert '\\@' in escaped_tpl

    wrapped_instance._escape_symbols = set()
    assert wrapped_instance._tpl == wrapped_instance._escape_tpl()


class TestType:
    pass


type ParamAnnotation = Annotated[TestType, 'metadata']
ParamAnnotationVar = Annotated[TestType, 'metadata']


def test_get_type_alias_origin():
    assert Wrapped._get_type_alias_origin(ParamAnnotation) is Annotated[TestType, 'metadata']
    assert ParamAnnotationVar is Annotated[TestType, 'metadata']


def test_get_annotation_from_parameter(wrapped_instance_with_annotations):
    d = {'name': 'user', 'kind': inspect.Parameter.POSITIONAL_OR_KEYWORD, 'default': None}
    param = inspect.Parameter(**d, annotation=Annotated[User, (opt := Option())])

    annotation = wrapped_instance_with_annotations._get_annotation_from_parameter(param)
    assert annotation == Annotated[User, opt]

    param = inspect.Parameter(**d, annotation=ParamAnnotation)
    annotation = wrapped_instance_with_annotations._get_annotation_from_parameter(param)
    assert annotation == Annotated[TestType, 'metadata']
