Py-Templatify

---

**Documentation**: https://py-templatify.readthedocs.io/latest/  
**Source Code**: https://github.com/YourUsername/py-templatify/

---  

**Py-Templatify** is a lightweight library designed for developers who want a straightforward and efficient way to generate simple string 
templates in Python applications. 

It is particularly well-suited for building templates for **chat bots**, **LLM prompts**, 
and other scenarios where concise and easily managed string outputs are required.

## Key Features:
- **Simple Templating**: Transform standard functions into templated functions using the `@templatify` decorator, enabling straightforward dynamic string generation without the complexity of traditional templating engines.

- **Minimal Logic**: Py-Templatify is tailored for small templates, making it ideal for quick formatting tasks without extensive in-template transformation logic.

- **Markdown Support**: Integrate Markdown formatting by default, allowing for structured output, and extend functionality using base classes for custom tags and formatting.

- **Strong Typing with Annotations**: Utilize Python's type hints to define the format and behavior of each argument in the template, ensuring clarity and maintainability.

- **Optional Values**: Manage optional template values gracefully with the `Option` class, providing fallback values when needed.

- **Roadmap**: A roadmap for future improvements and features is in the works.

---

## Table of Contents
- [Installation](#installation)
- [How to use](#how-to-use)
- [Basic examples](#basic-examples)
- [Contributing](#contributing)
- [License](#license)

## Installation
```bash
$ pip install py-templatify
# ---> 100% Successfully installed py-templatify
```

## How to use
To start using the library, decorate your function with `@templatify`:
```python
from py_templatify import templatify

@templatify(description='Short description')  # The `description` param is not utilized anywhere
def your_function(arg1: str, ...):
    """
    This is where your template message goes.
    
    Here is the {arg1}!
    """
    ...
    

print(your_function(arg1='Awesome argument'))
```

After running the above you'll get the following output:
```
This is where your template message goes.

Here is the Awesome argument!
```

### Example Usage
Here's a generic example demonstrating the functionality of the library:

```python
from decimal import Decimal
from typing import Annotated, Any
from py_templatify import templatify, Option, Boolean
from py_templatify.markdown.shortcuts import BoldTag, CodeTag, LinkTag
from py_templatify.markdown.tags import Bold, Spoiler, Code


# You can define a reusable tag like this
MyOption = Option[str](if_none='No notes provided')


@templatify(description='Template for user information')
def user_info_template(
    username: BoldTag[str],
    email: Annotated[str, Bold],  # this is the same as `username`, but explicitly
    account_status: CodeTag[str],
    plan_price: Annotated[Decimal, Code, Spoiler],  # You can chain tags, they are executed from left to right
    subscription_active: Annotated[bool, Boolean(if_true='Yes')],
    optional_note: Annotated[str | None, MyOption],
    optional_note_2: Annotated[str | None, MyOption],
    user_link: LinkTag[tuple[str, str]],
) -> None:
    """
    User Information:
    
    Welcome, {username}!
    Your email: {email}.
    
    Account Status: {account_status}
    Plan price: {plan_price}

    Link: {user_link}
    
    Note: {optional_note}
    Note 2: {optional_note_2}
    
    Subscription Active: {subscription_active}
    """

if __name__ == '__main__':
    print(
        user_info_template(
            username='john_doe', 
            email='john.doe@google.com', 
            account_status='Active', 
            plan_price=Decimal('4.99'),
            subscription_active=True,
            optional_note=None,
            optional_note_2='Note number 2',
            user_link=('User Profile', 'https://example.com/user'),
        )
    )
```

After running the above:
```
User Information:

Welcome, **john_doe**!
Your email: **john.doe@google.com**.

Account Status: `Active`
Plan price: ||`4.99`||

Link: [User Profile](https://example.com/user)

Note: No notes provided
Note 2: Note number 2

Subscription Active: Yes
```

## Basic examples
### Using Tags
Utilize the built-in tags like `Bold`, `Link`, and `Code` to format your templates effortlessly. 
Basic Markdown formatting is shipped by default, but you can extend and create custom tags using the provided base classes.

## Contributing
If you'd like to contribute to Py-Templatify, please discuss your changes using Issues or open a Pull Request which will be reviewed soon.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/EzyGang/py-templatify/blob/main/LICENSE) file for details.