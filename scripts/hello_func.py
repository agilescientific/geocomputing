#!/usr/bin/env python

import sys


def greet(person):
    """
    Return a greeting, given a person.

    Args:
        person (str): The person's name.

    Returns:
        str. The greeting.

    Example:
    >>> greet('Matt')
    Hello Matt!
    """
    return "Hello {}!".format(person)


if __name__ == "__main__":
    print(greet(sys.argv[1]))
