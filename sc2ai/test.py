"""
    Test Module

    A module to test the functionality of Sphinx.

    See Also:
        - https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
        - https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
        - https://github.com/google/styleguide/blob/gh-pages/pyguide.md
        - https://google.github.io/styleguide/pyguide.html
"""

#: A doc comment for bar
bar = "bar"

test = "abc"
""" A sample global variable. """
class Test(object):
    """A class which receives two inputs and does various operations on.

    Args:
        a (int): value of a
        b (int): value of b
    """

    def __init__(self, a: int, b: int):
        self._a = a
        self._b = b

    def is_same(self):
        """Compares a and b if the are the same

        Returns:
            bool: True if a and b are the same, otherwise false

        Example:
            Use as follows.

            >>> Test(1, 2).is_same()
            False
        """
        return self._a == self._b

    def plus(self):
        """Returns the sum of both a and b

        Returns:
            int: the result of a + b

        Example:
            Use as follows:

            >>> Test(2, 1).plus()
            3
        """
        return self._a + self._b
