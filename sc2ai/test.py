"""
    Test Module
"""

class Test(object):
    """A class which receives two inputs and does various operations on.

    :param int a: value of a
    :param int b: value of b
    """

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def is_same(self):
        """Compares a and b if the are the same

        :return: boolean True if a and b are the same, otherwise false

        Example:
            Use as follows.

            >>> Test(1, 2).is_same()
            False

        """
        return self._a == self._b

    def plus(self):
        """Returns the sum of both a and b

        :returns: int the result of a + b

        Example:
            Use as follows:

            >>> Test(2, 1).plus()
            3


        """
        return self._a + self._b
