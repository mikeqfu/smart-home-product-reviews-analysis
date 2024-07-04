"""
A module for toy examples.
"""


class ToyExamples:
    """
    A class for brief demonstration of example models.
    """

    def __init__(self, random_state=0):
        """
        :param random_state: random seed number, defaults to ``0``
        :type random_state: int or None

        :ivar int or None random_state: A random seed number.
        """

        self.random_state = random_state
