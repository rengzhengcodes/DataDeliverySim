"""Defines the PE classes for topology."""
from elements import Feature


class PE(Feature):
    """Defines the PE class for topology."""

    def __init__(self, loc: tuple, data: set[int]):
        """
        Initializes a PE.

        @loc    The id reference of the PE.
        @data   A list of UUIDs of the packets needed in the PE.

        @pre    We expect data size of a PE to be rate limited by the
                topology.
        """
        self.loc = loc
        self.data = data

    def __repr__(self):
        """String function for a PE."""
        return f"PE({self.loc}, {self.data})"

    def __str__(self):
        """String function for a PE."""
        return f"PE {self.loc}: {self.data}"
