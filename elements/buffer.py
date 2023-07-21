"""
Declares the Buffer classes of a Topology.
"""
from elements import Feature


class Buffer(Feature):
    """Represents a buffer in a Topology."""

    def __init__(self, buf_id: int, data: list[int]):
        """
        Initializes a Buffer.

        @id     The id reference of the buffer.
        @data   A list of UUIDs of the packets in the buffer.

        @pre    We expect data size of a buffer to be rate limited by the
                topology.
        """
        self.buf_id = buf_id
        self.data = data

    def __repr__(self):
        """String function for a Buffer."""
        return f"Buffer({self.buf_id}, {self.data})"

    def __str__(self):
        """String function for a Buffer."""
        return f"Buffer {self.buf_id}: {self.data}"
