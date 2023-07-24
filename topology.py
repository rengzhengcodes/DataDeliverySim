"""Abstracts out the topology of the network."""
from __future__ import annotations

from elements.buffer import Buffer
from elements.processing import PE

class Topology():
    """The topology we're trying to simulate."""

    def __init__(self, dims: tuple, pes: tuple[PE], buffers: tuple[Buffer]) -> None:
        """
        Initializes the topology class.

        @dims   A tuple representing the dims of a topology grid.
        @pes    The PEs of a topology.
        @buffer The buffers of a topology.
        """

        self._dims = dims
        self._pes = pes
        self._buffer = buffers
    
    def run_propagation(self) -> :