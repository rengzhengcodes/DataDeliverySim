"""Abstracts out the topology of the network."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Iterable

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE

class Topology():
    """The topology we're trying to simulate."""

    def __init__(self, dims: tuple, pes: tuple[PE], buffers: tuple[Buffer]) -> None:
        """
        Initializes the topology class.

        @dims   A tuple representing the dims of a topology grid. First element
                corresponds to the first index used to access. E.g. (x, y, z)
                corresponds to topology[x][y][z]. 0-indexed.
        @pes    The PEs of a topology.
        @buffer The buffers of a topology.
        """
        # Keeps track of all the input variables.
        self._dims: tuple = dims
        self._pes: tuple[PE] = pes
        self._buffer: tuple[Buffer] = buffers

        # Destination and source directories.
        self._dsts: dict[Any, set] = Topology.build_directory(self._pes)
        self._srcs: dict[Any, set] = Topology.build_directory(self._buffer)

    def build_directory(features: Iterable[Feature]) -> dict[Any, set]:
        """
        Builds a directory corresponding the "data" property of a Feature to
        a reference to the Feature.

        @param elem     The features for which we are building the directory.
        
        @return         A dictionary corresponding the data identifier to a set
                        of all features with that identifier.
        """
        # Output dictionary
        ret: dict[Any, set] = {}

        # Builds the destination directory for all packet ids.
        feature: Feature
        for feature in features:
            # Pulls out all pkt_ids and adds the PE to the set.
            pkt_id: Any
            for pkt_id in feature.data:
                # If a set doesn't exist at a pkt_id, create one
                if pkt_id not in ret:
                    ret[pkt_id] = set()
                # Adds PE to the pkt_id directory
                ret[pkt_id].add(feature)
        
        return ret

    def build_diffusion_grid(self, dim_idx: int) -> list:
        """Recursively builds a grid for the topological diffusion to happen on
        
        @param dim_idx  The index of the dim we're currently accessing.
        """
        # Iterator we use for intialization.
        iterator: Iterable = range(self._dims[dim_idx])

        # Recurses if we haven't reached the base case.
        if (dim_idx < len(self._dims) - 1):
            lower_grid: tuple = self.build_diffusion_grid(dim_idx + 1)
            return [deepcopy(lower_grid) for i in iterator]

        # Otherwise, build the lowest layer diffusion grid.
        return [-1 for i in iterator]

    def deduce_subspace(space: Iterable, coords: tuple) -> Iterable:
        """
        Given a multidimensional iterable, uses the coordinates given to return
        the subspace the last coord is contained in.

        @space  The space whose subspace we're deducing.
        @coord  The coordinates within the grid we want to reach.

        @return The Iterable that contains the last subspace referred to in coord.
        """
        # Tracks the current subspace we are in.
        subspace: tuple = space
        # Goes recursively through the indices until it reaches it.
        i: int
        coord: int
        for i, coord in enumerate(coords):
            # Assigns if we're on the last coord.
            if i == len(coords) - 1:
                return subspace
            # Otherwise, continue reducing subspace based on coords.
            else:
                subspace = subspace[coord]

    
    def diffuse_packet(self, pkt: int) -> tuple:
        """
        Diffuses a packet through the grid until it reaches all its destinations.

        @pkt        The packet to diffuse.
        @pkt_grid   A grid representing the diffusion of a packet.

        @return     A tuple where the first element is the number of steps needed to
                    diffuse the packet to all elements and the second element is the
                    total number of steps taken by the diffusion.
        """
        # Notes the target locations.
        target_locs: set[tuple] = {pe.loc for pe in self._dsts[pkt]}
        # Reference check for correct calcs.
        ref_locs: set[tuple] = target_locs.copy()

        # Initializes the diffusion grid, tracking steps from the nearest packet.
        pkt_grid: tuple = self.build_diffusion_grid(0)

        # Initializes the diffusion grid with the sources.
        src: tuple[int, int]
        for src in self._srcs[pkt]:
            # Gets the smallest subspace containing the cell we want to seed.
            subspace: tuple = Topology.deduce_subspace(pkt_grid, src.loc)
            # Seeds that cell.
            subspace[src.loc[-1]] = 0

        # Tracks the maximum number of steps taken to reach all destinations.
        max_steps: int = 0
        # Tracks the total number of steps taken to reach all destinations.
        tot_steps: int = 0
        # lambda function to detect if a packet has reached a locations.
        reached: callable([tuple[int, int]], bool) = (
            lambda loc: Topology.deduce_subspace(pkt_grid, loc)[loc[-1]] >= 0
        )
        # Adjacencies in the topology grid, representing the diffusion.
        adjacencies: tuple[tuple[int, int]] = ((0, 1), (0, -1), (1, 0), (-1, 0))
        # Runs the diffusion until all destinations are reached.
        while any(not reached(loc) for loc in target_locs):
            # Goes through the grid, diffusing the packet.
            i: int
            for i in range(self._dims[0]):
                j: int
                for j in range(self._dims[1]):
                    # If the packet is at this location, diffuses it.
                    if reached((i, j)):
                        # Goes through each adjacency.
                        adj: tuple
                        for adj in adjacencies:
                            # Calculates the adjacent location.
                            adj_loc: tuple[int, int] = (i + adj[0], j + adj[1])
                            # If the adjacent location is in the grid and has a higher
                            # or negative step count, diffuses the packet.
                            if (0 <= adj_loc[0] < self._dims[0] and 0 <= adj_loc[1] < self._dims[1]) and (
                                pkt_grid[adj_loc[0]][adj_loc[1]] < 0
                                or pkt_grid[adj_loc[0]][adj_loc[1]] > pkt_grid[i][j] + 1
                            ):
                                # If the adj_loc is a destination.
                                if adj_loc in target_locs:
                                    # Diffuse 0.
                                    pkt_grid[adj_loc[0]][adj_loc[1]] = 0
                                    # Add number of steps to tot_steps.
                                    tot_steps += pkt_grid[i][j] + 1
                                    # Remove the destination from the set of target
                                    # locations.
                                    target_locs.remove(adj_loc)
                                # Otherwise, diffuse the packet.
                                else:
                                    pkt_grid[adj_loc[0]][adj_loc[1]] = pkt_grid[i][j] + 1

            max_steps += 1

        # Sanity check the program works correctly.
        assert max_steps <= tot_steps
        # The only case where it should be zero in a topology where buffers and
        # PEs are discrete units is when no PEs request the data.
        if max_steps == 0:
            assert not ref_locs

        return (max_steps, tot_steps)