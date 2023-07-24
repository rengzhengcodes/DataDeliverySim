"""Abstracts out the topology of the network."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
from typing import Any, Iterable

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE


class Topology:
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
        self._dsts: dict[Any, set] = self.build_directory(self._pes)
        self._srcs: dict[Any, set] = self.build_directory(self._buffer)

    def build_adjacencies(self) -> tuple:
        """
        Builds adjacencies based off of the dims.
        """

        def sub_adj_builder(dim_idx: int = 0) -> Iterable:
            """
            Builds the sub_adjacencies.

            @param dim_idx  The index of the dimension we're currently on.
            """
            # If not at base case.
            if dim_idx < len(self._dims) - 1:
                # Goes through all sub adjacencies.
                sub_adj: list
                for sub_adj in sub_adj_builder(dim_idx + 1):
                    # If not already in an orthogonal, append something.
                    if not any(sub_adj):
                        for i in (-1, 0, 1):
                            yield (i,) + sub_adj
                    # Otherwise, append 0
                    else:
                        yield (0,) + sub_adj
            # If at base case, yield the following
            else:
                yield from ((-1,), (0,), (1,))

        # Makes sure not to return the 0 case.
        adj: tuple
        for adj in sub_adj_builder():
            yield adj

    def build_directory(self, features: Iterable[Feature]) -> dict[Any, set]:
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

    def build_diffusion_grid(self, init_val: Any, dim_idx: int = 0) -> list:
        """Recursively builds a grid for the topological diffusion to happen on

        @param dim_idx  The index of the dim we're currently accessing.
        """
        # Iterator we use for intialization.
        iterator: Iterable = range(self._dims[dim_idx])

        # Recurses if we haven't reached the base case.
        if dim_idx < len(self._dims) - 1:
            lower_grid: tuple = self.build_diffusion_grid(init_val, dim_idx + 1)
            return [deepcopy(lower_grid) for i in iterator]

        # Otherwise, build the lowest layer diffusion grid.
        return [init_val for i in iterator]

    def build_coords(self) -> tuple:
        """
        Yields all locations within the grid.
        """
        yield from product(*(range(dim) for dim in self._dims))

    def deduce_subspace(self, space: Iterable, coords: tuple) -> Iterable:
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
            subspace = subspace[coord]

        return subspace

    def bounds_check(self, loc: tuple) -> bool:
        """Checks if a location is in bounds."""
        for index, elem in enumerate(loc):
            if not 0 <= elem < self._dims[index]:
                return False

        return True

    def diffuse_packet(self, pkt: int) -> tuple:
        """
        Diffuses a packet through the grid until it reaches all its destinations.

        @pkt        The packet to diffuse.
        @pkt_grid   A grid representing the diffusion of a packet.

        @return     A tuple where the first element is the number of steps needed to
                    diffuse the packet to all elements and the second element is the
                    total number of steps taken by the diffusion.
        """
        # If the packet was never assigned to a dst, no need to run.
        if pkt not in self._dsts:
            return (0, 0)

        # Notes the target locations.
        target_locs: set[tuple] = {pe.loc for pe in self._dsts[pkt]}
        # Notes number of target_locs
        num_locs: int = len(target_locs)

        # Initializes the diffusion grid, tracking steps from the nearest packet.
        pkt_grid: tuple = self.build_diffusion_grid(-1)
        # Initializes the diffusion priority queue.
        queue: list[tuple[int, tuple]] = []
        # Initializes the diffusion grid with the sources.
        src: tuple
        for src in self._srcs[pkt]:
            # Seeds the grid.
            self.deduce_subspace(pkt_grid, src.loc)[src.loc[-1]] = 0
            # Adds to diffusion queue
            queue.append((0, 0, src.loc))

        # Tracks the maximum number of steps taken to reach all destinations.
        max_steps: int = 0
        # Tracks the total number of steps taken to reach all destinations.
        tot_steps: int = 0
        # Goes through the grid, diffusing the packet.
        while queue:
            # Grabs first off queue.
            steps: int
            dist: int
            loc: tuple
            steps, dist, loc = queue.pop(0)

            # If there are no more targets, stop at max_steps.
            if not target_locs and steps > max_steps:
                continue

            # Deduces current location.
            loc_space: list = self.deduce_subspace(pkt_grid, loc)
            # Deduces current value.
            loc_val: int = loc_space[loc[-1]]

            # If we are a target_loc.
            if loc in target_locs:
                # Remove the destination from the set of target locations.
                target_locs.remove(loc)
                # Replaces max_steps if steps+1 is greater
                max_steps = max(max_steps, steps)
                # Add number of steps to tot_steps.
                tot_steps += dist
                # Resets dist to 0 because we already need to be here.
                dist = 0
                # Sets loc_val to distance from propagation.
                loc_space[loc[-1]] = steps
            # Otherwise, if the location has a higher or equal or negative step
            # count, compared to the current diffusion path diffuses the packet.
            elif loc_val < 0 or loc_val > steps:
                loc_space[loc[-1]] = steps

            # If there are no more targets, just resolve current queue space.
            if target_locs:
                # Goes through each adjacency.
                adj: tuple
                for adj in self.build_adjacencies():
                    # Calculates the adjacent location.
                    adj_loc: tuple = tuple(
                        loc[i] + adj[i] for i in range(len(self._dims))
                    )
                    # Adds adjacency to queue if in bounds.
                    if self.bounds_check(adj_loc):
                        queue.append((steps + 1, dist + 1, adj_loc))

        # Sanity check the program works correctly.
        assert max_steps <= tot_steps <= (max_steps * num_locs)

        return (pkt, max_steps, tot_steps, pkt_grid)
