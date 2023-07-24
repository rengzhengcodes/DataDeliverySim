"""Creates and runs the simulation."""

import random

from joblib import Parallel, delayed

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE

# Length and width of the grid representing the topology.
M: int = 100
N: int = 100

# Number of packets to be generated.
num_packets: int = 5000

# Grid representing the topology.
topology: tuple[tuple[Feature]] = [[None for j in range(M)] for i in range(N)]

# List of packets to deliver.
packets = list(range(num_packets))

# List of packets that need to be placed into an src.
src_packets = list(packets)

# Set of all PEs that need a key.
dest = {key: set() for key in packets}
# Sources of packets.
srcs = {key: set() for key in packets}

# Populates the topology, alternating rows of PEs and Buffers.
x: int
for x in range(M):
    y: int
    for y in range(N):
        if x % 2 == 0:
            # Chooses 3 packets that need to be delivered.
            deliveries: set[int] = set(random.sample(packets, 3))
            # Initializes a PE with those packets.
            topology[x][y] = PE((x, y), deliveries)
            # For each PE that needs to receive a delivery, adds the PE to the
            # set of dests.
            packet: int
            for packet in deliveries:
                dest[packet].add(topology[x][y])
        else:
            # Chooses 1 packet from packets.
            packet: set[int] = random.choice(src_packets)
            # Removes  that packet from future packets to be put in.
            src_packets.remove(packet)
            # Initializes a buffer with that packet.
            topology[x][y] = Buffer((x, y), {packet})
            # Adds the buffer to the set of sources for that packet.
            srcs[packet].add(topology[x][y])


def diffuse_packet(pkt: int) -> tuple:
    """
    Diffuses a packet through the grid until it reaches all its destinations.

    @pkt        The packet to diffuse.
    @pkt_grid   A grid representing the diffusion of a packet.

    @return     A tuple where the first element is the number of steps needed to
                diffuse the packet to all elements and the second element is the
                total number of steps taken by the diffusion.
    """
    # Notes the target locations.
    target_locs: set[tuple] = {pe.loc for pe in dest[pkt]}
    # Reference check for correct calcs.
    ref_locs:set[tuple] = target_locs.copy()

    # Initializes the diffusion grid, tracking steps from the nearest packet.
    pkt_grid: list[list[bool]] = [[-1 for j in range(M)] for i in range(N)]

    # Initializes the diffusion grid with the sources.
    src: tuple[int, int]
    for src in srcs[pkt]:
        pkt_grid[src.loc[0]][src.loc[1]] = 0

    # Tracks the maximum number of steps taken to reach all destinations.
    max_steps: int = 0
    # Tracks the total number of steps taken to reach all destinations.
    tot_steps: int = 0
    # lambda function to detect if a packet has reached a locations.
    reached: callable([tuple[int, int]], bool) = (
        lambda loc: pkt_grid[loc[0]][loc[1]] >= 0
    )
    # Adjacencies in the topology grid, representing the diffusion.
    adjacencies: tuple[tuple[int, int]] = ((0, 1), (0, -1), (1, 0), (-1, 0))
    # Runs the diffusion until all destinations are reached.
    while any(not reached(loc) for loc in target_locs):
        # Goes through the grid, diffusing the packet.
        i: int
        for i in range(M):
            j: int
            for j in range(N):
                # If the packet is at this location, diffuses it.
                if reached((i, j)):
                    # Goes through each adjacency.
                    adj: tuple
                    for adj in adjacencies:
                        # Calculates the adjacent location.
                        adj_loc: tuple[int, int] = (i + adj[0], j + adj[1])
                        # If the adjacent location is in the grid and has a higher
                        # or negative step count, diffuses the packet.
                        if (0 <= adj_loc[0] < M and 0 <= adj_loc[1] < N) and (
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
    assert(max_steps <= tot_steps)
    # The only case where it should be zero in a topology where buffers and
    # PEs are discrete units is when no PEs request the data.
    if max_steps == 0:
        assert(not ref_locs)

    return (max_steps, tot_steps)

# Runs the simulation until all packets are delivered.
for packet in packets:
    max_steps, tot_steps = diffuse_packet(packet)
    print(f"Packet {packet} took {max_steps} steps to deliver, with all steps being {tot_steps}.")
