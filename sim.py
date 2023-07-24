"""Creates and runs the simulation."""

import random

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE

from pprint import pprint

from topology import Topology

# Length and width of the grid representing the topology.
M: int = 10
N: int = 10

# Number of packets to be generated.
num_packets: int = M * N // 2

# List of packets to deliver.
packets = list(range(num_packets))

# List of packets that need to be placed into an src.
src_packets = list(packets)

# Set of all PEs.
pes: set[PE] = set()
# Set of all buffers.
bufs: set[Buffer] = set()

# Populates the topology, alternating rows of PEs and Buffers.
x: int
for x in range(M):
    y: int
    for y in range(N):
        if x % 2 == 0:
            # Chooses 3 packets that need to be delivered.
            deliveries: set[int] = set(random.sample(packets, 3))
            # Instantiates a PE with those packets.
            pes.add(PE((x, y), deliveries))
        else:
            # Chooses 1 packet from packets.
            packet: set[int] = random.choice(src_packets)
            # Removes  that packet from future packets to be put in.
            src_packets.remove(packet)
            # Instantiates a buffer with that packet.
            bufs.add(Buffer((x, y), {packet}))

topology: Topology = Topology((M, N), pes, bufs)
