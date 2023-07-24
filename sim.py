"""Creates and runs the simulation."""

import random

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE

from pprint import pprint
from joblib import Parallel, delayed

from topology import Topology

# Length and width of the grid representing the topology.
M: int = 100
N: int = 100

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

"""
Mapping:
Buffer
-----
par-for m in 0..2: (x)
 par-for n in 0..2: (y)
PE
--
for k in 0..2:
"""

from itertools import product

def generate_topology_for_example_mapping(deduplicate_A=False,
                                          deduplicate_B=False):
    bufs = set()
    pes = set()
    for x, y in product(range(3), range(3)):
        # Data for A
        if deduplicate_A:
            data_A = {('A', x, y)}
        else:
            data_A = set(('A', x, k) for k in range(3))

        if deduplicate_B:
            data_B = {('B', x, y)}
        else:
            data_B = set(('B', k, y) for k in range(3))

        data_C = {('C', x, y)}

        bufs.add(Buffer((x, y), data_A | data_B | data_C))

        data_A = set(('A', x, k) for k in range(3))
        data_B = set(('B', k, y) for k in range(3))
        pes.add(PE((x, y), data_A | data_B | data_C))

    topology = Topology((3, 3), pes, bufs)

    return topology

topology: Topology = generate_topology_for_example_mapping(deduplicate_A=True)
results = Parallel(n_jobs=-1)(delayed(topology.diffuse_packet)(pkt) for pkt in topology._dsts)

for pkt, max_cycles, tot_cycles, heatmap in results:
    print(pkt, max_cycles, tot_cycles)
    for row in heatmap:
        print(row)
    print('---')
