"""Creates and runs the simulation."""

import random

from elements import Feature
from elements.buffer import Buffer
from elements.processing import PE

from pprint import pprint
from joblib import Parallel, delayed

from topology import Topology

# Length and width of the grid representing the topology.
M: int = 3
N: int = 3

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
    for x, y in product(range(M), range(N)):
        # Data for A
        if deduplicate_A:
            data_A = {('A', x, y)}
        else:
            data_A = set(('A', x, k) for k in range(N))

        if deduplicate_B:
            data_B = {('B', x, y)}
        else:
            data_B = set(('B', k, y) for k in range(M))

        data_C = {('C', x, y)}

        bufs.add(Buffer((x, y), data_A | data_B | data_C))

        data_A = set(('A', x, k) for k in range(N))
        data_B = set(('B', k, y) for k in range(M))
        pes.add(PE((x, y), data_A | data_B | data_C))

    topology = Topology((M, N), pes, bufs)

    return topology

topology: Topology = generate_topology_for_example_mapping(deduplicate_A=True)
results = Parallel(n_jobs=-1)(delayed(topology.diffuse_packet)(pkt) for pkt in topology._dsts)

for pkt, max_cycles, tot_cycles, heatmap in results:
    print(pkt, max_cycles, tot_cycles)
    for row in heatmap:
        print(row)
    print('---')
