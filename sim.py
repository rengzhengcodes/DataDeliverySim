"""Creates and runs the simulation."""

from itertools import product
from typing import Any

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

from elements.buffer import Buffer
from elements.processing import PE


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


def generate_topology_for_example_mapping(deduplicate_a=False, deduplicate_b=False):
    """
    Generates an example topology for the mapping above.

    @param deduplicate_a    Whether to deduplicate the data for A.
    @param deduplicate_b    Whether to deduplicate the data for B.
    """
    bufs = set()
    pes = set()
    for i, j in product(range(M), range(N)):
        # Data for A
        if deduplicate_a:
            data_a = {("A", i, j)}
        else:
            data_a = set(("A", i, k) for k in range(N))

        if deduplicate_b:
            data_b = {("B", i, j)}
        else:
            data_b = set(("B", k, j) for k in range(M))

        data_c = {("C", i, j)}

        bufs.add(Buffer((i, j), data_a | data_b | data_c))

        data_a = set(("A", i, k) for k in range(N))
        data_b = set(("B", k, j) for k in range(M))
        pes.add(PE((i, j), data_a | data_b | data_c))

    topo = Topology((M, N), pes, bufs)

    return topo


topology: Topology = generate_topology_for_example_mapping(deduplicate_a=True)
results = Parallel(n_jobs=-1)(
    delayed(topology.diffuse_packet)(pkt) for pkt in topology._dsts
)

# Converts the results into a dictionary.
res_dict: dict = {}

pkt: Any
max_cycles: int
tot_cycles: int
heatmap: list
for pkt, max_cycles, tot_cycles, heatmap in results:
    res_dict[pkt] = (max_cycles, tot_cycles, heatmap)

# Builds the heatmap by summing the heatmaps of each packet, with non-negative
# values being treated as 1.
tot_heatmap = np.zeros((M, N))
for pkt, (max_cycles, tot_cycles, heatmap) in res_dict.items():
    tot_heatmap += np.array(heatmap) >= 0

print(tot_heatmap)

# Plots the heatmap.
plt.imshow(tot_heatmap, cmap="hot", interpolation="nearest")
plt.show()

# Plots a histogram of the max number of cycles taken for each packet.
plt.hist([max_cycles for _, (max_cycles, tot_cycles, heatmap) in res_dict.items()])
# Calls the x-axis distance.
plt.xlabel("Distance")
# Calls the y-axis number of cycles..
plt.ylabel("Number of Cycles")
plt.show()

# Plots a histogram of the total number of cycles taken for each packet.
plt.hist([tot_cycles for _, (max_cycles, tot_cycles, heatmap) in res_dict.items()])
# Calls the x-axis distance.
plt.xlabel("Distance")
# Calls the y-axis number of cycles.
plt.ylabel("Number of Cycles")
plt.show()
