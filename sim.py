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
M: int = 6
N: int = 6
K: int = 6

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


def generate_topology_for_example_mapping(duplicate_a=1, duplicate_b=1):
    """
    Generates an example topology for the mapping above.

    @param deduplicate_a    Whether to deduplicate the data for A.
    @param deduplicate_b    Whether to deduplicate the data for B.
    """
    # Initializes the set of buffers and PEs passed to the topology.
    bufs = set()
    pes = set()

    # Creates the buffers and PEs.
    i: int
    j: int
    for i, j in product(range(M), range(N)):
        # The set of data that each buffer will hold and PE will need.
        data_a = {
            ("A", i, (duplicate_a*j + a_k)%K) for a_k in range(duplicate_a)
        }
        data_b = {
            ("B", (duplicate_b*i + b_k)%K, j) for b_k in range(duplicate_b)
        }
        data_c = {("C", i, j)}

        # Adds a buffer holding all the data above.
        bufs.add(Buffer((i, j), data_a | data_b | data_c))

        data_a = set(("A", i, k) for k in range(K))
        data_b = set(("B", k, j) for k in range(K))
        pes.add(PE((i, j), data_a | data_b | data_c))

    topo = Topology((M, N), pes, bufs)

    return topo

heatmap_fig, heatmap_axes = plt.subplots(4, 4, figsize=(10, 10))
latency_fig, latency_axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
hops_fig, hops_axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)

for i_a, duplicate_a in enumerate([1, 2, 3, 6]):
    for i_b, duplicate_b in enumerate([1, 2, 3, 6]):
        heatmap_ax = heatmap_axes[i_a, i_b]
        latency_ax = latency_axes[i_a, i_b]
        hops_ax = hops_axes[i_a, i_b]

        topology: Topology = generate_topology_for_example_mapping(duplicate_a,
                                                                   duplicate_b)
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

        # Plots the heatmap.
        heatmap_ax.imshow(tot_heatmap, cmap="hot", interpolation="nearest")

        # Plots a histogram of the max number of cycles taken for each packet.
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        latency_ax.hist([max_cycles for _, (max_cycles, tot_cycles, heatmap) in res_dict.items()])
        # Calls the x-axis distance.
        if i_a == 3:
            latency_ax.set_xlabel("Latency")
        # Calls the y-axis number of cycles..
        if i_b == 0:
            latency_ax.set_ylabel("Data Count")
        latency_ax.set_title(f'Latency of ({duplicate_a}, {duplicate_b})')

        # Plots a histogram of the total number of cycles taken for each packet.
        hops_ax.hist([tot_cycles for _, (max_cycles, tot_cycles, heatmap) in res_dict.items()])
        # Calls the x-axis distance.
        if i_a == 3:
            hops_ax.set_xlabel("Hops")
        # Calls the y-axis number of cycles.
        if i_b == 0:
            hops_ax.set_ylabel("Data Count")
        hops_ax.set_title(f'Hops of ({duplicate_a}, {duplicate_b})')

heatmap_fig.savefig('heatmap.png', dpi=400)
latency_fig.savefig('latency.png', dpi=400)
hops_fig.savefig('hops.png', dpi=400)