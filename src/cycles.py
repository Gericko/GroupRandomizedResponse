from math import ceil
import numpy as np
from scipy.stats import laplace
from itertools import combinations, zip_longest
import heapq

from graph import smaller_neighbors
from graph_dp import GraphGRR, estimate_down_degrees
from smooth_sensitivity import SmoothAccessMechanism


GAMMA = 4
DEGREE_SHARE = 0.1
GRAPH_SHARE = 0.45
COUNT_SHARE = 0.45


def get_max_shared_2_hop_neighbors(vertex_id, graph, obfuscated_graph):
    mask_vertex = np.array(
        [1] * vertex_id + [0] * (graph.number_of_nodes() - vertex_id)
    )
    two_hop_neighbors = sum(
        (
            mask_vertex * obfuscated_graph.smaller_neighbors(neighbor)
            for neighbor in graph.neighbors(vertex_id)
        ),
        start=np.zeros(graph.number_of_nodes()),
    )
    return max(
        (
            (obfuscated_graph.smaller_neighbors(node) * two_hop_neighbors).sum()
            for node in graph.nodes
        ),
        default=0,
    )


def get_max_common_neighbors(graph, obfuscated_graph):
    return max(
        (
            obfuscated_graph.smaller_neighbors(i)
            * obfuscated_graph.smaller_neighbors(j)
        ).sum()
        for i, j in combinations(graph.nodes, 2)
    )


def get_smooth_sensitivity_cycles_grr(
    beta, vertex_id, graph, obfuscated_graph, max_common_neighbors
):
    max_neighbors_sharing = get_max_shared_2_hop_neighbors(
        vertex_id, graph, obfuscated_graph
    )

    def local_sensitivity(k):
        return max_neighbors_sharing + max_common_neighbors * k

    def smooth_bound(k):
        return np.exp(-beta * k) * local_sensitivity(k)

    return max(smooth_bound(i) for i in range(ceil(1 / beta) + 1))


def count_cycles_local(vertex_id, graph, obfuscated_graph):
    mask_vertex = np.array(
        [1] * vertex_id + [0] * (graph.number_of_nodes() - vertex_id)
    )
    return sum(
        (
            obfuscated_graph.smaller_neighbors(i)
            * obfuscated_graph.smaller_neighbors(j)
            * mask_vertex
        ).sum()
        for i, j in combinations(graph.neighbors(vertex_id), 2)
    )


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


class SmoothLocalCycleCountingGRR(SmoothAccessMechanism):
    def __init__(self, epsilon, gamma, graph, obfuscated_graph, max_common_neighbors):
        super(SmoothLocalCycleCountingGRR, self).__init__(epsilon, gamma)
        self.graph = graph
        self.obfuscated_graph = obfuscated_graph
        self.max_common_neighbors = max_common_neighbors

    def function(self, x):
        return count_cycles_local(x, self.graph, self.obfuscated_graph)

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_cycles_grr(
            self.beta, x, self.graph, self.obfuscated_graph, self.max_common_neighbors
        )


def count_cycles_smooth(graph, obfuscated_graph, counting_budget):
    max_common_neighbors = get_max_common_neighbors(graph, obfuscated_graph)
    publishing_mechanism = SmoothLocalCycleCountingGRR(
        counting_budget, GAMMA, graph, obfuscated_graph, max_common_neighbors
    )
    return tuple_sum(
        (publishing_mechanism.publish(vertex_id) for vertex_id in graph.nodes),
        output_size=2,
    )


def estimate_cycles_smooth(graph, privacy_budget, sample_size):
    degree_budget = DEGREE_SHARE * privacy_budget
    publishing_budget = GRAPH_SHARE * privacy_budget
    counting_budget = COUNT_SHARE * privacy_budget

    down_degrees = estimate_down_degrees(graph, degree_budget)

    obfuscated_graph = GraphGRR(graph, publishing_budget, sample_size, down_degrees)

    count, noise = count_cycles_smooth(graph, obfuscated_graph, counting_budget)
    return count, 0, noise, obfuscated_graph.download_cost()


def estimate_cycles_smooth_without_count(graph, privacy_budget, sample_size):
    down_degrees = estimate_down_degrees(graph, np.inf)

    obfuscated_graph = GraphGRR(graph, privacy_budget, sample_size, down_degrees)

    count = sum(
        count_cycles_local(vertex_id, graph, obfuscated_graph)
        for vertex_id in graph.nodes
    )

    return count, 0, 0, obfuscated_graph.download_cost()


def count_4_cycles_clip(graph, obfuscated_graph, privacy_budget, degrees):
    def count_4_cycles_local(vertex_id):
        clipping_thres = clipping_threshold(vertex_id)
        contributions = {i: clipping_thres for i in smaller_neighbors(graph, vertex_id)}
        cycles_count, bias = 0, 0
        for i, j in combinations(smaller_neighbors(graph, vertex_id), 2):
            count = np.sum(
                obfuscated_graph.smaller_neighbors(i)
                * obfuscated_graph.smaller_neighbors(j)
            )
            biased_count = min(count, contributions[i], contributions[j])
            bias += biased_count - count
            cycles_count += count
        return cycles_count, bias, laplace(0, clipping_thres / privacy_budget).rvs()

    def clipping_threshold(vertex_id, k=10):
        alpha = obfuscated_graph.get_alpha()
        beta = obfuscated_graph.get_beta_max()
        variance_edge_max = (1 + beta) * (alpha - 1 - beta)
        d_i = max(0, degrees[vertex_id])
        d_max = max(degrees.values())
        n = graph.number_of_nodes()
        s = obfuscated_graph.sample_size
        variance_cycles_max = (n + s**2) * d_i * variance_edge_max**2 + (
            2 * n * d_i
            + 2 * d_i * d_max**2 * s / n
            + d_i**2 * d_max
            + d_i**2 * d_max**2 * s / n
        ) * variance_edge_max
        return max(0, d_i + k * np.sqrt(variance_cycles_max))

    return tuple_sum(count_4_cycles_local(vertex_id) for vertex_id in graph.nodes)


def estimate_4_cycles_clip(
    graph, degree_budget, publishing_budget, counting_budget, sample_size
):
    noise = laplace(0, 1 / degree_budget)
    degrees = {n: d + noise.rvs() for n, d in graph.degree()}
    obfuscated_graph = GraphGRR(graph, publishing_budget, sample_size)
    return count_4_cycles(graph, obfuscated_graph, counting_budget, degrees)
