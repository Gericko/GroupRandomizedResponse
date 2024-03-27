from math import ceil
import numpy as np
from itertools import combinations

from graph_dp import GraphDP, GraphARR, GraphGRR, estimate_down_degrees
from smooth_sensitivity import SmoothAccessMechanism


GAMMA = 4
DEGREE_SHARE = 0.1
GRAPH_SHARE = 0.45
COUNT_SHARE = 0.45


def get_max_shared_2_hop_neighbors(vertex_id, graph, obfuscated_graph: GraphDP):
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


def get_max_common_neighbors(graph, obfuscated_graph: GraphDP):
    return max(
        (
            obfuscated_graph.smaller_neighbors(i)
            * obfuscated_graph.smaller_neighbors(j)
        ).sum()
        for i, j in combinations(graph.nodes, 2)
    )


def get_smooth_sensitivity_cycles(
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


def count_cycles_local(vertex_id, graph, obfuscated_graph: GraphDP):
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


class SmoothLocalCycleCounting(SmoothAccessMechanism):
    def __init__(self, epsilon, gamma, graph, obfuscated_graph):
        super(SmoothLocalCycleCounting, self).__init__(epsilon, gamma)
        self.graph = graph
        self.obfuscated_graph = obfuscated_graph
        self.max_common_neighbors = get_max_common_neighbors(graph, obfuscated_graph)

    def function(self, x):
        return count_cycles_local(x, self.graph, self.obfuscated_graph)

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_cycles(
            self.beta, x, self.graph, self.obfuscated_graph, self.max_common_neighbors
        )


def count_cycles_smooth(graph, obfuscated_graph, counting_budget):
    publishing_mechanism = SmoothLocalCycleCounting(
        counting_budget, GAMMA, graph, obfuscated_graph
    )
    return tuple_sum(
        (publishing_mechanism.publish(vertex_id) for vertex_id in graph.nodes),
        output_size=3,
    )


def estimate_cycles_grr(graph, privacy_budget, sample_size):
    degree_budget = DEGREE_SHARE * privacy_budget
    publishing_budget = GRAPH_SHARE * privacy_budget
    counting_budget = COUNT_SHARE * privacy_budget

    down_degrees = estimate_down_degrees(graph, degree_budget)

    sample = np.ceil(sample_size ** (1 / 2))
    obfuscated_graph = GraphGRR(graph, publishing_budget, sample, down_degrees)

    count, bias, noise = count_cycles_smooth(graph, obfuscated_graph, counting_budget)
    return count, bias, noise, obfuscated_graph.download_cost()


def estimate_cycles_arr(graph, privacy_budget, sample_size):
    publishing_budget = privacy_budget / 2
    counting_budget = privacy_budget / 2

    mu = np.exp(publishing_budget) / (np.exp(publishing_budget) + 1) / sample_size
    obfuscated_graph = GraphARR(graph, publishing_budget, mu)

    count, bias, noise = count_cycles_smooth(graph, obfuscated_graph, counting_budget)
    return count, bias, noise, obfuscated_graph.download_cost()
