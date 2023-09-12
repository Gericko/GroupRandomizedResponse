from itertools import product
from math import ceil
import heapq
import numpy as np

from smooth_sensitivity import SmoothAccessMechanism
from graph import smaller_neighbors, down_degree
from graph_dp import estimate_down_degrees, GraphGRR, GraphARR


GAMMA = 4

GRR_DEGREE_SHARE = 0.1
GRR_GRAPH_SHARE = 0.45
GRR_COUNT_SHARE = 0.45

ARR_GRAPH_SHARE = 0.5
ARR_COUNT_SHARE = 0.5


def get_max_neighbors(vertex_id, graph, obfuscated_graph, k=1):
    max_neighbors = heapq.nlargest(
        k,
        (
            sum(
                obfuscated_graph.has_edge(v, w)
                for w in graph.neighbors(vertex_id)
                if w < vertex_id and w != v
            )
            for v in graph.nodes
            if v < vertex_id
        ),
    )
    return max_neighbors + (k - len(max_neighbors)) * [0]


def get_smooth_sensitivity_triangles_grr(beta, vertex_id, graph, obfuscated_graph):
    max_neighbors = get_max_neighbors(vertex_id, graph, obfuscated_graph, ceil(2 / beta) + 1)

    def local_sensitivity(k):
        return (
            obfuscated_graph.alpha * sum(max_neighbors[:k])
            - obfuscated_graph.betas[vertex_id] * k * down_degree(graph, vertex_id)
            + obfuscated_graph.alpha * k * (k - 1) / 2
            - obfuscated_graph.betas[vertex_id] * k * (k - 1) / 2
        )

    def smooth_bound(k):
        return np.exp(-beta * k) * local_sensitivity(k)

    return max(smooth_bound(i) for i in range(ceil(2 / beta) + 1))


def count_triangles_local(vertex_id, graph, obfuscated_graph):
    return sum(
        obfuscated_graph.edge_estimation(i, j)
        for i, j in filter(
            lambda x: x[0] > x[1],
            product(smaller_neighbors(graph, vertex_id), repeat=2),
        )
    )


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


class SmoothLocalTriangleCountingGRR(SmoothAccessMechanism):
    def __init__(self, epsilon, gamma, graph, obfuscated_graph):
        super(SmoothLocalTriangleCountingGRR, self).__init__(epsilon, gamma)
        self.graph = graph
        self.obfuscated_graph = obfuscated_graph

    def function(self, x):
        return count_triangles_local(x, self.graph, self.obfuscated_graph)

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_triangles_grr(
            self.beta, x, self.graph, self.obfuscated_graph
        )


def count_triangles_smooth_grr(graph, obfuscated_graph, counting_budget):
    publishing_mechanism = SmoothLocalTriangleCountingGRR(
        counting_budget, GAMMA, graph, obfuscated_graph
    )
    return tuple_sum(
        (publishing_mechanism.publish(vertex_id) for vertex_id in graph.nodes),
        output_size=2,
    )


def estimate_triangles_smooth_grr(graph, privacy_budget, sample_size):
    degree_budget = GRR_DEGREE_SHARE * privacy_budget
    publishing_budget = GRR_GRAPH_SHARE * privacy_budget
    counting_budget = GRR_COUNT_SHARE * privacy_budget

    down_degrees = estimate_down_degrees(graph, degree_budget)

    obfuscated_graph = GraphGRR(graph, publishing_budget, sample_size, down_degrees)

    count, noise = count_triangles_smooth_grr(graph, obfuscated_graph, counting_budget)
    return count, 0, noise, obfuscated_graph.download_cost()


def estimate_triangles_smooth_grr_without_count(graph, privacy_budget, sample_size):
    down_degrees = estimate_down_degrees(graph, np.inf)

    obfuscated_graph = GraphGRR(graph, privacy_budget, sample_size, down_degrees)

    count = sum(
        count_triangles_local(vertex_id, graph, obfuscated_graph)
        for vertex_id in graph.nodes
    )

    return count, 0, 0, obfuscated_graph.download_cost()


def get_max_contributions_arr(vertex_id, graph, obfuscated_graph, k=1):
    max_contributions = heapq.nlargest(
        k,
        (
            sum(
                obfuscated_graph.alpha * obfuscated_graph.is_observed(v, w, vertex_id)
                - obfuscated_graph.betas[vertex_id]
                for w in graph.neighbors(vertex_id)
                if w < vertex_id and w != v
            )
            for v in graph.nodes
            if v < vertex_id
        ),
    )
    return max_contributions + (k - len(max_contributions)) * [0]


def get_smooth_sensitivity_triangles_arr(beta, vertex_id, graph, obfuscated_graph):
    max_contributions = get_max_contributions_arr(vertex_id, graph, obfuscated_graph)

    def local_sensitivity(k):
        return sum(max_contributions[:k]) + (
            obfuscated_graph.alpha - obfuscated_graph.betas[vertex_id]
        ) * (k * (k - 1) / 2)

    def smooth_bound(k):
        return np.exp(-beta * k) * local_sensitivity(k)

    return max(smooth_bound(i) for i in range(ceil(2 / beta) + 1))


def count_triangles_local_arr(vertex_id, graph, obfuscated_graph):
    return sum(
        obfuscated_graph.edge_estimation(i, j, vertex_id)
        for i, j in filter(
            lambda x: x[0] > x[1],
            product(smaller_neighbors(graph, vertex_id), repeat=2),
        )
    )


class SmoothLocalTriangleCountingARR(SmoothAccessMechanism):
    def __init__(self, epsilon, gamma, graph, obfuscated_graph):
        super(SmoothLocalTriangleCountingARR, self).__init__(epsilon, gamma)
        self.graph = graph
        self.obfuscated_graph = obfuscated_graph

    def function(self, x):
        return count_triangles_local_arr(x, self.graph, self.obfuscated_graph)

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_triangles_arr(
            self.beta, x, self.graph, self.obfuscated_graph
        )


def count_triangles_smooth_arr(graph, obfuscated_graph, counting_budget):
    publishing_mechanism = SmoothLocalTriangleCountingARR(
        counting_budget, GAMMA, graph, obfuscated_graph
    )
    return tuple_sum(
        (publishing_mechanism.publish(vertex_id) for vertex_id in graph.nodes),
        output_size=2,
    )


def estimate_triangles_smooth_arr(graph, privacy_budget, sample_size):
    publishing_budget = ARR_GRAPH_SHARE * privacy_budget
    counting_budget = ARR_COUNT_SHARE * privacy_budget
    sample_rate = (
        np.exp(publishing_budget) / (np.exp(publishing_budget) + 1) / sample_size
    )

    obfuscated_graph = GraphARR(graph, publishing_budget, sample_rate)

    count, noise = count_triangles_smooth_arr(graph, obfuscated_graph, counting_budget)
    return count, 0, noise, obfuscated_graph.download_cost()


def estimate_triangles_smooth_arr_without_count(graph, privacy_budget, sample_size):
    sample_rate = np.exp(privacy_budget) / (np.exp(privacy_budget) + 1) / sample_size

    obfuscated_graph = GraphARR(graph, privacy_budget, sample_rate)

    count = sum(
        count_triangles_local_arr(vertex_id, graph, obfuscated_graph)
        for vertex_id in graph.nodes
    )

    return count, 0, 0, obfuscated_graph.download_cost()
