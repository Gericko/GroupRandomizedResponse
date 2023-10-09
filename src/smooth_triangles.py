from math import ceil
import numpy as np

from smooth_sensitivity import SmoothAccessMechanism
from graph_dp import estimate_down_degrees, GraphGRR, GraphARR
from graph_view_dp import OneDownload


GAMMA = 4

GRR_DEGREE_SHARE = 0.1
GRR_GRAPH_SHARE = 0.45
GRR_COUNT_SHARE = 0.45

ARR_GRAPH_SHARE = 0.5
ARR_COUNT_SHARE = 0.5


def get_max_contribution_from_neighbors(vertex_id, graph, graph_view):
    return max(
        (
            sum(
                graph_view.edge_estimation(v, w)
                for w in graph.neighbors(vertex_id)
                if w < vertex_id and w != v
            )
            for v in graph.nodes
            if v < vertex_id
        ),
        default=0,
    )


def get_smooth_sensitivity_triangles(beta, vertex_id, graph, graph_view):
    max_contribution_from_neighbors = get_max_contribution_from_neighbors(
        vertex_id, graph, graph_view
    )

    def local_sensitivity(k):
        return min(
            max_contribution_from_neighbors + graph_view.max_estimation() * k,
            graph_view.max_unbiased_degree(),
        )

    def smooth_bound(k):
        return np.exp(-beta * k) * local_sensitivity(k)

    return max(smooth_bound(k) for k in range(ceil(1 / beta) + 1))


class SmoothLocalTriangleCounting(SmoothAccessMechanism):
    def __init__(self, epsilon, gamma, graph, graph_download_scheme):
        super(SmoothLocalTriangleCounting, self).__init__(epsilon, gamma)
        self.graph = graph
        self.graph_download_scheme = graph_download_scheme

    def function(self, x):
        return self.graph_download_scheme.get_local_view(x).count_triangles_local()

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_triangles(
            self.beta, x, self.graph, self.graph_download_scheme.get_local_view(x)
        )


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


def count_triangles_smooth(graph, graph_download_scheme, counting_budget):
    publishing_mechanism = SmoothLocalTriangleCounting(
        counting_budget, GAMMA, graph, graph_download_scheme
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
    graph_download_scheme = OneDownload(obfuscated_graph)

    count, noise = count_triangles_smooth(graph, graph_download_scheme, counting_budget)
    return count, 0, noise, graph_download_scheme.download_cost()


def estimate_triangles_smooth_grr_without_count(graph, privacy_budget, sample_size):
    down_degrees = estimate_down_degrees(graph, np.inf)

    obfuscated_graph = GraphGRR(graph, privacy_budget, sample_size, down_degrees)
    graph_download_scheme = OneDownload(obfuscated_graph)

    count = sum(
        graph_download_scheme.get_local_view(vertex_id).count_triangles_local()
        for vertex_id in graph.nodes
    )

    return count, 0, 0, graph_download_scheme.download_cost()


def estimate_triangles_smooth_arr(graph, privacy_budget, sample_size):
    publishing_budget = ARR_GRAPH_SHARE * privacy_budget
    counting_budget = ARR_COUNT_SHARE * privacy_budget
    sample_rate = (
        np.exp(publishing_budget) / (np.exp(publishing_budget) + 1) / sample_size
    )

    obfuscated_graph = GraphARR(graph, publishing_budget, sample_rate)
    graph_download_scheme = OneDownload(obfuscated_graph)

    count, noise = count_triangles_smooth(graph, graph_download_scheme, counting_budget)
    return count, 0, noise, graph_download_scheme.download_cost()


def estimate_triangles_smooth_arr_without_count(graph, privacy_budget, sample_size):
    sample_rate = np.exp(privacy_budget) / (np.exp(privacy_budget) + 1) / sample_size

    obfuscated_graph = GraphARR(graph, privacy_budget, sample_rate)
    graph_download_scheme = OneDownload(obfuscated_graph)

    count = sum(
        graph_download_scheme.get_local_view(vertex_id).count_triangles_local()
        for vertex_id in graph.nodes
    )

    return count, 0, 0, graph_download_scheme.download_cost()
