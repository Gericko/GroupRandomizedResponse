import numpy as np
from scipy.stats import laplace
from itertools import product

from dp_tools import make_clip_dict
from graph import smaller_neighbors
from graph_dp import GraphGRR


ALPHA = 150


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


def count_triangles(graph, obfuscated_graph, counting_budget, degrees):
    def fixed_edge(vertex_id, j, contributions):
        count, bias = 0, 0
        for i in filter(lambda x: x < j, smaller_neighbors(graph, vertex_id)):
            unbiased_edge = obfuscated_graph.edge_estimation(i, j)
            count += unbiased_edge
            if (
                contributions[i] - unbiased_edge >= 0
                and contributions[j] - unbiased_edge >= 0
            ):
                contributions[i] -= unbiased_edge
                contributions[j] -= unbiased_edge
            else:
                bias -= unbiased_edge
        return count, bias

    def clipping_threshold(vertex_id, k=10):
        alpha = obfuscated_graph.get_alpha()
        beta = obfuscated_graph.get_beta_max()
        variance_max = (
            (1 + beta)
            * (alpha - 1 - beta)
            * max(ALPHA + degrees[vertex_id] - 1, 0)
            * (
                1
                + obfuscated_graph.sample_size
                / obfuscated_graph.graph.number_of_nodes()
                * max(ALPHA + degrees[vertex_id] - 2, 0)
            )
        )
        return max(0, ALPHA + degrees[vertex_id] + k * np.sqrt(variance_max))

    def count_triangles_local(vertex_id):
        clipping_thres = clipping_threshold(vertex_id)
        contributions = {i: clipping_thres for i in smaller_neighbors(graph, vertex_id)}
        return (
            *tuple_sum(
                (
                    fixed_edge(vertex_id, j, contributions)
                    for j in smaller_neighbors(graph, vertex_id)
                ),
                output_size=2,
            ),
            laplace(0, clipping_thres / counting_budget).rvs(),
        )

    return tuple_sum(
        (count_triangles_local(vertex_id) for vertex_id in graph.nodes), output_size=3
    )


def estimate_triangles(
    graph, degree_budget, publishing_budget, counting_budget, sample_size
):
    noise = laplace(0, 1 / degree_budget)
    degrees = {n: d + noise.rvs() for n, d in graph.degree()}

    obfuscated_graph = GraphGRR(graph, publishing_budget, sample_size)

    return count_triangles(graph, obfuscated_graph, counting_budget, degrees)
