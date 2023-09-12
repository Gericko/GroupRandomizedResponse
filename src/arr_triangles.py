import numpy as np
from scipy.stats import laplace
from itertools import product

from dp_tools import make_clip_dict
from graph import smaller_neighbors, down_degree
from graph_dp import GraphARR


ALPHA = 150
BETA = 1e-6

DEGREE_SHARE = 0.1
GRAPH_SHARE = 0.45
COUNT_SHARE = 0.45


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


def kullback_leibler(p1, p2):
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))


class ARRLocalTriangleCounting:
    def __init__(self, epsilon, graph, obfuscated_graph, degrees):
        self.epsilon = epsilon
        self.graph = graph
        self.obfuscated_graph = obfuscated_graph
        self.degrees = degrees
        self.rv = laplace(0, 1)

    def _probability_bound(self, threshold, vertex_id):
        return np.exp(
            -self.degrees[vertex_id]
            * kullback_leibler(threshold, self.obfuscated_graph.sample_rate)
        )

    def clipping_threshold(self, vertex_id):
        if self.obfuscated_graph.sample_rate >= 1:
            return self.degrees[vertex_id]
        if self.degrees[vertex_id] <= 1:
            return 0
        lam = 1
        while (
            self._probability_bound(lam * self.obfuscated_graph.sample_rate, vertex_id)
            > BETA
        ):
            lam += 1
            if lam * self.obfuscated_graph.sample_rate >= 1:
                return self.degrees[vertex_id]
        return lam * self.obfuscated_graph.sample_rate * self.degrees[vertex_id]

    def publish(self, vertex_id):
        contributions = {
            i: self.clipping_threshold(vertex_id)
            for i in smaller_neighbors(self.graph, vertex_id)
        }
        is_clipped = make_clip_dict(
            list(smaller_neighbors(self.graph, vertex_id)),
            int(self.degrees[vertex_id]),
        )
        t_i, bias = 0, 0
        d = down_degree(self.graph, vertex_id)
        s_i = d * (d - 1) / 2
        for i, j in filter(
            lambda x: x[0] > x[1],
            product(smaller_neighbors(self.graph, vertex_id), repeat=2),
        ):
            t_i += self.obfuscated_graph.is_observed(i, j, vertex_id)
            if (
                contributions[i] >= 1
                and contributions[j] >= 1
                and not is_clipped[i]
                and not is_clipped[j]
            ):
                contributions[i] -= self.obfuscated_graph.is_observed(i, j, vertex_id)
                contributions[j] -= self.obfuscated_graph.is_observed(i, j, vertex_id)
            else:
                bias -= self.obfuscated_graph.is_observed(i, j, vertex_id)
        return (
            self.obfuscated_graph.unbiased_count(t_i, s_i),
            self.obfuscated_graph.unbiased_count(bias, 0),
            self.obfuscated_graph.unbiased_count(
                self.clipping_threshold(vertex_id) / self.epsilon * self.rv.rvs(), 0
            ),
        )

    def std(self, vertex_id):
        return self.obfuscated_graph.unbiased_count(
            self.clipping_threshold(vertex_id) / self.epsilon * self.rv.std(), 0
        )


def count_triangles_arr(graph, obfuscated_graph, counting_budget, degrees):
    publishing_mechanism = ARRLocalTriangleCounting(
        counting_budget, graph, obfuscated_graph, degrees
    )
    return tuple_sum(
        (publishing_mechanism.publish(vertex_id) for vertex_id in graph.nodes),
        output_size=3,
    )


def estimate_triangles_arr(graph, privacy_budget, sample_size):
    degree_budget = DEGREE_SHARE * privacy_budget
    publishing_budget = GRAPH_SHARE * privacy_budget
    counting_budget = COUNT_SHARE * privacy_budget
    sample_rate = (
        np.exp(publishing_budget) / (np.exp(publishing_budget) + 1) / sample_size
    )

    noise = laplace(0, 1 / degree_budget)
    degrees = {n: max(d + noise.rvs() + ALPHA, 0) for n, d in graph.degree()}

    obfuscated_graph = GraphARR(graph, publishing_budget, sample_rate)

    count, bias, noise = count_triangles_arr(
        graph, obfuscated_graph, counting_budget, degrees
    )
    return count, bias, noise, obfuscated_graph.download_cost()
