import networkx as nx
import numpy as np
from scipy.stats import laplace
from functools import lru_cache

from graph import (
    smaller_neighbors,
    down_degree,
)
from dp_tools import (
    group_randomized_response,
    asymmetric_randomized_response,
    proba_grr_from_one,
    proba_grr_from_zero,
    proba_arr_from_one,
    proba_arr_from_zero,
    get_max_estimation_grr,
    get_max_alpha_grr,
    get_min_proba_from_one_grr,
)
from partition import PseudoRandomPartition


def estimate_degrees(graph, privacy_budget):
    return {n: laplace(d, 1 / privacy_budget).rvs() for n, d in graph.degree()}


def estimate_down_degrees(graph, privacy_budget):
    return {
        node: laplace(down_degree(graph, node), 1 / privacy_budget).rvs()
        for node in graph.nodes
    }


def get_seeds(size):
    return np.random.randint(
        np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max, size=size
    )


def get_partitions(graph, sample_size):
    return {
        node: PseudoRandomPartition(graph.number_of_nodes(), sample_size, seed)
        for node, seed in zip(graph.nodes, get_seeds(graph.number_of_nodes()))
    }


class GraphDP:
    def __init__(self, graph, privacy_budget):
        self.graph = graph
        self.privacy_budget = privacy_budget
        self._nx_graph = None

    def has_edge(self, i, j):
        raise NotImplementedError

    def proba_from_one(self, i, j):
        raise NotImplementedError

    def proba_from_zero(self, i, j):
        raise NotImplementedError

    def alpha(self, i, j):
        p1 = self.proba_from_one(i, j)
        p0 = self.proba_from_zero(i, j)
        return 1 / (p1 - p0)

    def beta(self, i, j):
        p1 = self.proba_from_one(i, j)
        p0 = self.proba_from_zero(i, j)
        return p0 / (p1 - p0)

    def edge_estimation(self, i, j):
        return self.alpha(i, j) * self.has_edge(i, j) - self.beta(i, j)

    def smaller_neighbors(self, vertex):
        vector = np.zeros(self.graph.number_of_nodes())
        for i in range(vertex):
            vector[i] = self.edge_estimation(vertex, i)
        return vector

    def to_graph(self):
        if self._nx_graph:
            return self._nx_graph
        adjacency_dict = {
            v: {w for w in self.graph.nodes if v != w and self.has_edge(v, w)}
            for v in self.graph.nodes
        }
        self._nx_graph = nx.Graph(adjacency_dict)
        return self._nx_graph

    def max_unbiased_degree(self):
        raise NotImplementedError

    def max_estimation(self):
        raise NotImplementedError

    def max_alpha(self):
        raise NotImplementedError

    def min_proba_from_one(self):
        raise NotImplementedError


def publish_edge_list_grr(graph, privacy_budget, partition_dict):
    return {
        node: group_randomized_response(
            smaller_neighbors(graph, node), privacy_budget, partition_dict[node]
        )
        for node in graph.nodes
    }


class GraphGRR(GraphDP):
    def __init__(self, graph, privacy_budget, sample_size, down_degrees):
        super(GraphGRR, self).__init__(graph, privacy_budget)
        self.sample_size = sample_size
        self.down_degrees = down_degrees
        self.partition_set = get_partitions(graph, sample_size)
        self.published_edges = publish_edge_list_grr(
            graph, privacy_budget, self.partition_set
        )
        self._max_unbiased_degree = None

    def has_edge(self, i, j):
        if i == j:
            raise ValueError("No self-loop in the obfuscated graph")
        if i > j:
            i, j = j, i
        return self.partition_set[j].bin_of(i) in self.published_edges[j]

    @lru_cache
    def proba_from_one(self, i, j):
        if i > j:
            i, j = j, i
        return proba_grr_from_one(
            self.privacy_budget, self.partition_set[j], self.down_degrees[j]
        )

    @lru_cache
    def proba_from_zero(self, i, j):
        if i > j:
            i, j = j, i
        return proba_grr_from_zero(
            self.privacy_budget, self.partition_set[j], self.down_degrees[j]
        )

    def download_cost(self):
        return (
            self.graph.number_of_nodes() ** 2
            * np.log(self.graph.number_of_nodes() / self.sample_size)
            / self.sample_size
            / (2 + self.sample_size * (np.exp(self.privacy_budget) - 1))
        )

    def max_unbiased_degree(self):
        if self._max_unbiased_degree:
            return self._max_unbiased_degree
        nx_graph = self.to_graph()
        self._max_unbiased_degree = (
            max(d for n, d in nx_graph.degree()) * self.max_estimation()
        )
        return self._max_unbiased_degree

    def max_estimation(self):
        return get_max_estimation_grr(self.privacy_budget, self.partition_set[0])

    def max_alpha(self):
        return get_max_alpha_grr(self.privacy_budget, self.partition_set[0])

    def min_proba_from_one(self):
        return get_min_proba_from_one_grr(self.partition_set[0])


def publish_edge_list_arr(graph, privacy_budget, sample_rate):
    return {
        node: asymmetric_randomized_response(
            list(smaller_neighbors(graph, node)), privacy_budget, sample_rate, node
        )
        for node in graph.nodes
    }


class GraphARR(GraphDP):
    def __init__(self, graph, privacy_budget, mu):
        super(GraphARR, self).__init__(graph, privacy_budget)
        self.sample_rate = mu
        self.published_edges = publish_edge_list_arr(graph, privacy_budget, mu)
        self._max_unbiased_degree = None

    def has_edge(self, i, j):
        if i > j:
            i, j = j, i
        return i in self.published_edges[j]

    def proba_from_one(self, i, j):
        return proba_arr_from_one(self.sample_rate)

    def proba_from_zero(self, i, j):
        return proba_arr_from_zero(self.privacy_budget, self.sample_rate)

    def download_cost(self):
        return (
            self.sample_rate
            * np.exp(-2 * self.privacy_budget)
            * self.graph.number_of_nodes() ** 2
            * np.log(self.graph.number_of_nodes())
        )

    def max_unbiased_degree(self):
        if self._max_unbiased_degree:
            return self._max_unbiased_degree
        self._max_unbiased_degree = max(
            sum(self.edge_estimation(i, j) for j in self.published_edges[i])
            for i in self.published_edges
        )
        return self._max_unbiased_degree

    def max_estimation(self):
        return self.alpha(0, 1) - self.beta(0, 1)

    def max_alpha(self):
        return self.alpha(0, 1)

    def min_proba_from_one(self):
        return proba_arr_from_one(self.sample_rate)
