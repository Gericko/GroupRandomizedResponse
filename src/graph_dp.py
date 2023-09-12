import networkx as nx
import numpy as np
from scipy.stats import laplace
from itertools import combinations

from graph import (
    smaller_neighbors,
    down_degree,
)
from dp_tools import (
    group_randomized_response,
    get_grr_alpha,
    get_grr_beta,
    asymmetric_randomized_response,
    unbiased_arr_count,
    get_arr_alpha,
    get_arr_beta,
)
from partition import PseudoRandomPartition


def estimate_degrees(graph, privacy_budget):
    return {n: laplace(d, 1 / privacy_budget).rvs() for n, d in graph.degree()}


def estimate_down_degrees(graph, privacy_budget):
    return {
        node: laplace(down_degree(graph, node), 1 / privacy_budget).rvs()
        for node in graph.nodes
    }


def estimate_max_down_degree(graph, privacy_budget):
    return max(estimate_down_degrees(graph, privacy_budget).values())


def get_seeds(size):
    return np.random.randint(
        np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max, size=size
    )


def get_partitions(graph, sample_size):
    return {
        node: PseudoRandomPartition(graph.number_of_nodes(), sample_size, seed)
        for node, seed in zip(graph.nodes, get_seeds(graph.number_of_nodes()))
    }


def smaller_forks(graph, vertex_id):
    return combinations(sorted(smaller_neighbors(graph, vertex_id)), 2)


def publish_edge_list(graph, privacy_budget, partition_dict):
    return {
        node: group_randomized_response(
            smaller_neighbors(graph, node), privacy_budget, partition_dict[node]
        )
        for node in graph.nodes
    }


def get_alpha(privacy_budget, partition_dict):
    partition = next(iter(partition_dict.values()))
    return get_grr_alpha(
        privacy_budget,
        partition.bin_size,
        partition.data_size,
    )


def get_beta_dict(graph, privacy_budget, partition_dict, down_degrees):
    return {
        node: get_grr_beta(
            privacy_budget,
            partition_dict[node].bin_size,
            partition_dict[node].data_size,
            down_degrees[node] / partition_dict[node].data_size,
        )
        for node in graph.nodes
    }


class GraphGRR:
    def __init__(self, graph, privacy_budget, sample_size, down_degrees):
        self.graph = graph
        self.privacy_budget = privacy_budget
        self.sample_size = sample_size
        self.down_degrees = down_degrees
        self.partition_set = get_partitions(graph, sample_size)
        self.published_edges = publish_edge_list(
            graph, privacy_budget, self.partition_set
        )
        self.alpha = get_alpha(privacy_budget, self.partition_set)
        self.betas = get_beta_dict(
            graph, privacy_budget, self.partition_set, self.down_degrees
        )

    def has_edge(self, i, j):
        if i == j:
            return False
        if i > j:
            i, j = j, i
        return self.partition_set[j].bin_of(i) in self.published_edges[j]

    def edge_estimation(self, i, j):
        if i == j:
            raise ValueError("No self-loop in the obfuscated graph")
        if i > j:
            i, j = j, i
        return self.alpha * self.has_edge(i, j) - self.betas[j]

    def unbiase_edges(self, vector, publishing_node):
        return self.alpha * vector - self.betas[publishing_node]

    def smaller_neighbors(self, vertex):
        vector = np.zeros(self.graph.number_of_nodes())
        for i in range(vertex):
            vector[i] = int(self.has_edge(vertex, i))
        vector[:vertex] = self.unbiase_edges(vector[:vertex], vertex)
        return vector

    def get_beta_max(self):
        return max(self.betas.values())

    def download_cost(self):
        return (
            self.graph.number_of_nodes() ** 2
            * np.log(self.graph.number_of_nodes() / self.sample_size)
            / self.sample_size
            / (2 + self.sample_size * (np.exp(self.privacy_budget) - 1))
        )


def publish_edge_list_arr(graph, privacy_budget, sample_rate):
    return {
        node: asymmetric_randomized_response(
            list(smaller_neighbors(graph, node)), privacy_budget, sample_rate, node
        )
        for node in graph.nodes
    }


class GraphARR:
    def __init__(self, graph, privacy_budget, mu):
        self.graph = graph
        self.privacy_budget = privacy_budget
        self.sample_rate = mu**2
        self.published_edges = publish_edge_list_arr(graph, privacy_budget, mu)
        self.alpha = get_arr_alpha(self.privacy_budget, self.sample_rate)
        self.betas = {node: get_arr_beta(self.privacy_budget) for node in graph.nodes}

    def has_edge(self, i, j):
        if i > j:
            i, j = j, i
        return i in self.published_edges[j]

    def is_downloaded(self, i, j, vertex_id):
        if i > j:
            i, j = j, i
        return self.has_edge(vertex_id, j)

    def is_observed(self, i, j, vertex_id):
        return self.is_downloaded(i, j, vertex_id) and self.has_edge(i, j)

    def edge_estimation(self, i, j, vertex_id):
        if i > j:
            i, j = j, i
        return self.alpha * self.is_observed(i, j, vertex_id) - self.betas[j]

    def unbiased_count(self, count, out_of):
        return unbiased_arr_count(count, out_of, self.privacy_budget, self.sample_rate)

    def to_graph(self):
        return nx.Graph(self.published_edges)

    def download_cost(self):
        return (
            self.sample_rate
            * np.exp(-2 * self.privacy_budget)
            * self.graph.number_of_nodes() ** 2
            * np.log(self.graph.number_of_nodes())
        )
