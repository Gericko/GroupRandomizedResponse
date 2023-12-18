from typing import Any, Mapping
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


def estimate_degrees(graph: nx.Graph, privacy_budget: float) -> dict[int, float]:
    """Returns a dictionary linking each node to its degree estimation"""
    return {n: laplace(d, 1 / privacy_budget).rvs() for n, d in graph.degree()}


def estimate_down_degrees(graph: nx.Graph, privacy_budget: float) -> dict[int, float]:
    """Returns a dictionary linking each node to its down degree estimation"""
    return {
        node: laplace(down_degree(graph, node), 1 / privacy_budget).rvs()
        for node in graph.nodes
    }


def get_seeds(
    size: int,
) -> np.ndarray[Any, np.dtype[np.unsignedinteger]]:
    """Returns an array of random seeds for random generation initialization

    :param size: size of the return array
    :return: The array of random seeds
    """
    return np.random.randint(
        np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max, size=size
    )


def get_partitions(
    graph: nx.Graph, sample_size: int
) -> dict[int, PseudoRandomPartition]:
    """Generates a dictionary linking each node to its partition

    :param graph: Graph whose node's adjacency lists are partitioned
    :param sample_size: Bin size of each partition
    :return: The partitions dictionary
    """
    return {
        node: PseudoRandomPartition(graph.number_of_nodes(), sample_size, seed)
        for node, seed in zip(graph.nodes, get_seeds(graph.number_of_nodes()))
    }


class GraphDP:
    """A generic class that describes differential private graph publication"""

    def __init__(self, graph: nx.Graph, privacy_budget: float) -> None:
        """Constructor for the class

        :param graph: The graph to obfuscate
        :param privacy_budget: The privacy budget of the mechanism
        """
        self.graph = graph
        self.privacy_budget = privacy_budget
        self._nx_graph = None

    def has_edge(self, i: int, j: int) -> bool:
        """Indicates whether an edge exists in the obfuscated graph"""
        raise NotImplementedError

    def proba_from_one(self, i: int, j: int) -> float:
        """
        Probability of presence of an edge in the obfuscated graph given that
        the edge was present in the original graph
        """
        raise NotImplementedError

    def proba_from_zero(self, i: int, j: int) -> float:
        """
        Probability of presence of an edge in the obfuscated graph given that
        the edge was not present in the original graph
        """
        raise NotImplementedError

    def alpha(self, i: int, j: int) -> float:
        """Multiplicative unbiasing for edge estimation"""
        p1 = self.proba_from_one(i, j)
        p0 = self.proba_from_zero(i, j)
        return 1 / (p1 - p0)

    def beta(self, i: int, j: int) -> float:
        """Additive unbiasing for edge estimation"""
        p1 = self.proba_from_one(i, j)
        p0 = self.proba_from_zero(i, j)
        return p0 / (p1 - p0)

    def edge_estimation(self, i: int, j: int) -> float:
        """Unbiased estimation of the presence of an edge in the graph"""
        return self.alpha(i, j) * self.has_edge(i, j) - self.beta(i, j)

    def smaller_neighbors(self, vertex: int) -> np.ndarray[Any, np.dtype[np.float_]]:
        vector = np.zeros(self.graph.number_of_nodes())
        for i in range(vertex):
            vector[i] = self.edge_estimation(vertex, i)
        return vector

    def to_graph(self) -> nx.Graph:
        """Convert the obfuscated graph to a networkx Graph object"""
        if self._nx_graph:
            return self._nx_graph
        adjacency_dict = {
            v: {w for w in self.graph.nodes if v != w and self.has_edge(v, w)}
            for v in self.graph.nodes
        }
        self._nx_graph = nx.Graph(adjacency_dict)
        return self._nx_graph

    def upload_cost(self) -> float:
        """
        Upload cost incurred by each user for the publication of the
        obfuscated graph
        """
        raise NotImplementedError

    def download_cost(self) -> float:
        """
        Download cost incurred by each user for the publication of the
        obfuscated graph
        """
        return self.graph.number_of_nodes() * self.upload_cost()

    def max_estimation(self) -> float:
        """Maximal possible estimation for an edge of the obfuscated graph"""
        raise NotImplementedError

    def max_alpha(self) -> float:
        """
        Maximal possible multiplicative unbiasing coefficient for an edge of
        the obfuscated graph
        """
        raise NotImplementedError

    def min_proba_from_one(self) -> float:
        """
        Minimal possible probability of presence of an edge in the obfuscated
        graph given that the edge was present in the original graph
        """
        raise NotImplementedError


def publish_edge_list_grr(
    graph: nx.Graph,
    privacy_budget: float,
    partition_dict: dict[int, PseudoRandomPartition],
) -> dict[int, set[int]]:
    """
    Returns a dictionary mapping each node to the set of edges that were
    published via Group Randomized Response

    :param graph: The graph to be published
    :param privacy_budget: The privacy budget of the mechanism
    :param partition_dict: The partitions assigned to each user
    :return: The obfuscated graph in the form of adjacency lists
    """
    return {
        node: group_randomized_response(
            smaller_neighbors(graph, node), privacy_budget, partition_dict[node]
        )
        for node in graph.nodes
    }


class GraphGRR(GraphDP):
    """Class implementing graph publication via Group Randomized Response"""

    def __init__(
        self,
        graph: nx.Graph,
        privacy_budget: float,
        sample_size: int,
        down_degrees: Mapping[int, float],
    ) -> None:
        """
        Constructor for GraphGRR extending the base constructor with GRR
        specific parameters
        """
        super(GraphGRR, self).__init__(graph, privacy_budget)
        self.sample_size = sample_size
        self.down_degrees = down_degrees
        self.partition_set = get_partitions(graph, sample_size)
        self.published_edges = publish_edge_list_grr(
            graph, privacy_budget, self.partition_set
        )
        self._max_unbiased_degree = None

    def has_edge(self, i: int, j: int) -> bool:
        if i == j:
            raise ValueError("No self-loop in the obfuscated graph")
        if i > j:
            i, j = j, i
        return self.partition_set[j].bin_of(i) in self.published_edges[j]

    @lru_cache
    def proba_from_one(self, i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        return proba_grr_from_one(
            self.privacy_budget, self.partition_set[j], self.down_degrees[j]
        )

    @lru_cache
    def proba_from_zero(self, i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        return proba_grr_from_zero(
            self.privacy_budget, self.partition_set[j], self.down_degrees[j]
        )

    def upload_cost(self) -> float:
        return (
            self.graph.number_of_nodes()
            * np.log(self.graph.number_of_nodes() / self.sample_size)
            / self.sample_size
            / (2 + self.sample_size * (np.exp(self.privacy_budget) - 1))
        )

    def max_estimation(self) -> float:
        return get_max_estimation_grr(self.privacy_budget, self.partition_set[0])

    def max_alpha(self) -> float:
        return get_max_alpha_grr(self.privacy_budget, self.partition_set[0])

    def min_proba_from_one(self) -> float:
        return get_min_proba_from_one_grr(self.partition_set[0])


def publish_edge_list_arr(
    graph: nx.Graph, privacy_budget: float, sample_rate: float
) -> dict[int, set[int]]:
    """
    Returns a dictionary mapping each node to the set of edges that were
    published via Asymmetric Randomized Response

    :param graph: The graph to be published
    :param privacy_budget: The privacy budget of the mechanism
    :param sample_rate: Upload cost reduction
    :return: The obfuscated graph in the form of adjacency lists
    """
    return {
        node: asymmetric_randomized_response(
            list(smaller_neighbors(graph, node)), privacy_budget, sample_rate, node
        )
        for node in graph.nodes
    }


class GraphARR(GraphDP):
    """Class implementing graph publication via Asymmetric Randomized Response"""

    def __init__(self, graph, privacy_budget, mu):
        """
        Constructor for GraphGRR extending the base constructor with GRR
        specific parameters
        """
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

    def upload_cost(self):
        return (
            self.sample_rate
            * np.exp(-2 * self.privacy_budget)
            * self.graph.number_of_nodes()
            * np.log(self.graph.number_of_nodes())
        )

    def max_estimation(self):
        return self.alpha(0, 1) - self.beta(0, 1)

    def max_alpha(self):
        return self.alpha(0, 1)

    def min_proba_from_one(self):
        return proba_arr_from_one(self.sample_rate)
