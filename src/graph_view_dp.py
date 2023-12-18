from itertools import product
import numpy as np

from graph import smaller_neighbors
from graph_dp import GraphDP, GraphGRR


class GraphView:
    """
    Class to represent the local view that a user has of the obfuscated
    graph after having downloading the message from the server
    """

    def __init__(self, obfuscated_graph: GraphDP, vertex: int) -> None:
        """Constructor for the GraphView class"""
        self.graph = obfuscated_graph.graph
        self.obfuscated_graph = obfuscated_graph
        self.vertex = vertex

    def is_downloaded(self, i: int, j: int) -> bool:
        """Indicates whether the edge was downloaded by the user"""
        raise NotImplementedError

    def has_edge(self, i: int, j: int) -> bool:
        """Indicates whether the edge is present in the local view"""
        return self.is_downloaded(i, j) and self.obfuscated_graph.has_edge(i, j)

    def proba_from_one(self, i: int, j: int) -> float:
        """
        Probability of presence of an edge in the local view given that the
        edge was present in the original graph
        """
        raise NotImplementedError

    def proba_from_zero(self, i: int, j: int) -> float:
        """
        Probability of presence of an edge in the local view given that the
        edge was not present in the original graph
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

    def max_estimation(self) -> float:
        """Maximal possible estimation for an edge of the obfuscated graph"""
        raise NotImplementedError

    def count_triangles_local(self) -> float:
        """Count the number of triangles that the edge is responsible for"""
        return sum(
            self.edge_estimation(i, j)
            for i, j in filter(
                lambda x: x[0] > x[1],
                product(smaller_neighbors(self.graph, self.vertex), repeat=2),
            )
        )


class GraphViewFull(GraphView):
    """Class implementing the full view of the graph"""

    def __init__(self, obfuscated_graph, vertex):
        super(GraphViewFull, self).__init__(obfuscated_graph, vertex)

    def is_downloaded(self, i, j):
        return True

    def proba_from_one(self, i, j):
        return self.obfuscated_graph.proba_from_one(i, j)

    def proba_from_zero(self, i, j):
        return self.obfuscated_graph.proba_from_zero(i, j)

    def max_estimation(self):
        return self.obfuscated_graph.max_estimation()


class GraphViewOne(GraphView):
    """Class implementing the local view using the 4-cycle trick"""

    def __init__(self, obfuscated_graph, vertex):
        super().__init__(obfuscated_graph, vertex)
        self._max_unbiased_degree = None

    def is_downloaded(self, i, j):
        if i > j:
            i, j = j, i
        return self.obfuscated_graph.has_edge(self.vertex, i)

    def proba_download(self, i: int, j: int) -> float:
        """Probability that the edge was downloaded"""
        if i > j:
            i, j = j, i
        return self.obfuscated_graph.proba_from_one(self.vertex, i)

    def proba_from_one(self, i, j):
        return self.proba_download(i, j) * self.obfuscated_graph.proba_from_one(i, j)

    def proba_from_zero(self, i, j):
        return self.proba_download(i, j) * self.obfuscated_graph.proba_from_zero(i, j)

    def max_estimation(self):
        return (
            self.obfuscated_graph.max_alpha()
            / self.obfuscated_graph.min_proba_from_one()
        )

    def count_triangles_local(self):
        count = 0
        for i, u in enumerate(sorted(smaller_neighbors(self.graph, self.vertex))):
            count -= i * self.beta(u, 0)
            if not self.obfuscated_graph.has_edge(self.vertex, u):
                continue
            count += sum(
                self.alpha(u, v) * self.has_edge(u, v)
                for v in smaller_neighbors(self.graph, self.vertex)
                if v > u
            )
        return count


class GraphViewCSS(GraphView):
    """Class implementing the local view using central server sampling"""

    def __init__(self, obfuscated_graph: GraphGRR, vertex, sampling_rate: float):
        super().__init__(obfuscated_graph, vertex)
        self.sampling_rate = sampling_rate

    def is_downloaded(self, i, j):
        if i > j:
            i, j = j, i
        return self._is_bin_downloaded(
            j, self.obfuscated_graph.partition_set[j].bin_of(i)
        )

    def _is_bin_downloaded(self, vertex: int, bin_id: int) -> bool:
        """Indicates whether a bin was downloaded"""
        rng = np.random.default_rng(seed=abs(hash((self.vertex, vertex, bin_id))))
        return rng.random() < self.sampling_rate

    def proba_download(self) -> float:
        """Probability that an edge was downloaded"""
        return self.sampling_rate

    def proba_from_one(self, i, j):
        return self.proba_download() * self.obfuscated_graph.proba_from_one(i, j)

    def proba_from_zero(self, i, j):
        return self.proba_download() * self.obfuscated_graph.proba_from_zero(i, j)

    def max_estimation(self):
        return self.obfuscated_graph.max_alpha() / self.proba_download()


class GraphDownloadScheme:
    """
    Generic class that represents a download scheme for graph views.
    The main purpose of this class is to output a GraphView object for
    each vertex id
    """

    def __init__(self, obfuscated_graph: GraphDP) -> None:
        """Constructor for GraphDownloadScheme"""
        self.obfuscated_graph = obfuscated_graph

    def get_local_view(self, vertex: int) -> GraphView:
        """Returns the local view of the given vertex"""
        raise NotImplementedError

    def download_cost(self) -> float:
        """Returns the download cost of each local view"""
        raise NotImplementedError


class FullDownload(GraphDownloadScheme):
    """
    Class implementing the trivial download scheme where every edge gets
    downloaded
    """

    def get_local_view(self, vertex):
        return GraphViewFull(self.obfuscated_graph, vertex)

    def download_cost(self):
        return self.obfuscated_graph.download_cost()


class OneDownload(GraphDownloadScheme):
    """Class implementing the 4-cycle trick download scheme"""

    def get_local_view(self, vertex):
        return GraphViewOne(self.obfuscated_graph, vertex)

    def download_cost(self):
        return (
            self.obfuscated_graph.download_cost()
            * self.obfuscated_graph.min_proba_from_one()
        )


class CSSDownload(GraphDownloadScheme):
    """Class implementing the central server sampling trick download scheme"""

    def __init__(self, obfuscated_graph: GraphGRR, sampling_rate: float):
        super(CSSDownload, self).__init__(obfuscated_graph)
        self.sampling_rate = sampling_rate

    def get_local_view(self, vertex):
        return GraphViewCSS(self.obfuscated_graph, vertex, self.sampling_rate)

    def download_cost(self):
        return self.obfuscated_graph.download_cost() * self.sampling_rate
