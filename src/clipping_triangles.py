import numpy as np
import networkx as nx
from scipy.stats import laplace
from itertools import combinations

from graph import smaller_neighbors
from graph_view_dp import GraphView, GraphDownloadScheme

ALPHA = 150  # Margin taken on the degree estimation
BETA = 1e-6  # Probability of clipping when the Chernoff bound is used
BETA_CHEBYSHEV = 1e-3  # Probability of clipping when the Chebyshev bound is used


class ClipLocalTriangleCounting:
    """Generic class for counting triangles using clipping"""

    def __init__(
        self,
        epsilon: float,
        graph: nx.Graph,
        graph_download_scheme: GraphDownloadScheme,
        degrees: dict[int, float],
    ) -> None:
        """Constructor for ClipLocalTriangleCounting"""
        self.epsilon = epsilon
        self.graph = graph
        self.graph_download_scheme = graph_download_scheme
        self.degrees = degrees
        self.rv = laplace(0, 1)

    def threshold(self, local_view: GraphView) -> float:
        """
        Calculates the maximal contribution acceptable for an edge in the
        local triangle count

        :param local_view: Local view accessible to the user performing the local counting
        """
        raise NotImplementedError

    def local_count(
        self, local_view: GraphView, threshold: float
    ) -> tuple[float, float]:
        """Returns the triangle count performed by a given user after clipping

        :return: A tuple (count, bias) such that the triangle count after clipping is equal to count + bias
        """
        contributions = {
            i: threshold for i in smaller_neighbors(self.graph, local_view.vertex)
        }
        max_estimation = local_view.max_estimation()
        count = 0
        bias = 0
        for j, k in combinations(smaller_neighbors(self.graph, local_view.vertex), 2):
            unbiased_edge = local_view.edge_estimation(j, k)
            count += unbiased_edge
            if (
                contributions[j] >= max_estimation
                and contributions[k] >= max_estimation
            ):
                contributions[j] -= unbiased_edge
                contributions[k] -= unbiased_edge
            else:
                bias -= unbiased_edge
        return count, bias

    def publish(self, vertex_id: int) -> tuple[float, float, float]:
        """Publishes the local triangle count counted by a given user

        :param vertex_id: The user responsible for the count
        :return: A tuple (count, bias, noise) such that the obfuscated triangle
         count published by the user is equal to count + bias + noise
        """
        local_view = self.graph_download_scheme.get_local_view(vertex_id)
        threshold = self.threshold(local_view)
        count, bias = self.local_count(local_view, threshold)
        noise = threshold / self.epsilon * self.rv.rvs()
        return count, bias, noise


def kullback_leibler(p1: float, p2: float) -> float:
    """Computes the Kullback-Leibler divergence between 2 probabilities"""
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))


class ChernoffClip(ClipLocalTriangleCounting):
    """Class implementing clipping using the Chernoff bound"""

    def __init__(
        self, epsilon, graph, graph_download_scheme, degrees, proba_of_presence
    ):
        super(ChernoffClip, self).__init__(
            epsilon, graph, graph_download_scheme, degrees
        )
        self.proba_of_presence = proba_of_presence

    def _probability_bound(self, vertex: int, tentative_threshold: float) -> float:
        return np.exp(
            -(self.degrees[vertex] + ALPHA)
            * kullback_leibler(
                tentative_threshold,
                self.proba_of_presence,
            )
        )

    def threshold(self, local_view):
        safe_degree = self.degrees[local_view.vertex] + ALPHA
        if safe_degree <= 1:
            return 0
        lam = 1
        while (
            self._probability_bound(
                local_view.vertex,
                self.proba_of_presence,
            )
            > BETA
        ):
            lam += 1
            if lam * self.proba_of_presence >= 1:
                return local_view.max_estimation() * safe_degree
        return local_view.max_estimation() * lam * self.proba_of_presence * safe_degree


class ChebyshevClip(ClipLocalTriangleCounting):
    """Class implementing clipping using the Bienayme–Chebyshev inequality"""

    def __init__(
        self, epsilon, graph, graph_download_scheme, degrees, variance, covariance
    ):
        super(ChebyshevClip, self).__init__(
            epsilon, graph, graph_download_scheme, degrees
        )
        self.variance = variance
        self.covariance = covariance

    def threshold(self, local_view):
        safe_degree = self.degrees[local_view.vertex] + ALPHA
        variance_sum = safe_degree * self.variance + safe_degree**2 * self.covariance
        return safe_degree + np.sqrt(max(0, variance_sum / BETA_CHEBYSHEV))
