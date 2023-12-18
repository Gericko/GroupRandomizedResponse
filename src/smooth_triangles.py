from math import ceil

import networkx as nx
import numpy as np

from smooth_sensitivity import SmoothAccessMechanism
from graph_view_dp import GraphView, GraphDownloadScheme


GAMMA = 4  # Parameter of the distribution used for the smooth mechanism


def get_max_contribution_from_neighbors(
    vertex_id: int, graph: nx.Graph, graph_view: GraphView
) -> float:
    """
    Computes the maximal possible change to the local triangle count that
    can incur on the addition of one neighbor, given a local view
    """
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


def get_smooth_sensitivity_triangles(
    beta: float, vertex_id: int, graph: nx.Graph, graph_view: GraphView
) -> float:
    """Smooth sensitivity of local triangle count given a local view"""
    max_contribution_from_neighbors = get_max_contribution_from_neighbors(
        vertex_id, graph, graph_view
    )

    def local_sensitivity(k: int) -> float:
        """Local sensitivity at distance k"""
        return max_contribution_from_neighbors + graph_view.max_estimation() * k

    def smooth_bound(k: int) -> float:
        return np.exp(-beta * k) * local_sensitivity(k)

    return max(smooth_bound(k) for k in range(ceil(1 / beta) + 1))


class SmoothLocalTriangleCounting(SmoothAccessMechanism):
    """Class implementing the Smooth Access Mechanism for triangle counting"""

    def __init__(self, epsilon, graph, graph_download_scheme):
        super(SmoothLocalTriangleCounting, self).__init__(epsilon, GAMMA)
        self.graph = graph
        self.graph_download_scheme = graph_download_scheme

    def function(self, x):
        return self.graph_download_scheme.get_local_view(x).count_triangles_local()

    def smooth_sensitivity(self, x):
        return get_smooth_sensitivity_triangles(
            self.beta, x, self.graph, self.graph_download_scheme.get_local_view(x)
        )
