import numpy as np

from graph_dp import GraphGRR, GraphARR, estimate_down_degrees, estimate_degrees
from graph_view_dp import FullDownload, OneDownload, CSSDownload
from smooth_triangles import SmoothLocalTriangleCounting
from clipping_triangles import ChernoffClip, ChebyshevClip

DEGREE_SHARE = 0.1
GRAPH_SHARE = 0.5
COUNT_SHARE = 0.5


class RawTriangleCounting:
    def __init__(self, graph, graph_download_scheme):
        self.graph = graph
        self.graph_download_scheme = graph_download_scheme

    def publish(self, vertex_id):
        local_view = self.graph_download_scheme.get_local_view(vertex_id)
        return local_view.count_triangles_local(), 0, 0


def tuple_sum(iter, output_size=0):
    return tuple(sum(x) for x in zip([0] * output_size, *iter))


class TriangleEstimator:
    def __init__(self, graph, privacy_budget, sample_size, steps):
        self.graph = graph
        self.privacy_budget = privacy_budget
        self.sample_size = sample_size
        self.user_sample = sample_size
        self.server_sample = 1
        self.steps = steps
        self.degree_budget = 0
        self.publishing_budget = 0
        self.counting_budget = 0
        self.degrees = None
        self.obfuscated_graph = None
        self.graph_download_scheme = None

    def _degree_publishing(self):
        if self.steps["graph_publishing"] == "GRR":
            self.degree_budget = DEGREE_SHARE * self.privacy_budget
            self.degrees = estimate_down_degrees(self.graph, self.degree_budget)
        elif (
            self.steps["counting"] == "chernoff"
            or self.steps["counting"] == "chebyshev"
        ):
            self.degree_budget = DEGREE_SHARE * self.privacy_budget
            self.degrees = estimate_degrees(self.graph, self.degree_budget)

    def _graph_publishing(self):
        if self.steps["counting"] == "raw":
            self.publishing_budget = self.privacy_budget - self.degree_budget
        else:
            remaining_budget = self.privacy_budget - self.degree_budget
            self.publishing_budget = GRAPH_SHARE * remaining_budget
            self.counting_budget = COUNT_SHARE * remaining_budget

        if self.steps["downloading"] == "css" or self.steps["downloading"] == "one":
            if self.steps["graph_publishing"] == "GRR":
                self.user_sample = np.ceil(self.sample_size ** (1 / 3))
                self.server_sample = self.user_sample**2 / self.sample_size
            elif self.steps["graph_publishing"] == "ARR":
                self.user_sample = self.sample_size ** (1 / 2)
                self.server_sample = 1 / self.sample_size ** (1 / 2)
        elif self.steps["downloading"] == "full":
            if self.steps["graph_publishing"] == "GRR":
                self.user_sample = np.ceil(self.sample_size ** (1 / 2))
                self.server_sample = 1
            elif self.steps["graph_publishing"] == "ARR":
                self.user_sample = self.sample_size
                self.server_sample = 1

        if self.steps["graph_publishing"] == "GRR":
            self.obfuscated_graph = GraphGRR(
                self.graph, self.publishing_budget, self.user_sample, self.degrees
            )
        elif self.steps["graph_publishing"] == "ARR":
            self.user_sample_rate = (
                np.exp(self.publishing_budget)
                / (np.exp(self.publishing_budget) + 1)
                / self.user_sample
            )
            self.obfuscated_graph = GraphARR(
                self.graph, self.publishing_budget, self.user_sample_rate
            )
        else:
            raise ValueError(
                "{} is not a valid name for a graph publishing step,"
                " it has to be either 'GRR' or 'ARR'".format(
                    self.steps["graph_publishing"]
                )
            )

    def _graph_communication(self):
        if self.steps["downloading"] == "full":
            self.graph_download_scheme = FullDownload(self.obfuscated_graph)
        elif self.steps["downloading"] == "one":
            self.graph_download_scheme = OneDownload(self.obfuscated_graph)
        elif self.steps["downloading"] == "css":
            self.graph_download_scheme = CSSDownload(
                self.obfuscated_graph, self.server_sample
            )
        else:
            raise ValueError(
                "{} is not a valid name for a downloading scheme,"
                " it has to be either 'full', 'one' or 'css'".format(
                    self.steps["downloading"]
                )
            )

    def _triangle_counting(self):
        if self.steps["counting"] == "raw":
            publishing_mechanism = RawTriangleCounting(
                self.graph, self.graph_download_scheme
            )
        elif self.steps["counting"] == "smooth":
            publishing_mechanism = SmoothLocalTriangleCounting(
                self.counting_budget, self.graph, self.graph_download_scheme
            )
        elif self.steps["counting"] == "chernoff":
            if self.steps["graph_publishing"] != "ARR":
                raise ValueError(
                    "The use of Chernoff bound is only possible when ARR is used"
                )
            publishing_mechanism = ChernoffClip(
                self.counting_budget,
                self.graph,
                self.graph_download_scheme,
                self.degrees,
                self.user_sample_rate,
            )
        elif self.steps["counting"] == "chebyshev":
            if self.steps["graph_publishing"] != "GRR":
                raise ValueError(
                    "The use of Chebyshev bound is only possible when GRR is used"
                )
            s = self.user_sample
            m = self.obfuscated_graph.partition_set[0].nb_bins
            variance_max = (
                self.obfuscated_graph.max_alpha()
                * (2 + m / (m - 1) / (np.exp(self.publishing_budget) - 1))
                / self.server_sample
            )
            covariance_max = 2 * (s - 1) / (s * m - 1) * variance_max
            publishing_mechanism = ChebyshevClip(
                self.counting_budget,
                self.graph,
                self.graph_download_scheme,
                self.degrees,
                variance_max,
                covariance_max,
            )
        else:
            raise ValueError(
                "{} is not a valid name for a downloading scheme,"
                " it has to be either 'raw', 'smooth', 'chernoff' or 'chebyshev'".format(
                    self.steps["counting"]
                )
            )
        return tuple_sum(
            (publishing_mechanism.publish(vertex_id) for vertex_id in self.graph.nodes),
            output_size=3,
        )

    def publish(self):
        self._degree_publishing()
        self._graph_publishing()
        self._graph_communication()
        count, bias, noise = self._triangle_counting()
        return (
            count,
            bias,
            noise,
            self.obfuscated_graph.upload_cost(),
            self.graph_download_scheme.download_cost(),
        )
