from itertools import product
from functools import lru_cache
from graph import smaller_neighbors
from dp_tools import event_with_proba


class GraphView:
    def __init__(self, obfuscated_graph, vertex):
        self.graph = obfuscated_graph.graph
        self.obfuscated_graph = obfuscated_graph
        self.vertex = vertex

    def is_downloaded(self, i, j):
        raise NotImplementedError

    def has_edge(self, i, j):
        return self.is_downloaded(i, j) and self.obfuscated_graph.has_edge(i, j)

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

    def unbiased_count(self, count, out_of):
        return self.alpha(0, 1) * count - self.beta(0, 1) * out_of

    def max_estimation(self):
        raise NotImplementedError

    def count_triangles_local(self):
        return sum(
            self.edge_estimation(i, j)
            for i, j in filter(
                lambda x: x[0] > x[1],
                product(smaller_neighbors(self.graph, self.vertex), repeat=2),
            )
        )


class GraphViewFull(GraphView):
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
    def __init__(self, obfuscated_graph, vertex):
        super().__init__(obfuscated_graph, vertex)
        self._max_unbiased_degree = None

    def is_downloaded(self, i, j):
        if i > j:
            i, j = j, i
        return self.obfuscated_graph.has_edge(self.vertex, i)

    def proba_download(self, i, j):
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
    def __init__(self, obfuscated_graph, vertex, sampling_rate):
        super().__init__(obfuscated_graph, vertex)
        self.sampling_rate = sampling_rate

    def is_downloaded(self, i, j):
        if i > j:
            i, j = j, i
        return self._is_bin_downloaded(
            j, self.obfuscated_graph.partition_set[j].bin_of(i)
        )

    @lru_cache(None)
    def _is_bin_downloaded(self, vertex, bin):
        return event_with_proba(self.sampling_rate)

    def proba_download(self):
        return self.sampling_rate

    def proba_from_one(self, i, j):
        return self.proba_download() * self.obfuscated_graph.proba_from_one(i, j)

    def proba_from_zero(self, i, j):
        return self.proba_download() * self.obfuscated_graph.proba_from_zero(i, j)

    def max_estimation(self):
        return self.obfuscated_graph.max_alpha() / self.proba_download()


class GraphDownloadScheme:
    def __init__(self, obfuscated_graph):
        self.obfuscated_graph = obfuscated_graph

    def get_local_view(self, vertex):
        raise NotImplementedError

    def download_cost(self):
        raise NotImplementedError

    def max_variance(self):
        raise NotImplementedError

    def max_covariance(self):
        raise NotImplementedError


class FullDownload(GraphDownloadScheme):
    def get_local_view(self, vertex):
        return GraphViewFull(self.obfuscated_graph, vertex)

    def download_cost(self):
        return self.obfuscated_graph.download_cost()


class OneDownload(GraphDownloadScheme):
    def get_local_view(self, vertex):
        return GraphViewOne(self.obfuscated_graph, vertex)

    def download_cost(self):
        return (
            self.obfuscated_graph.download_cost()
            * self.obfuscated_graph.min_proba_from_one()
        )


class CSSDownload(GraphDownloadScheme):
    def __init__(self, obfuscated_graph, sampling_rate):
        super(CSSDownload, self).__init__(obfuscated_graph)
        self.sampling_rate = sampling_rate

    def get_local_view(self, vertex):
        return GraphViewCSS(self.obfuscated_graph, vertex, self.sampling_rate)

    def download_cost(self):
        return self.obfuscated_graph.download_cost() * self.sampling_rate
