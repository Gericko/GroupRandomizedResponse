import pytest
import networkx as nx
from src.graph import (
    extract_random_subgraph,
    remove_unconnected,
    get_largest_connected_component,
    get_largest_bipartite_decomposition,
    get_vertices_rank_dict,
    get_reorder_graph,
    smaller_neighbors,
    down_degree,
)


G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])


def test_extract_random_subgraph():
    assert extract_random_subgraph(G, 3).number_of_nodes() == 3


def test_remove_unconnected():
    unconnected_graph = nx.Graph()
    unconnected_graph.add_nodes_from([1, 2, 3])
    unconnected_graph.add_edge(1, 2)
    remove_unconnected(unconnected_graph)
    assert set(unconnected_graph.nodes) == {1, 2}


def test_get_largest_connected_component():
    unconnected_graph = nx.Graph()
    unconnected_graph.add_edges_from([(1, 2), (1, 3), (2, 3)] + [(4, 5)])
    assert set(get_largest_connected_component(unconnected_graph).nodes) == {1, 2, 3}


def test_get_largest_bipartite_decomposition():
    bipartite_graph = nx.Graph()
    bipartite_graph.add_edges_from([(1, 4), (1, 5), (2, 4), (3, 5)])
    decomposed_graph = nx.Graph()
    decomposed_graph.add_edges_from([(1, 2), (1, 3)])
    assert nx.is_isomorphic(
        get_largest_bipartite_decomposition(bipartite_graph), decomposed_graph
    )


def test_get_vertices_rank_dict():
    graph = nx.Graph()
    graph.add_edges_from([(3, 2), (3, 1), (3, 4), (2, 4)])
    rank = get_vertices_rank_dict(graph)
    assert rank == {3: 0, 2: 1, 4: 2, 1: 3} or rank == {3: 0, 4: 1, 2: 2, 1: 3}


def test_get_reorder_graph():
    graph = nx.Graph()
    graph.add_edges_from([(3, 2), (3, 1), (3, 4), (2, 4)])
    reordered_graph = get_reorder_graph(graph)
    assert nx.is_isomorphic(reordered_graph, graph)


def test_iter_smaller_neighbors():
    assert set(smaller_neighbors(G, 2)) == {1}


def test_down_degree():
    assert down_degree(G, 1) == 0
    assert down_degree(G, 3) == 2
