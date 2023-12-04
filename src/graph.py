import numpy as np
from scipy.special import comb
import networkx as nx
from networkx.algorithms import bipartite
from random import sample
from time import time
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DATA = BASE_DIR / "data"

DATA_FILE_IMDB = "imdb_edge.csv"
CLEAN_IMDB = "imdb_clean.csv"
DATA_FILE_ORKUT = "sanitized-orkut.txt"
CLEAN_ORKUT = "orkut_clean.csv"
GPLUS = "gplus_combined.txt"
WIKI = "Wiki-Vote.txt"


def size_of_graph(g):
    return sum(sys.getsizeof(e) for e in g.edges) + sum(
        sys.getsizeof(n) for n in g.nodes
    )


def show_infos(g):
    print(
        "Main information on the graph:\n"
        "\t- Memory used: {}\n"
        "\t- number of vertices in the graph: {}\n"
        "\t- number of edges in the graph: {}\n"
        "\t- maximum degree in the graph: {}\n"
        "\t- average degree in the graph: {}\n"
        "\t- maximum down degree in the graph: {}".format(
            size_of_graph(g),
            g.number_of_nodes(),
            g.number_of_edges(),
            max(d for n, d in g.degree()),
            sum(d for n, d in g.degree()) / g.number_of_nodes(),
            max(down_degree(g, node) for node in g.nodes),
        )
    )


def extract_random_subgraph(graph, size):
    vertices = sample(list(graph.nodes), size)
    subgraph = graph.subgraph(vertices)
    return nx.convert_node_labels_to_integers(subgraph)


def remove_unconnected(graph):
    graph.remove_nodes_from(list(nx.isolates(graph)))


def get_largest_connected_component(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component)


def get_largest_bipartite_decomposition(graph):
    bottom_nodes, top_nodes = bipartite.sets(graph)
    if len(bottom_nodes) > len(top_nodes):
        bottom_nodes, top_nodes = top_nodes, bottom_nodes
    return bipartite.projected_graph(graph, top_nodes)


def load_graph_from_file(file_name):
    print("Loading graph from {}...".format(file_name))
    starting_time = time()
    graph = nx.read_edgelist(DIR_DATA / file_name, nodetype=int)
    graph = nx.convert_node_labels_to_integers(graph)
    duration = time() - starting_time
    print("The loading took {} seconds.".format(duration))
    return graph


def get_clean_imdb():
    graph = load_graph_from_file(DATA_FILE_IMDB)
    graph_connected = get_largest_connected_component(graph)
    graph_bipartite = get_largest_bipartite_decomposition(graph_connected)
    graph_clean = get_largest_connected_component(graph_bipartite)
    nx.write_edgelist(graph_clean, CLEAN_IMDB, data=False)


def load_imdb():
    graph = load_graph_from_file(DIR_DATA / CLEAN_IMDB)
    show_infos(graph)
    return graph


def get_clean_orkut():
    graph = load_graph_from_file(DIR_DATA / DATA_FILE_ORKUT)
    graph = get_largest_connected_component(graph)
    nx.write_edgelist(graph, CLEAN_ORKUT, data=False)


def load_orkut():
    graph = load_graph_from_file(DIR_DATA / CLEAN_ORKUT)
    show_infos(graph)
    return graph


def load_gplus():
    graph = load_graph_from_file(DIR_DATA / GPLUS)
    show_infos(graph)
    return graph


def load_wiki():
    graph = load_graph_from_file(DIR_DATA / WIKI)
    show_infos(graph)
    return graph


def get_vertices_rank_dict(graph):
    sorted_degrees = sorted(list(graph.degree), key=lambda a: a[1], reverse=True)
    return {n: i for i, (n, _) in enumerate(sorted_degrees)}


def get_reorder_graph(graph):
    mapping = get_vertices_rank_dict(graph)
    new_graph = nx.relabel_nodes(graph, mapping, copy=True)
    return new_graph


def smaller_neighbors(graph, vertex_id):
    return (node for node in graph.neighbors(vertex_id) if node < vertex_id)


def down_degree(graph, vertex_id):
    return len(list(smaller_neighbors(graph, vertex_id)))


def star_count(graph, star_size):
    sum(comb(d, star_size) for _, d in graph.degree())


def cycles(graph, cycle_size):
    cycle_list = nx.simple_cycles(graph, length_bound=cycle_size)
    return (cycle for cycle in cycle_list if len(cycle) == cycle_size)


def cycle_count(graph, cycle_size):
    return len(list(cycles(graph, cycle_size)))


def number_of_walks(G, walk_length):
    if walk_length < 0:
        raise ValueError(f"`walk_length` cannot be negative: {walk_length}")

    A = nx.adjacency_matrix(G, weight=None)
    power = np.linalg.matrix_power(A.toarray(), walk_length)
    result = {
        u: {v: power[u_idx, v_idx] for v_idx, v in enumerate(G)}
        for u_idx, u in enumerate(G)
    }
    return result


def walk_count(graph, path_length):
    walks = number_of_walks(graph, path_length)
    return sum(sum(value.values()) for value in walks.values())


def degeneracy(graph):
    return max(nx.core_number(graph).values())


def log_graph_infos(graph_name):
    raise NotImplementedError
