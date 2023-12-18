from typing import Iterable
import networkx as nx
from networkx.algorithms import bipartite
from random import sample
from time import time
from os import PathLike
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


def size_of_graph(g: nx.Graph) -> int:
    """Size occupied by the graph in memory

    :param g: Graph
    :return: Size in memory
    """
    return sum(sys.getsizeof(e) for e in g.edges) + sum(
        sys.getsizeof(n) for n in g.nodes
    )


def show_infos(g: nx.Graph) -> None:
    """Prints information about the graph

    :param g: Graph
    """
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


def extract_random_subgraph(graph: nx.Graph, size: int) -> nx.Graph:
    """
    Extracts of random subgraph of a given from a graph by randomly
    sampling from the original nodes

    :param graph: Graph
    :param size: Size of the resulting subgraph
    :return: The subgraph
    """
    vertices = sample(list(graph.nodes), size)
    subgraph = graph.subgraph(vertices)
    return nx.convert_node_labels_to_integers(subgraph)


def get_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Extracts the largest connected component of a graph

    :param graph: Graph
    :return: The graph containing only the largest connected component
    """
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component)


def get_largest_bipartite_decomposition(graph: nx.Graph) -> nx.Graph:
    """
    Transforms a bipartite graph into a new graph whose set of node is the
    largest component of the bipartite decomposition and in which two nodes
    are connected if they shared a neighbor in the bipartite graph

    :param graph: Graph
    :return: The projected graph
    """
    bottom_nodes, top_nodes = bipartite.sets(graph)
    if len(bottom_nodes) > len(top_nodes):
        bottom_nodes, top_nodes = top_nodes, bottom_nodes
    return bipartite.projected_graph(graph, top_nodes)


def load_graph_from_file(file_name: str | PathLike[str]) -> nx.Graph:
    """Reads a graph store as an edgelist from a file

    :param file_name: File in which th egraph is stored
    :return: The stored graph
    """
    print("Loading graph from {}...".format(file_name))
    starting_time = time()
    graph = nx.read_edgelist(DIR_DATA / file_name, nodetype=int)
    graph = nx.convert_node_labels_to_integers(graph)
    duration = time() - starting_time
    print("The loading took {} seconds.".format(duration))
    return graph


def get_clean_imdb() -> None:
    """Saves a graph representing the IMDB dataset after preprocessing"""
    graph = load_graph_from_file(DATA_FILE_IMDB)
    graph_connected = get_largest_connected_component(graph)
    graph_bipartite = get_largest_bipartite_decomposition(graph_connected)
    graph_clean = get_largest_connected_component(graph_bipartite)
    nx.write_edgelist(graph_clean, CLEAN_IMDB, data=False)


def load_imdb() -> nx.Graph:
    """Loads the preprocessed IMDB dataset

    :return: The graph representing the IMDB dataset
    """
    graph = load_graph_from_file(DIR_DATA / CLEAN_IMDB)
    show_infos(graph)
    return graph


def get_clean_orkut() -> None:
    """Saves a graph representing the Orkut dataset after preprocessing"""
    graph = load_graph_from_file(DIR_DATA / DATA_FILE_ORKUT)
    graph = get_largest_connected_component(graph)
    nx.write_edgelist(graph, CLEAN_ORKUT, data=False)


def load_orkut() -> nx.Graph:
    """Loads the preprocessed Orkut dataset

    :return: The graph representing the Orkut dataset
    """
    graph = load_graph_from_file(DIR_DATA / CLEAN_ORKUT)
    show_infos(graph)
    return graph


def load_gplus() -> nx.Graph:
    """Loads the preprocessed Gplus dataset

    :return: The graph representing the Gplus dataset
    """
    graph = load_graph_from_file(DIR_DATA / GPLUS)
    show_infos(graph)
    return graph


def load_wiki() -> nx.Graph:
    """Loads the preprocessed Wiki dataset

    :return: The graph representing the Wiki dataset
    """
    graph = load_graph_from_file(DIR_DATA / WIKI)
    show_infos(graph)
    return graph


def smaller_neighbors(graph: nx.Graph, vertex_id: int) -> Iterable[int]:
    """
    Returns an iterator over all the neighbors of a nodes that have a
    smaller index than that node

    :param graph: Graph
    :param vertex_id: Index of the node to get neighbors for
    :return: Iterator over the neighbors
    """
    return (node for node in graph.neighbors(vertex_id) if node < vertex_id)


def down_degree(graph: nx.Graph, vertex_id: int) -> int:
    """
    Returns the number neighbors of a nodes that have a smaller index
    than that node

    :param graph: Graph
    :param vertex_id: Index of the node to get neighbors for
    :return: Number neighbors
    """
    return len(list(smaller_neighbors(graph, vertex_id)))


def cycles(graph: nx.Graph, cycle_size: int) -> Iterable[list[int]]:
    """Returns an iterator over all the cycles of a given length

    :param graph: Graph
    :param cycle_size: the length of the cycles to return
    :return: Iterator over the cycles
    """
    cycle_gen = nx.simple_cycles(graph, length_bound=cycle_size)
    return (cycle for cycle in cycle_gen if len(cycle) == cycle_size)


def cycle_count(graph: nx.Graph, cycle_size: int) -> int:
    """Returns the number cycles of a given length

    :param graph: Graph
    :param cycle_size: the length of the cycles to return
    :return: Number of cycles
    """
    return len(list(cycles(graph, cycle_size)))
