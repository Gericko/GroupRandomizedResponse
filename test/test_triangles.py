import pytest
import numpy as np
import networkx as nx
from src.graph import load_wiki
from src.triangles import tuple_sum, estimate_triangles


G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])
WIKI = load_wiki()


def test_tuple_sum():
    assert tuple_sum(((i, i+1) for i in range(3)), output_size=2) == (3, 6)
    assert tuple_sum(((i, i + 1) for i in range(0)), output_size=2) == (0, 0)


def test_estimation_triangle():
    assert estimate_triangles(G, np.inf, np.inf, np.inf, 1) == (1, 0, 0)
    assert estimate_triangles(WIKI, np.inf, np.inf, np.inf, 32) == (sum(nx.triangles(WIKI).values()) / 3, 0, 0)
