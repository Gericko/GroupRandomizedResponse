import pytest
import numpy as np
from src.partition import PseudoRandomPartition
from src.dp_tools import (
    randomized_response,
    add_laplacian_noise,
    group_randomized_response,
    unbiased_grr_count,
)


def test_randomized_response():
    value_list = np.array([0, 1, 1, 0, 0, 1])
    assert np.all(randomized_response(value_list, np.inf) == value_list)


def test_add_laplacian_noise():
    value_list = np.array([1, 27, 42, 0])
    assert np.all(add_laplacian_noise(value_list, np.inf, 1) == value_list)


def test_group_randomized_response(monkeypatch):
    partition = PseudoRandomPartition(100, 100, 42)

    def bin_of_return(x):
        return x
    monkeypatch.setattr(partition, "bin_of", bin_of_return)

    index_set = {1, 42, 5, 67}
    assert group_randomized_response(index_set, np.inf, partition) == index_set


def test_unbiased_grr_count():
    data = np.array([True, False, False, False, True, True])
    assert unbiased_grr_count(data, np.inf, 1, 100, 1) == 3
