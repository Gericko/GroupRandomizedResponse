import pytest
from collections import Counter
from src.partition import PseudoRandomPartition


class TestPseudoRandomPartition:
    def setup_method(self):
        self.partition = PseudoRandomPartition(20, 4, 42)
        self.affectations = [self.partition.bin_of(i) for i in range(self.partition.data_size)]

    def test_bin_of(self):
        for b in self.affectations:
            assert b in range(self.partition.nb_bins)

    def test_min_size_of_bin(self):
        size_of_bins = Counter(self.affectations)
        for b in range(self.partition.nb_bins):
            assert size_of_bins[b] >= self.partition.min_size_of_bin()
