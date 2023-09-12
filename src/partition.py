from math import ceil
import numpy as np
from sympy import nextprime


class PseudoRandomPartition:
    def __init__(self, data_size, bin_size, seed):
        self.data_size = nextprime(data_size)
        self.bin_size = bin_size
        self.nb_bins = ceil(self.data_size / self.bin_size)
        self.seed = seed

        rng = np.random.default_rng(seed=self.seed)
        self.multiplier = rng.integers(1, self.data_size)
        self.addition_constant = rng.integers(self.data_size)

    def bin_of(self, x):
        return (
            (self.multiplier * x + self.addition_constant) % self.data_size
        ) % self.nb_bins
