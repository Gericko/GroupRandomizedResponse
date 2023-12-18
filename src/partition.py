import math
import numpy as np
import sympy


class PseudoRandomPartition:
    """
    Class performing the partition of a set into several bins in a
    reproducible fashion and such that the number of elements by bins
    varies by at most 1 across all bins.
    """

    def __init__(self, data_size, bin_size, seed):
        """Constructor of the class"""
        self.data_size = sympy.nextprime(data_size)
        self.bin_size = bin_size
        self.nb_bins = math.ceil(self.data_size / self.bin_size)
        self.seed = seed

        rng = np.random.default_rng(seed=self.seed)
        self.multiplier = rng.integers(1, self.data_size)
        self.addition_constant = rng.integers(self.data_size)

    def bin_of(self, x: int):
        """Returns the bin of element x in this partition"""
        return (
            (self.multiplier * x + self.addition_constant) % self.data_size
        ) % self.nb_bins
