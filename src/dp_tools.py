import numpy as np
from scipy.stats import binom
from collections import Counter
from itertools import compress
from functools import lru_cache
from typing import Iterable

from partition import PseudoRandomPartition


def sparse_randomized_response(
    data_list: list[int] | set[int], epsilon: float, data_size: int
) -> set[int]:
    """
    Performs the Randomized Response of a collection of values, assuming
    that the size of the  collection is small compared to the size of the
    data space.

    :param data_list: List of values to randomize
    :param epsilon: Privacy budget for randomization
    :param data_size: Size of the data space
    :return: Randomized response of the data stored in a set
    """
    p = 1 / (1 + np.exp(epsilon))
    sample_size = binom(data_size, p).rvs()
    random_indices = np.random.choice(data_size, sample_size, replace=False)
    output_list_from_zeros = {x for x in random_indices if x not in data_list}
    output_list_from_ones = set(compress(data_list, np.random.rand(len(data_list)) > p))
    return output_list_from_ones | output_list_from_zeros


def event_with_proba(probability: float) -> bool:
    """Returns True with probability equal to 'probability'

    :param probability: The probability that the event occurs
    :return: A boolean indicating if the event occurred
    """
    return np.random.rand() < probability


def group_sampling(
    data_list: Iterable[int], partition: PseudoRandomPartition
) -> set[int]:
    """Samples the data by keeping only 1 value for each bin of the partition

    :param data_list: List of values to sample
    :param partition: Partition according to which data is to be grouped
    :return: A set containing the sampled values for each bin of the
    partition if they were in data_list
    """
    counts = Counter([partition.bin_of(x) for x in data_list])
    return {
        bin_id
        for bin_id, count in counts.items()
        if event_with_proba(count / partition.bin_size)
    }


def budget_sampling(epsilon: float, bin_size: int) -> float:
    """
    The budget that can be used for randomization after the amplification
    by sampling

    :param epsilon: Original privacy budget
    :param bin_size: Minimum size of each bin in the partition
    :return: Amplified budget
    """
    return np.log(1 + bin_size * (np.exp(epsilon) - 1))


def group_randomized_response(
    data_list: Iterable[int], epsilon: float, partition: PseudoRandomPartition
) -> set[int]:
    """Performs Group Randomized Response

    :param data_list: The list of values to be randomized
    :param epsilon: Privacy budget
    :param partition: Partition according to which data is to be grouped
    :return: Randomized data
    """
    sampled_list = group_sampling(data_list, partition)
    boosted_budget = budget_sampling(epsilon, partition.bin_size)
    return sparse_randomized_response(sampled_list, boosted_budget, partition.nb_bins)


def proba_grr_from_one(
    epsilon: float, partition: PseudoRandomPartition, nb_ones: float
) -> float:
    """
    Probability that the bin of a value is present in the output of
    group_randomized_response knowing that the value was present in
    data_list

    :param epsilon: Privacy budget
    :param partition: Partition according to which data is to be grouped
    :param nb_ones: number of elements present in data_list
    :return: Probability of presence in the randomized output
    """
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    p = ((s - 1) * (nb_ones - 1) / (n - 1) + 1) / s
    return (np.exp(boosted_budget) * p + 1 - p) / (1 + np.exp(boosted_budget))


def proba_grr_from_zero(
    epsilon: float, partition: PseudoRandomPartition, nb_ones: float
) -> float:
    """
    Probability that the bin of a value is present in the output of
    group_randomized_response knowing that the value was not present in
    data_list

    :param epsilon: Privacy budget
    :param partition: Partition according to which data is to be grouped
    :param nb_ones: number of elements present in data_list
    :return: Probability of presence in the randomized output
    """
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    p = (s - 1) / s * nb_ones / (n - 1)
    return (np.exp(boosted_budget) * p + 1 - p) / (1 + np.exp(boosted_budget))


def get_max_estimation_grr(epsilon: float, partition: PseudoRandomPartition) -> float:
    """
    Maximum estimation of the presence of a value in the output of
    group_randomized_response

    :param epsilon: Privacy budget
    :param partition: Partition according to which data is to be grouped
    :return: Maximum estimation
    """
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    return (n - 1) * s / (n - s) * np.exp(boosted_budget) / (np.exp(boosted_budget) - 1)


def get_max_alpha_grr(epsilon: float, partition: PseudoRandomPartition) -> float:
    """Maximum value of the multiplicative unbiasing parameter

    :param epsilon: Privacy budget
    :param partition: Partition according to which data is to be grouped
    :return: Maximum value of the multiplicative unbiasing parameter
    """
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    return (
        (n - 1)
        * s
        / (n - s)
        * (np.exp(boosted_budget) + 1)
        / (np.exp(boosted_budget) - 1)
    )


def get_min_proba_from_one_grr(partition: PseudoRandomPartition) -> float:
    """
    Minimum probability that the bin of a value is present in the output
    of group_randomized_response knowing that the value was present in
    data_list

    :param partition: Partition according to which data is to be grouped
    :return: Minimum probability of presence in the randomized output
    """
    return 1 / partition.bin_size


def asymmetric_randomized_response(
    data_list: list[int] | set[int], epsilon: float, mu: float, data_size: int
) -> set[int]:
    """Performs Asymmetric Randomized Response

    :param data_list: List of data to be randomized
    :param epsilon: Privacy budget
    :param mu: Sampling rate
    :param data_size: Size of the data space
    :return: Randomized data
    """
    rho = np.exp(-epsilon)
    proba_of_1_from_0 = mu * rho
    proba_of_1_from_1 = mu
    random_indices = np.random.choice(
        data_size, binom(data_size, proba_of_1_from_0).rvs(), replace=False
    )
    output_list_from_zeros = {x for x in random_indices if x not in data_list}
    output_list_from_ones = set(
        compress(data_list, np.random.rand(len(data_list)) < proba_of_1_from_1)
    )
    return output_list_from_ones | output_list_from_zeros


def proba_arr_from_one(mu: float) -> float:
    """
    Probability of the presence in the output of
    asymmetric_randomized_response knowing that the value was in data_list

    :param mu: Sampling rate
    :return: Probability of presence in the randomized output
    """
    return mu


@lru_cache
def proba_arr_from_zero(epsilon: float, mu: float) -> float:
    """
    Probability of the presence in the output of
    asymmetric_randomized_response knowing that the value was not in
    data_list

    :param epsilon: Privacy budget
    :param mu: Sampling rate
    :return: Probability of presence in the randomized output
    """
    return mu * np.exp(-epsilon)
