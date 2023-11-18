import numpy as np
from scipy.stats import laplace, binom
from collections import Counter
from itertools import compress
from functools import lru_cache


def randomized_response(x, epsilon):
    p = 1 / (1 + np.exp(epsilon))
    is_flip = np.random.rand(x.size) < p
    return np.where(is_flip, 1 - x, x)


def add_laplacian_noise(data, budget, sensi=1):
    scale = sensi / budget
    noisy_data = data + laplace(0, scale).rvs(size=len(data))
    return noisy_data


def clip_list(data, max_size):
    if len(data) <= max_size:
        return data
    random_indices = np.random.choice(len(data), max_size, replace=False)
    return [x for i, x in enumerate(data) if i in random_indices]


def make_clip_dict(data, max_size):
    if len(data) <= max_size:
        return {x: False for x in data}
    random_indices = np.random.choice(len(data), max_size, replace=False)
    return {x: i not in random_indices for i, x in enumerate(data)}


def sparse_randomized_response(data_list, epsilon, data_size):
    p = 1 / (1 + np.exp(epsilon))
    random_indices = np.random.choice(
        data_size, binom(data_size, p).rvs(), replace=False
    )
    output_list_from_zeros = {x for x in random_indices if x not in data_list}
    output_list_from_ones = set(compress(data_list, np.random.rand(len(data_list)) > p))
    return output_list_from_ones | output_list_from_zeros


def event_with_proba(probability):
    return np.random.rand() < probability


def group_sampling(data_list, partition):
    counts = Counter([partition.bin_of(x) for x in data_list])
    return {
        bin_id
        for bin_id, count in counts.items()
        if event_with_proba(count / partition.bin_size)
    }


def budget_sampling(epsilon, bin_size):
    return np.log(1 + bin_size * (np.exp(epsilon) - 1))


def group_randomized_response(data_list, epsilon, partition):
    sampled_list = group_sampling(data_list, partition)
    boosted_budget = budget_sampling(epsilon, partition.bin_size)
    return sparse_randomized_response(sampled_list, boosted_budget, partition.nb_bins)


def proba_grr_from_one(epsilon, partition, nb_ones):
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    p = ((s - 1) * (nb_ones - 1) / (n - 1) + 1) / s
    return (np.exp(boosted_budget) * p + 1 - p) / (1 + np.exp(boosted_budget))


def proba_grr_from_zero(epsilon, partition, nb_ones):
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    p = (s - 1) / s * nb_ones / (n - 1)
    return (np.exp(boosted_budget) * p + 1 - p) / (1 + np.exp(boosted_budget))


def get_grr_alpha(epsilon, partition, nb_ones):
    p1 = proba_grr_from_one(epsilon, partition, nb_ones)
    p0 = proba_grr_from_zero(epsilon, partition, nb_ones)
    return 1 / (p1 - p0)


def get_grr_beta(epsilon, partition, nb_ones):
    p1 = proba_grr_from_one(epsilon, partition, nb_ones)
    p0 = proba_grr_from_zero(epsilon, partition, nb_ones)
    return p0 / (p1 - p0)


def get_max_estimation_grr(epsilon, partition):
    n = partition.nb_bins * partition.bin_size
    s = partition.bin_size
    boosted_budget = budget_sampling(epsilon, s)
    return (n - 1) * s / (n - s) * np.exp(boosted_budget) / (np.exp(boosted_budget) - 1)


def get_max_alpha_grr(epsilon, partition):
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


def get_min_proba_from_one_grr(partition):
    return 1 / partition.bin_size


def unbiase_grr(x, epsilon, partition, nb_ones):
    alpha = get_grr_alpha(epsilon, partition, nb_ones)
    beta = get_grr_beta(epsilon, partition, nb_ones)
    return alpha * x - beta


def asymmetric_randomized_response(data_list, epsilon, mu, data_size):
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


def proba_arr_from_one(mu):
    return mu


@lru_cache
def proba_arr_from_zero(epsilon, mu):
    return mu * np.exp(-epsilon)


def get_arr_alpha(epsilon, mu):
    return 1 / mu / (1 - np.exp(-epsilon))


def get_arr_beta(epsilon):
    rho = np.exp(-epsilon)
    return rho / (1 - rho)


def unbiased_arr_count(count, out_of, epsilon, mu):
    rho = np.exp(-epsilon)
    return count / mu / (1 - rho) - out_of * rho / (1 - rho)
