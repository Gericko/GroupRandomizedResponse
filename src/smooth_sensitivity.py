import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import quad


class PolynomialTailRV(rv_continuous):
    def __init__(self, gamma):
        super(PolynomialTailRV, self).__init__()
        self.gamma = gamma
        self.leading_coefficient = quad(
            lambda y: 1 / (1 + abs(y) ** gamma), -np.inf, np.inf
        )[0]

    def _pdf(self, x, *args):
        return 1 / (self.leading_coefficient * (1 + abs(x) ** self.gamma))


class SmoothAccessMechanism:
    def __init__(self, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = epsilon / (4 * gamma)
        self.beta = epsilon / gamma
        self.rv = PolynomialTailRV(gamma=self.gamma)
        self.sd = self.rv.std()

    def function(self, x):
        raise NotImplementedError

    def smooth_sensitivity(self, x):
        raise NotImplementedError

    def publish(self, x):
        return (
            self.function(x),
            self.smooth_sensitivity(x) / self.alpha * self.rv.rvs(),
        )

    def std(self, x):
        return self.smooth_sensitivity(x) / self.alpha * self.sd
