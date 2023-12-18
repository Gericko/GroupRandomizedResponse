import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import quad


class PolynomialTailRV(rv_continuous):
    """Class implementing a random variable with a polynomial tail distribution"""

    def __init__(self, gamma: int) -> None:
        """Constructor for PolynomialTailRV

        :param gamma: parameter governing the decreasing order of the distribution
        """
        super(PolynomialTailRV, self).__init__()
        self.gamma = gamma
        self.leading_coefficient = quad(
            lambda y: 1 / (1 + abs(y) ** gamma), -np.inf, np.inf
        )[0]

    def _pdf(self, x: float, *args) -> float:
        return 1 / (self.leading_coefficient * (1 + abs(x) ** self.gamma))


class SmoothAccessMechanism:
    """
    Generic class implementing the smooth access mechanism to privately
    publish statistics using their smooth sensitivity

    To work, the methods "function" and "smooth_sensitivity" should be implemented
    """

    def __init__(self, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = epsilon / (2 * (gamma - 1))
        self.beta = epsilon / (2 * (gamma - 1))
        self.rv = PolynomialTailRV(gamma=self.gamma)
        self.sd = self.rv.std()

    def function(self, x):
        """Statistics function to be published"""
        raise NotImplementedError

    def smooth_sensitivity(self, x):
        """Smooth sensitivity of the function on a given data point"""
        raise NotImplementedError

    def publish(self, x):
        """Publishes the obfuscated data as a tuple (count, bias, sensitivity)"""
        return (
            self.function(x),
            0,
            self.smooth_sensitivity(x) / self.alpha * self.rv.rvs(),
        )

    def std(self, x):
        """Standard deviation of the publication on a given data point"""
        return self.smooth_sensitivity(x) / self.alpha * self.sd
