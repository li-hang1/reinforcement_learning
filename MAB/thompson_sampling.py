import numpy as np

from plot_results import plot_results
from basic_framework_of_the_algorithm import Solver
from multi_armed_bandit import BernoulliBandit


class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self.a = np.ones(self.bandit.K)  # the number of times each lever receives a bonus of 1
        self.b = np.ones(self.bandit.K)  # the number of times each lever receives a bonus of 0

    def run_one_step(self):
        samples = np.random.beta(self.a, self.b)  # A set of reward samples were sampled according to the Beta distribution.
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self.a[k] += r
        self.b[k] += 1 - r
        return k


np.random.seed(1)
bandit_10_arm = BernoulliBandit(K=10)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print("The cumulative regret of Thompson's sampling algorithm is:", thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])
