import numpy as np

from plot_results import plot_results
from basic_framework_of_the_algorithm import Solver
from multi_armed_bandit import BernoulliBandit


class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
coef = 1
bandit_10_arm = BernoulliBandit(K=10)
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print("The cumulative regret of the upper confidence bound algorithm is:", UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
