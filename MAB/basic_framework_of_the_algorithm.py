import numpy as np

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # number of attempts per lever
        self.regret = 0
        self.actions = []                      # record the actions taken at each step
        self.regrets = []                      # record the accumulated regrets of each step

    def update_regret(self, k):
        """
        Calculate and save the cumulative regret, where k is the lever number selected for this action.
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """
        Return to the current action and select which lever to pull.
        Determined by each specific strategy
        """
        raise NotImplementedError

    def run(self, num_steps):
        """
        Run a certain number of times, where num_steps is the total number of runs.
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


