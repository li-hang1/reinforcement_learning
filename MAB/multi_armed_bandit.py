import numpy as np

class BernoulliBandit:
    def __init__(self, K):
        """K represents the number of levers."""
        self.probs = np.random.uniform(size=K)      # The probability of winning for each lever.
        self.best_idx = np.argmax(self.probs)       # The lever with the highest probability of winning.
        self.best_prob = self.probs[self.best_idx]  # Maximum probability of winning.
        self.K = K

    def step(self, k):
        """
        When a player selects lever number k, the machine returns either 1 (winning) or 0 (not winning)
        based on the probability of winning by pulling lever number k.
        """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print(f"A {K}-arm Bernoulli slot machine was randomly generated.")
    print(f"The lever with the highest probability of winning is number {bandit_10_arm.best_idx}, with a probability of {bandit_10_arm.best_prob}.")


