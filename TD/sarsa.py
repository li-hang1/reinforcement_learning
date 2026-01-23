import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from cliff_walking import CliffWalkingEnv
from print_agent import print_agent


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # initialize the Q(s,a) table
        self.n_action = n_action                          # number of actions
        self.alpha = alpha                                # learning rate
        self.epsilon = epsilon                            # parameters in the epsilon-greedy strategy
        self.gamma = gamma                                # discount factor

    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # Printing strategy
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


ncol = 12
nrow = 4
env = CliffWalkingEnv(nrow, ncol)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc = f"Iteration {i}") as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"episode": num_episodes / 10 * i + i_episode + 1, "return0": f"{np.mean(return_list[-10:]):.3f}"})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print('The strategy that the Sarsa algorithm ultimately converges to is: ')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

