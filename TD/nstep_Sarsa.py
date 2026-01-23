import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cliff_walking import CliffWalkingEnv
from print_agent import print_agent

class nstep_Sarsa:
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n             # Using the n-step Sarsa algorithm
        self.state_list = []   # Save previous state
        self.action_list = []  # Save previous action
        self.reward_list = []  # Save previous reward

    def take_action(self, state):
        if np.random.random() < self.epsilon:
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

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:                # If the saved data can be updated in n steps
            G = self.Q_table[s1, a1]                      # Obtain Q(s_{t+n}, a_{t+n}), state_list[0] is s_{t}, action_list[0] is a_{t}
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # Continuously calculate the reward for each step forward.
                # If the terminal state is reached, even if the last few steps are less than n steps, the remaining steps can be used to update the current steps, and n steps are no longer required.
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # Remove the (state, action) that needs updating from the list; it won't need to be updated next time.
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n-step Sarsa main update steps
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # If the process reaches a termination state and is about to begin the next sequence, then the entire list is cleared.
            self.state_list = []
            self.action_list = []
            self.reward_list = []

ncol = 12
nrow = 4
env = CliffWalkingEnv(nrow, ncol)
np.random.seed(0)
n_step = 5
alpha = 0.1
epsilon = 0.1
gamma = 0.9
agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500   # The number of sequences that the agent runs in the environment.

return_list = []     # Record the reward for each sequence
for i in range(10):  # Display 10 progress bars
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # Sequence number of each progress bar
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward  # The return calculation here does not include discount factor decay.
                agent.update(state, action, reward, next_state, next_action,
                             done)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': num_episodes / 10 * i + i_episode + 1, 'return': f"{np.mean(return_list[-10:])}"})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print("The strategy obtained by the 5-step Sarsa algorithm upon final convergence is: ")
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])