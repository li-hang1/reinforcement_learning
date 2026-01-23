import numpy as np


np.random.seed(0)

# Define the state transition probability matrix P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5  # discount factor

def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print(f"The return calculated based on this sequence is: {G}")


def compute(P, rewards, gamma, states_num):
    """
    The analytical solution is calculated using the matrix form of the Bellman equations,
    where states_num is the number of states in the MRP.
    """
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

V = compute(P, rewards, gamma, 6)
print(f"The value of each state in MRP is \n", V)
