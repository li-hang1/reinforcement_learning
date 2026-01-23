import numpy as np


def compute(P, rewards, gamma, states_num):
    """
    The analytical solution is calculated using the matrix form of the Bellman equations,
    where states_num is the number of states in the MRP.
    """
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


S = ["s1", "s2", "s3", "s4", "s5"]  # state set
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # action set
# state transition function
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# reward function
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # discount factor
MDP = (S, A, P, R, gamma)

# Strategy 1, Random Strategy
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# Strategy 2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

def join(str1, str2):
    return str1 + '-' + str2

# Marginalizing the action selection of the strategy results in an MRP with no actions.
gamma = 0.5
# Transformed MRP state transition matrix
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print("The value of each state in MDP is as follows:\n", V)



def sample(MDP, Pi, timestep_max, number):
    """
    Sampling function
    Pi: strategy Pi
    timestep_max: maximum time step limit
    number: total number of sampled sequences
    return:
        List[tuple(s, a, r, s_next)], Sequence obtained by random sampling
    """
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # Randomly select a state s other than s5 as the starting point.
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next =s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

# Sample 5 times, each sequence should not exceed 20 steps.
episodes = sample(MDP, Pi_1, 20, 5)
print('First sequence:\n', episodes[0])
print('Second sequence:\n', episodes[1])
print('Fifth sequence:\n', episodes[4])

# Monte-Carlo
def MC(episodes, V, N, gamma):
    """The reward of a state in the sequence is calculated every time it appears."""
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1):
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

timestep_max = 20
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("The state value of the MDP is calculated using the Monte Carlo method:\n", V)


def occupancy(episodes, s, a, timestep_max, gamma):
    """
    The occupancy measure refers to the percentage of the discounted probability that a state-action pair (s,a) will
    be accessed in an infinite time trajectory, given a policy π and a discount factor γ.
    The frequency of occurrence of state-action pairs (s, a) is calculated to estimate the occupancy measure of the policy.
    """
    rho = 0  # ρ
    total_times = np.zeros(timestep_max)  # Record how many times each time step t is experienced.
    occur_times = np.zeros(timestep_max)  # Record the number of times (s_t, a_t) = (s, a).
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s ==s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho

gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(f"Pi_1_rho: {rho_1}, Pi_2_rho: {rho_2}")





