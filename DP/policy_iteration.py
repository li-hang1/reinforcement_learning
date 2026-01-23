import copy


class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # Initial value is 0
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.ncol * self.env.nrow)]  # Initialize as a uniform random policy
        self.theta = theta  # Policy evaluation convergence threshold
        self.gamma = gamma  # discount factor

    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow  # initialization
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res     # Transfer result
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print(f"Strategy evaluation is completed after {cnt} rounds.")

    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("Strategy upgrade completed")
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State value:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print(f"{agent.v[i * agent.env.ncol + j]:.3f} ", end=" ")
        print()

    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print("****", end=" ")
            elif (i * agent.env.ncol + j) in end:
                print("EEEE", end=" ")
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "o"
                print(pi_str, end=" ")
        print()


if __name__ == "__main__":
    from cliff_walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])


