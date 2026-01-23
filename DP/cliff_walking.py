class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP()
        # The transition matrix P[state][action] = [(p, next_state, reward, done)] contains the next state and the reward.

    def createP(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]  # initialization
        # Four actions: change[0]: up, change[1]: down, change[2]: left, change[3]: right
        # The origin of the coordinate system (0,0) is defined at the top left corner.
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # If the location is on a cliff or in a target state, no further interaction is possible, so any action will result in a reward of 0.
                    if i == self.nrow -1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # Other location
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))  # The key element is j + change[a][0]
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # The next location is either on the cliff or at the finish line.
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
