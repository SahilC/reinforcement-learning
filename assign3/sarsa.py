import random

class Sarsa:
    def __init__(self, actions, epsilon=0.05, alpha=0.2, gamma=0.99):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward 
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def add_noise_to_walk(self, action, i, ap = 0.9):
        if random.random() < ap:
                action = self.actions[i]
        elif i%2 == 0:
            if random.random() > 0.5:
                action = self.actions[(i + 2) % len(self.actions)]
            else:
                action = self.actions[(i + 3) % len(self.actions)]
        else:
            if random.random() > 0.5:
                action = self.actions[(i + 1) % len(self.actions)]
            else:
                action = self.actions[(i + 2) % len(self.actions)]
        return action

    def chooseAction(self, state, ap = 0.9):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.add_noise_to_walk(self.actions[i], i, ap)
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
