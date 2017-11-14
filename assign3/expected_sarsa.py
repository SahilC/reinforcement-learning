from collections import defaultdict
import random

class Expected_sarsa:
    def __init__(self, actions, epsilon=0.05, gamma=0.99):
        self.q = {}
        self.dalpha = defaultdict(lambda : defaultdict(int))

        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward 
        else:
            alpha = self.dalpha[state][action]
            if alpha == 0:
                alpha = 1/0.00001
            else:
                alpha = 1/alpha
            self.q[(state, action)] = oldv + alpha * (value - oldv)

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
        self.dalpha[state][action] += 1
        return action

    def learn(self, state1, action1, reward, state2):
        qvals = [self.getQ(state2, a) for a in self.actions]
        qmax = max(qvals)
        qsum = sum(qvals)
        qexpected = (1-self.epsilon)*qmax + (self.epsilon/len(self.actions))*(qsum)
        self.learnQ(state1, action1, reward, reward + self.gamma * qexpected)

        # (1-epsilon)*Q[s_new][best_action]+(epsilon/mdp.A)*sum(Q[s_new][act] for act in range(mdp.A))
