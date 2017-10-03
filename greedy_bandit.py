from bandit import Bandit
import numpy as np

def greedy_action_selection(k, numsteps, epsilon = 0):
    
    Apossible = {}
    A = np.zeros((numsteps,))
    N = np.zeros((k,numsteps+1))
    R = np.zeros((numsteps,))
    Q = np.zeros((k,(numsteps+1)))

    bandit = Bandit(k)
    for t in range(numsteps):
        if t % 100 == 0:
            print 'Progress!!',t
            
        if np.random.rand() < epsilon:
            Apossible[t] = np.arange(k)
        else:
            Apossible[t] = np.argwhere(Q[:,t] == np.amax(Q[:,t])).flatten()

        a = Apossible[t][np.random.randint(len(Apossible[t]))]

        A[t] = a

        R[t] = bandit.action(a)

        N[:,t+1] = N[:,t]
        N[a,t+1] += 1

        if N[a,t] > 0:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = Q[a,t] + (R[t] - Q[a,t]) / N[a,t]
        else:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = R[t]

    return {'bandit': bandit, 'Apossible': Apossible, 'A': A, 'N' : N, 'R' : R, 'Q' : Q}
