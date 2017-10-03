from bandit import Bandit
import numpy as np
def ucb_action_selection(k, numsteps, c):
    Apossible = {}
    A = np.zeros((numsteps,))
    N = np.zeros((k,numsteps+1))
    R = np.zeros((numsteps,))
    Q = np.zeros((k,(numsteps+1)))
    ucb = np.zeros((k, numsteps))
    bandit = Bandit(k)

    for t in range(numsteps):
        untaken_actions = np.where(N[:,t] == 0)[0]
    
        if len(untaken_actions) > 0:
            Apossible[t] = untaken_actions
        else:
            ucb[:,t] = Q[:,t] + c*np.sqrt(np.log(t)/N[:,t])
            Apossible[t] = np.argwhere(ucb[:,t] == np.amax(ucb[:,t])).flatten()

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

    return {'Apossible': Apossible,'ucb' : ucb, 'A': A, 'N' : N, 'R' : R, 'Q' : Q}
