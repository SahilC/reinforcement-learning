from bandit import Bandit
import numpy as np
def ucb_action_selection(k, numsteps, c):
    # k: number of bandit arms
    # numsteps: number of steps (repeated action selections)
    # c: parameter controlling the degree of exploration

    # Apossible[t]: list of possible actions at step t
    Apossible = {}
    
    # A[t]: action selected at step t
    A = np.zeros((numsteps,))
    
    # N[a,t]: the number of times action a was selected 
    #         in steps 0 through t-1
    N = np.zeros((k,numsteps+1))
    
    # R[t]: reward at step t
    R = np.zeros((numsteps,))
    
    # Q[a,t]: estimated value of action a at step t
    Q = np.zeros((k,(numsteps+1)))
    
    ucb = np.zeros((k, numsteps))

    # Initialize bandit
    bandit = Bandit(k)

    for t in range(numsteps):
        untaken_actions = np.where(N[:,t] == 0)[0]
    
        if len(untaken_actions) > 0:
            # If there are untaken actions, add them to possible actions
            Apossible[t] = untaken_actions
        else:
            # Otherwise, calculate UCB score for each action and select maximum
            ucb[:,t] = Q[:,t] + c*np.sqrt(np.log(t)/N[:,t])
            Apossible[t] = np.argwhere(ucb[:,t] == np.amax(ucb[:,t])).flatten()

        # Select action randomly from possible actions
        a = Apossible[t][np.random.randint(len(Apossible[t]))]

        # Record action taken
        A[t] = a

        # Perform action (= sample reward)
        R[t] = bandit.action(a)

        # Update action counts
        N[:,t+1] = N[:,t]
        N[a,t+1] += 1

        # Update action value estimates, incrementally
        if N[a,t] > 0:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = Q[a,t] + (R[t] - Q[a,t]) / N[a,t]
        else:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = R[t]

    return {'bandit' : bandit,
            'numsteps' : numsteps,
            'epsilon' : epsilon,
            'Apossible': Apossible,
            'ucb' : ucb,
            'A': A, 'N' : N, 'R' : R, 'Q' : Q}
