import numpy as np
def exp3(k, numsteps):
    w = np.ones((k,))

    # Initialize bandit
    bandit = Bandit(k)
    ita = np.sqrt(2*np.log(k)/(k*numsteps))
    total_loss = np.zeros((k,))
    arm_selected = np.zeros((numsteps,))
    for t in range(numsteps):
        loss = bandit.loss(0.1, t, numsteps)
        total_loss += loss
        a = np.argmin(total_loss)
        arm_selected[t] = a

        w = w*np.exp(-ita*total_loss)

    return {'w': w, 'A': arm_selected, 'R' : total_loss}


