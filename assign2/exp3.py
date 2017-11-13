from bandit import Bandit
import numpy as np
def exp3(k, numsteps):
    w = np.ones((k,))

    bandit = Bandit(k)
    ita = np.sqrt(2*np.log(k)/(k*numsteps))
    total_loss = np.zeros((k,))
    arm_selected = np.zeros((numsteps,))
    for t in range(numsteps):
        loss = bandit.loss(0.1, t, numsteps)
        total_loss += loss
        a = np.argmax(w)
        arm_selected[t] = a

        w[a] = w[a]*np.exp(-ita*total_loss[a])

    return {'bandit':bandit,'w': w, 'A': arm_selected, 'R' : total_loss}


