'''Linear Reward-Inaction Variable Structure Stochastic Automaton.'''
from random import uniform
import helpers as h
import numpy as np

# In this case it is not possible to feed the penalty probability vector,
# c, as an input. For this, one will have to be a little bit more
# creative.


class Linear_R(object):
    '''The Linear Reward-Inaction model (for now).'''

    def __init__(self, num_actions):
        '''Create a new Linear Reward-Inaction object.'''
        self.p = np.array(h.make_p(num_actions))
        self.k = 0.9
        self.n = 0
        self.best = 2 * num_actions  # Best time-cost.

    def next_action(self):
        randy = uniform(0, 1)  # Throwback to Archer.
        index = 0  # Worst case select the first action.
        # print("The p is: " + str(self.p))
        cdf = h.cdf(self.p)
        # print("The cdf is: " + str(cdf))
        index = h.get_index(randy, cdf)
        return index

    def do_reward(self, action):
        self.p[action] += self.k * (1 - self.p[action])
        # Need to update penalty probabilities.
        for i in range(len(self.p)):
            if(i != action):
                # print("i = " + str(i) + ". P = " + str(self.p))
                self.p[i] = (1 - self.k) * self.p[i]

    def do_penalty(self):
        pass
