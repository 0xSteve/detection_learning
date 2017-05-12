'''DiscretizedLinear Reward-Inaction Variable Structure Stochastic
   Automaton.'''
from random import uniform
import helpers as h
import numpy as np


class DLRI(object):

    def __init__(self, num_actions):
        self.p = np.array(h.make_dp(num_actions * 10))
        self.best = 2 * num_actions  # Best time-cost.

    def next_action(self):
        randy = uniform(0, 1)  # Throwback to Archer.
        index = 0  # Worst case select the first action.
        # print("The p is: " + str(self.p))
        probs = self.p / sum(self.p)
        cdf = h.cdf(probs)
        # print("The cdf is: " + str(cdf))
        index = h.get_index(randy, cdf)
        return index

    def do_reward(self, action):
        self.p[action] += np.count_nonzero(self.p)
        self.p = h.subtract_nonzero(self.p, 1)

    def do_penalty(self):
        pass
