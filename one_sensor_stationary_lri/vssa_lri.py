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
        # self.p is the learned action probability vector.
        self.p = np.array(h.make_p(num_actions))
        # self.k = 1-lambda of the reward function for a LRI automata.
        self.k = 0.9
        self.n = 0
        self.best = 2 * num_actions  # Best time-cost.

    def next_action(self):
        '''Pick the next action of the learning automata based on the
           probability vector, self.p, of the LRI automata. At the first
           time instant all action probabilities are equally likely.'''
        # Pick a uniformly distributed random number to be tested against
        # the CDF of self.p.  Used to determine the action of the
        # automaton.
        randy = uniform(0, 1)  # Throwback to Archer.
        # On catastrophic failure pick the first action.
        index = 0  # Worst case select the first action.
        # print("The p is: " + str(self.p))  # Debug the change in self.p.
        cdf = h.cdf(self.p)
        # print("The cdf is: " + str(cdf))  # Debug the selected action.
        # index is the index of the CDF corresponding to the random
        # action. This value is the next action the automaton will choose.
        index = h.get_index(randy, cdf)
        return index

    def do_reward(self, action):
        '''Update the action probability, self.p, given the environment
           issued a reward.'''
        self.p[action] += self.k * (1 - self.p[action])
        # Need to update penalty probabilities.
        for i in range(len(self.p)):
            if(i != action):
                # print("i = " + str(i) + ". P = " + str(self.p))
                self.p[i] = (1 - self.k) * self.p[i]

    def do_penalty(self):
        '''LRI automata do nothing on penalty.'''
        pass
