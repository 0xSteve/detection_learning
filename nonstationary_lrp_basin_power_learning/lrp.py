'''Linear Reward-Inaction Variable Structure Stochastic Automaton.'''
from random import uniform
import helpers as h
import numpy as np

# In this case it is not possible to feed the penalty probability vector,
# c, as an input. For this, one will have to be a little bit more
# creative.


class Linear_Reward_Penalty(object):
    '''The Linear Reward-Inaction model (for now).'''

    def __init__(self, num_actions):
        '''Create a new Linear Reward-Inaction object.'''
        # self.p is the learned action probability vector.
        self.p = np.array(h.make_p(num_actions))
        # self.a
        self.a = 0.9
        # sekf.b
        self.b = 0.2
        self.r = num_actions

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

    def reset_actions(self):
        self.p = np.array(h.make_p(len(self.p)))

    def do_reward(self, action):
        '''Update the action probability, self.p, given the environment
           issued a reward.'''
        self.p[action] += self.a * (1 - self.p[action])
        # Need to update penalty probabilities.
        for i in range(len(self.p)):
            if(i != action):
                # print("i = " + str(i) + ". P = " + str(self.p))
                self.p[i] = (1 - self.a) * self.p[i]

    def do_penalty(self, action):
        '''Update the action probability, self.p, give the environment issued
           a penalty.'''
        self.p[action] = (1 - self.b) * self.p[action]
        # Need to update penalty probabilities.
        for i in range(len(self.p)):
            if(i != action):
                # print("i = " + str(i) + ". P = " + str(self.p))
                self.p[i] = self.b / (self.r - 1) + (1 - self.b) * self.p[i]
