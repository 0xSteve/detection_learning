'''The Environment module.'''
import helpers as h
import numpy as np


class Environment(object):
    '''The Environment in which learning occurs.'''

    def __init__(self, num_actions):
        self.c = np.array(h.make_p(num_actions))
        self.best = 10 * num_actions  # Always worse than the worst time.

    def response(self, m, request):
        is_reward = self.reward_function(m, request)
        if(not is_reward):
            self.c += 1
            self.c[m] -= 1
            self.c = self.c / sum(self.c)
        return is_reward

    def reward_function(self, m, req):
        '''The reward function is the heart and soul of this problem.'''
        # The reward function needs to be good, otherwise learning does
        # not happen.
        if(m == req):
            return 0
        return 1
