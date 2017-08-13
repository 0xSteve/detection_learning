'''This module contains the S-model Environment.'''
import numpy as np
from random import uniform as u


class SEnvironment(object):
    '''The S-model Learning Environment.'''

    def __init__(self, p_vector):
        '''Create a penalty probability vector from the probability of
           success vector.'''
        c = np.zeros(len(p_vector))
        c += 1
        self.c = c - np.array(p_vector)
        # Only the environment knows the best xmission a priori.
        # best xmission is used to evaluate a posteriori learning.
        self.best_xmission = min(self.c)

    def response(self, depth_index):
        '''Respond to the mobile-agent the value of the timeout
           probability.'''
        return self.c[depth_index] <= u(0, 1)
