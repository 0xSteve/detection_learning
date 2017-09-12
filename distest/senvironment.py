'''This module contains the S-model Environment.'''
import numpy as np
# from random import uniform as u


class SEnvironment(object):
    '''The S-model Learning Environment.'''

    def __init__(self, p_vector, precision=1):
        '''Create a probability vector from the probability of
           success vector.'''
        self.p = np.array(p_vector)
        self.precision = precision
        # Only the environment knows the best xmission a priori.
        # best xmission is used to evaluate a posteriori learning.
        self.best_xmission = max(self.p)

    def response(self, depth_index):
        '''Respond to the mobile-agent the value of the timeout
           probability.'''
        # return self.p[depth_index] > u(0, 1)
        return self.p[(depth_index - 1) * self.precision]
