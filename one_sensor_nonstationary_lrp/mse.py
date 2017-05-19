'''The Markovian Swithcing Environment module.'''
import numpy as np


class MSE(object):
    '''Change the learning environment in accordance with a Markovian
       Switching Environment model.'''

    def __init__(self, environments, time_between=0):
        '''Define the number of environments and their order by passing
           the environments value. Optionally, choose the time between
           changing environments with the time_between variable. If the
           default value is used, then environment switching is assumed to be
           manual.'''
        self.envs = np.array(environments)
        self.time_between = 0
        '''Create a new instance of the MSE class.'''
        # Could possibly pass number of environments as a variable.
        # Could possibly pass amount of time between intervals as a variable.
        pass

    def set_environments(self, environments):
        '''Set the environments that are in the MSE.'''
        self.envs = np.array(environments)

    def get_environments(self):
        '''Return the list of environments in the MSE.'''
        return self.envs

    def next_env(self):
        '''Switch to the next environment in the MSE.'''
        pass
