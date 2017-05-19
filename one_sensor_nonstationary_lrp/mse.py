'''The Markovian Swithcing Environment module.'''


class MSE(object):
    '''Change the learning environment in accordance with a Markovian
       Switching Environment model.'''

    def __init__(self):
        '''Create a new instance of the MSE class.'''
        # Could possibly pass number of environments as a variable.
        # Could possibly pass amount of time between intervals as a variable.
        pass

    def set_environments(self):
        '''Set the environments that are in the MSE.'''
        pass

    def get_environments(self):
        '''Return the list of environments in the MSE.'''
        pass

    def next_env(self):
        '''Switch to the next environment in the MSE.'''
        pass
