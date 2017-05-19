'''Module containing the requester class.'''

from random import uniform
import helpers as h


class Pinger(object):
    '''An object pinging an acoustic signature from a depth. It pings with
        probability vector E. The Pinger object expects a numpy array.'''

    def __init__(self, E):
        self.E = E

    def request(self):
        '''The method that determines the actual ping that will be
           made by the pinger object.'''
        randy = uniform(0, 1)
        ping_cdf = h.cdf(self.E)
        ping = h.get_index(randy, ping_cdf)
        return ping

    def set_env(self, E):
        '''Change the environmental probabilities (E). E is expected to be a
           numpy array.'''
        self.E = E
