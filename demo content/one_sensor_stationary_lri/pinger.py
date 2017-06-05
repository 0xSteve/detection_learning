'''Module containing the requester class.'''
import numpy as np
from random import uniform
import helpers as h


class Pinger(object):
    '''An object pinging an acoustic signature from a depth. It pings with
        probability vector E.'''

    def __init__(self, E):
        self.E = np.array(E)

    def request(self):
        '''The method that determines the actual ping that will be
           made by the pinger object.'''
        randy = uniform(0, 1)
        ping_cdf = h.cdf(self.E)
        ping = h.get_index(randy, ping_cdf)
        return ping
