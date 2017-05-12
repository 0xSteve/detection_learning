'''Elevator test.'''
from discretized_lri import DLRI as DLRI
from environment import Environment
from pinger import Pinger
import numpy as np
import helpers as h
import math

num_actions = 5

env = Environment(num_actions)
dlri = DLRI(num_actions)
bestdepth = np.zeros(num_actions)
E = [0.1, 0.2, 0.3, 0.3, 0.1]
r = Pinger(E)
for k in range(5):
    for j in range(100):
        # Caught me again...
        dlri.p = np.array(h.make_dp(num_actions))
        m = math.floor(num_actions / 2)
        while(True):
            req = r.request()
            resp = env.response(m, req)
            if(not resp):
                dlri.do_reward(m)
            else:
                dlri.do_penalty()
            m = dlri.next_action()
            if(max(dlri.p) == (num_actions * num_actions)):
                # The best depth counting from 0 (seasurface).
                bestdepth[np.argmax(dlri.p)] += 1
                break
    # print("The best depth tally is : " + str(bestdepth))
    # print("Converge on depth: " + str(np.argmax(bestdepth)))
    print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
