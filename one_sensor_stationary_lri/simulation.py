'''Elevator test.'''
from vssa_lri import Linear_R as LRI
from environment import Environment
from pinger import Pinger
import numpy as np
import helpers as h

num_actions = 6

env = Environment(num_actions)
lri = LRI(num_actions)
bestdepth = np.zeros(num_actions)
E = [0.1, 0.2, 0.4, 0.2, 0.01, 0.09]
det_obj = Pinger(E)
for k in range(5):
    for j in range(100000):
        # Caught me again...
        lri.p = np.array(h.make_p(num_actions))
        for i in range(10000):
            m = lri.next_action()
            req = det_obj.request()
            resp = env.response(m, req)
            if(not resp):
                lri.do_reward(m)
            else:
                lri.do_penalty()
            if(max(lri.p) > 0.98):
                # The best floor counting from 0.
                bestdepth[np.argmax(lri.p)] += 1
                break
    # print("The best depth tally is : " + str(bestdepth))
    print("Converge on depth: " + str(np.argmax(bestdepth) + 1))
    print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
