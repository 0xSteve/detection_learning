from lrp import Linear_Reward_Penalty as LRP
from environment import Environment
from pinger import Pinger
import tune_lrp as tune
import numpy as np

test_lrp = LRP(5)
penaly_probs = [0.3, 0.1, 0.1, 0.1, 0.4]
penalizer = Pinger(np.array(penaly_probs))
env = Environment(5)
a = tune.find_optimal_a(test_lrp, env, penalizer)
print("The value for a after tuning is " + str(test_lrp.a))
b = tune.find_optimal_b(test_lrp, env, penalizer)
print("The value for b after tuning is " + str(test_lrp.b))
test_lrp.a = a
test_lrp.b = b
n = 10000
bestdepth = np.zeros(5)
for j in range(n):
        # reset the action probabilities.
        test_lrp.reset_actions()
        # Run a single experiment. Terminate if it reaches 10000 iterations.
        while(True):
            # Define m as the next action predicting the depth of the object.
            m = test_lrp.next_action()
            # Define req as the next detectable object depth.
            req = penalizer.request()
            # reward if m = req.
            resp = env.response(m, req)
            if(not resp):
                test_lrp.do_reward(m)
            else:
                test_lrp.do_penalty(m)
            if(max(test_lrp.p) > 0.98):
                # The best depth counting from 0.
                # Break at 98% convergence to a single depth.
                bestdepth[np.argmax(test_lrp.p)] += 1
                break
print("The desired probability vector is: " + str(penaly_probs))
print("The actual probability vector is: " + str(bestdepth / sum(bestdepth)))
