'''Underwater Detection Experiment.'''
from lrp import Linear_Reward_Penalty as LRP
from environment import Environment
from pinger import Pinger
import numpy as np

# Define the number of discrete depths between the surface and seabed.
num_actions = 6

# Define the environment with the number of discrete depths for the detectable
# object.
env = Environment(num_actions)
# Define the LRI automata with the same number of actions. This number does
# not correspond to the number of receivers on the array. It is merely the
# representation of the array's ability to detect the object at that depth.
lrp = LRP(num_actions)  # The learning automata.
# The most probable depth that the object exists at, as calculated by the
# learner.
bestdepth = np.zeros(num_actions)
# The penalty probabilities for the learner.
E = [0.1, 0.2, 0.4, 0.2, 0.01, 0.09]
det_obj = Pinger(E)  # Create the detectable object.

# Run 5 individual experiments experiments.
for k in range(1):
    # Generate an ensemble of 100000 experiments
    for j in range(100000):
        # reset the action probabilities.
        lrp.reset_actions()
        # Run a single experiment. Terminate if it reaches 10000 iterations.
        while(True):
            # Define m as the next action predicting the depth of the object.
            m = lrp.next_action()
            # Defin req as the next detectable object depth.
            req = det_obj.request()
            # reward if m = req.
            resp = env.response(m, req)
            if(not resp):
                lrp.do_reward(m)
            else:
                lrp.do_penalty(m)
            if(max(lrp.p) > 0.98):
                # The best depth counting from 0.
                # Break at 98% convergence to a single depth.
                bestdepth[np.argmax(lrp.p)] += 1
                break
    print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
