'''Underwater Detection Experiment.'''
from lrp import Linear_Reward_Penalty as LRP
from mse import MSE
from environment import Environment
from pinger import Pinger
import numpy as np
import csv

# Define the number of discrete depths between the surface and seabed.
num_actions = 35
n = 10000
interval = 1
time_between = (n / interval) - 1

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
# Define the Markovian Switching Environment that will feed probabilities to
# the Pinger object.
Es = []
with open('environments.csv', 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
            new = [float(x) for x in row if x != '']
            Es.append(new)
mse = MSE(Es)

det_obj = Pinger(mse.env_now())  # Create the detectable object.
# Run 5 individual experiments experiments.
for k in range(len(Es)):
    # Generate an ensemble of n experiments
    for j in range(n):
        # reset the action probabilities.
        lrp.reset_actions()
        # Run a single experiment. Terminate if it reaches 10000 iterations.
        while(True):
            # Define m as the next action predicting the depth of the object.
            m = lrp.next_action()
            # Define req as the next detectable object depth.
            req = det_obj.request()
            # reward if m = req.
            resp = env.response(m, req)
            if(not resp):
                lrp.do_reward(m)
            else:
                lrp.do_penalty(m)
            if(max(lrp.p) > 0.999):
                # The best depth counting from 0.
                # Break at 98% convergence to a single depth.
                bestdepth[np.argmax(lrp.p)] += 1
                break
        # if(j == time_between):
        print("The desired vector is now: " + str(mse.env_now()))
        print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
        mse.next_env()
        det_obj.set_env(mse.env_now())
