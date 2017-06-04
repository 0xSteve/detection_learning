'''Underwater Detection Experiment.'''
from lrp import Linear_Reward_Penalty as LRP
from mse import MSE
from environment import Environment
from pinger import Pinger
import tune_lrp as tune
import numpy as np
import csv

# Define the number of discrete depths between the surface and seabed.
num_actions = 5
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
Es = [[0.48796, 0.024438, 0.067891, 0.41971, 0.00],
      [0.50147, 0.021431, 0.071479, 0.40562, 0.00],
      [0.50582, 0.018288, 0.083153, 0.39274, 0.00],
      [0.48455, 0.015527, 0.18197, 0.31795, 0.00],
      [0.58845, 0.01675, 0.11313, 0.28167, 0.00]]
mse = MSE(Es)
det_obj = Pinger(mse.env_now())  # Create the detectable object.
# Run 5 individual experiments experiments.
for k in range(len(mse.envs)):
    # Generate an ensemble of n experiments
    det_obj.set_env(mse.env_now())
    print("The desired vector is now: " + str(mse.env_now()))
    # lrp.a = tune.find_optimal_a(lrp, env, det_obj)
    # print("Optimized value for a is: " + str(lrp.a))
    lrp.a = 0.99999999999999
    lrp.b = 0.2
    for j in range(n):
        # reset the action probabilities.
        lrp.reset_actions()
        count = 0
        # print("here, waiting for A")
        # print("here, waiting for B")
        # lrp.b = tune.find_optimal_b(lrp, env, det_obj)
        # Run a single experiment. Terminate if it reaches 10000 iterations.
        while(True and count < 10000):
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
            count += 1
            # print(count)
        # if(j == time_between):
    print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
    print("*************************************************************")
    mse.next_env()
