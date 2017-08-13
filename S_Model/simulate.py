'''Underwater Link Finding Experiment with unimodal distrib.'''
from lrp import Linear_Reward_Penalty as LRP
from mse import MSE
from environment import Environment
from pinger import Pinger
import numpy as np
import csv

f = open('unimodalNormalVector.csv', 'r')
P_env = []
n = 10000
reader = csv.reader(f)
for row in reader:
    P_env.append(float(row[0]))
Es = [P_env]
env = Environment(3)
lrp = LRP(3)  # The learning automata.
# The most probable depth that the object exists at, as calculated by the
# learner.
bestdepth = np.zeros(70) #  granularity of 1 meter.
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
        print("The desired vector is now: " + str(mse.env_now()))
    print("The learned vector is: " + str(bestdepth / sum(bestdepth)))
    print("*************************************************************")
    mse.next_env()
