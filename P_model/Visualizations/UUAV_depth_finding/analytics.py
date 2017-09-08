'''A script that gathers analytical data regarding the automata.'''
from lrp import Linear_Reward_Penalty as LRP
from mse import MSE
from environment import Environment
from pinger import Pinger
import numpy as np
# import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
# import tune_lrp as tune

# Define the number of discrete depths between the surface and seabed.
num_actions = 5
n = 10000
interval = 1
time_between = (n / interval) - 1

# Metrics
# ============================================================================
# The number of actions until the automata converges on an environment.
converge = np.zeros(5)
correct_actions = np.zeros(5)
incorrect_actions = np.zeros(5)
total_dist = np.zeros(5)

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
      [0.021431, 0.071479, 0.40562, 0.50147, 0.00],
      [0.018288, 0.083153, 0.50582, 0.39274, 0.00],
      [0.48455, 0.015527, 0.18197, 0.31795, 0.00],
      [0.01675, 0.58845, 0.11313, 0.28167, 0.00]]
mse = MSE(Es)
det_obj = Pinger(mse.env_now())  # Create the detectable object.
# Run 5 individual experiments experiments.
for k in range(len(mse.envs)):
    # Generate an ensemble of n experiments
    det_obj.set_env(mse.env_now())
    # lrp.a = tune.find_optimal_a(lrp, env, det_obj)
    # print("Optimized value for a is: " + str(lrp.a))
    lrp.a = 0.99999999999999
    lrp.b = 0.5
    bestdepth = np.zeros(num_actions)
    current_best = 0
    conv_per_k = []
    for j in range(n):
        # reset the action probabilities.
        # lrp.reset_actions()
        count = 0
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
            converge[k] += count + 1
            if(max(lrp.p) > 0.98):
                # The best depth counting from 0.
                # Break at 98% convergence to a single depth.
                bestdepth[np.argmax(lrp.p)] += 1
                break
            count += 1
            conv_per_k.append(count)
        if (current_best != np.argmax(bestdepth)):
            current_best = np.argmax(bestdepth)
            total_dist[k] += 14
        # Plot conv per k
    converge[k] = converge[k] / (n + 1)
    print("The convergence vector is: " + str(converge[k]))
    print("The desired vector is now: " + str(mse.env_now()))
    print("The learned vector is: " + str(bestdepth / sum(bestdepth)))
    print("The rate of convergence is: " + str(converge[k]))
    print("Best depth is: " + str(np.argmax(bestdepth) * 14 + 14) + "m. " +
          "The desired depth is: " + str(np.argmax(mse.env_now()) * 14 + 14) +
          "m.")
    print("*************************************************************")
    mse.next_env()
print("The distance covered by the automata before convergence is: " +
      str(total_dist))
# n, bins, patches = plt.hist(converge, 5, facecolor='g')
# plt.xlabel('Environment')
# plt.ylabel('Average Actions per 100000 Trials')
# plt.title('Actions Required for 95% Accurate Convergence')
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([1, 5, 0, 16])
# plt.grid(True)
# plt.show()
# data = [go.Bar(
#         x=['Environment 1', 'Environment 2', 'Environment 3',
#            'Environment 4',
#            'Environment 5'],
#         y=converge.tolist())]

# py.iplot(data, filename='basic-bar')
