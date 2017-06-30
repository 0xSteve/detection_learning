'''A module to visualize the underwater automata learning the best depth to
   operate at.'''
import turtle
from lrp import Linear_Reward_Penalty as LRP
from mse import MSE
from environment import Environment
from pinger import Pinger
import numpy as np
# import tune_lrp as tune
import time
#  setup window
window = turtle.Screen()
window.setup(800, 1000, 0, 0)
window.title("UUAV Learning the best depth")
window.register_shape("sub.gif")
turtle.colormode(255)

transmission = []
transmission1 = []
source = turtle.Turtle()
receiver = turtle.Turtle()
receiver1 = turtle.Turtle()
#  For 5 actions we need 5 depths
depths = [400, 200, 0, -200, -400]

#  Source setup
source.penup()
source.setpos(-300, 400)
source.shape("circle")
source.color("green")


#  Receiver setup
receiver.penup()
receiver.setpos(-100, 400)
receiver.shape("sub.gif")
receiver1.penup()
receiver1.setpos(100, 400)
receiver1.shape("sub.gif")

#  Simulation setup
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
lrp1 = LRP(num_actions)
# The most probable depth that the object exists at, as calculated by the
# learner.
bestdepth = np.zeros(num_actions)
bestdepth1 = np.zeros(num_actions)
# Define the Markovian Switching Environment that will feed probabilities to
# the Pinger object.
Es = [[0.48796, 0.024438, 0.067891, 0.41971, 0.00],
      [0.021431, 0.071479, 0.40562, 0.50147, 0.00],
      [0.018288, 0.083153, 0.50582, 0.39274, 0.00],
      [0.48455, 0.015527, 0.18197, 0.31795, 0.00],
      [0.01675, 0.58845, 0.11313, 0.28167, 0.00]]
mse = MSE(Es)
det_obj = Pinger(mse.env_now())  # Create the detectable object.
first_uav = Pinger(np.array([0.32, 0.1, 0.06, 0.42, 0.1]))
#  set up transmission vectors
for i in range(num_actions):
        transmission.append(turtle.Turtle())
        transmission1.append(turtle.Turtle())


# Run 5 individual experiments experiments.
for k in range(len(mse.envs)):
    # Generate an ensemble of n experiments
    source.goto(-300, depths[k])
    receiver.clear()
    receiver1.clear()
    for i in range(num_actions):
        transmission[i].clear()
        transmission[i].color("green")
        transmission[i].shape("arrow")
        transmission[i].shapesize(.5, .5)
        transmission[i].penup()
        transmission[i].setpos(-300, depths[k])
        transmission[i].pendown()
        transmission[i].goto(-150, depths[i])
        transmission[i].write(mse.env_now()[i])

    det_obj.set_env(mse.env_now())
    print("The desired vector is now: " + str(mse.env_now()))
    # lrp.a = tune.find_optimal_a(lrp, env, det_obj)
    # print("Optimized value for a is: " + str(lrp.a))
    lrp.a = 0.99999999999999
    lrp.b = 0.5
    bestdepth = np.zeros(num_actions)
    bestdepth1 = np.zeros(num_actions)
    current_best = 0
    current_best1 = 0
    for j in range(n):
        # reset the action probabilities.
        # lrp.reset_actions()
        count = 0
        count1 = 0
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
        if (current_best != np.argmax(bestdepth)):
            receiver.goto(-100, depths[np.argmax(bestdepth)])
            current_best = np.argmax(bestdepth)

    receiver.goto(-100, depths[np.argmax(bestdepth)])
    receiver.write(bestdepth[np.argmax(bestdepth)] / sum(bestdepth))
    for i in range(num_actions):
        transmission1[i].clear()
        transmission1[i].color("green")
        transmission1[i].shape("arrow")
        transmission1[i].shapesize(.5, .5)
        transmission1[i].penup()
        transmission1[i].setpos(-90, depths[np.argmax(bestdepth)])
        transmission1[i].pendown()
        transmission1[i].goto(50, depths[i])
        transmission1[i].write(first_uav.E[i])
    while(True and count1 < 10000):
        # Define m as the next action predicting the depth of the object.
        m = lrp1.next_action()
        # Define req as the next detectable object depth.
        req1 = first_uav.request()
        # reward if m = req.
        resp1 = env.response(m, req)
        if(not resp):
            lrp1.do_reward(m)
        else:
            lrp1.do_penalty(m)
        if(max(lrp1.p) > 0.999):
            # The best depth counting from 0.
            # Break at 98% convergence to a single depth.
            bestdepth1[np.argmax(lrp1.p)] += 1
            break
        count1 += 1
    if (current_best1 != np.argmax(bestdepth1)):
        receiver1.goto(100, depths[np.argmax(bestdepth1)])
        current_best1 = np.argmax(bestdepth1)
    receiver1.goto(100, depths[np.argmax(bestdepth1)])
    receiver1.write(bestdepth[np.argmax(bestdepth1)] / sum(bestdepth1))
    print("The probability vector is: " + str(bestdepth / sum(bestdepth)))
    print("Best depth is: " + str(np.argmax(bestdepth) * 14 + 14) + "m. " +
          "The desired depth is: " + str(np.argmax(mse.env_now()) * 14 + 14) +
          "m.")
    print("*************************************************************")
    mse.next_env()
    time.sleep(5)
print("Ready to exit.")
#  Exit
turtle.exitonclick()
