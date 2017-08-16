'''This module will contain one round of learning and updating for the
   mobile-agent. This will serve to separate concerns from the simulation
   and the mobile-agent implementation.'''
from agent import Agent
import time
from senvironment import SEnvironment
import numpy as np


class Experiment(object):

    def __init__(self, depth, channel_depth, p_vector, precision=1):
        self.agent = Agent(precision, depth, channel_depth)
        self.environment = SEnvironment(p_vector)
        # The learned best depth is the index. Stored in the index is the
        # confidence that the index is the best.
        self.learned_best = np.zeros(int(len(p_vector) / precision))
        # Attempt to recreate the probability distribution of the environment.
        self.dist_est = np.zeros(int(len(p_vector) / precision))
        self.max = 10000  # Do 20k iterations.
        self.start = 0  # Allow 5000 iterations of training.
        # I need a metric to attest to the increase rate of entering the do-
        # nothing state. To achieve this, I will append the probability of
        # choosing this action as I approach the end of the ensemble. In this
        # way we can see that the ergodic LA will tend towards this action as
        # it encounters the maximum.
        self.action1_p = np.zeros(self.max - self.start)
        self.action0_p = np.zeros(self.max - self.start)
        self.action2_p = np.zeros(self.max - self.start)

    def evaluate(self):
        self.agent.move()
        self.agent.receive(self.environment.response(self.agent.send()))
        self.agent.next_action()

    def ensemble_evaluation(self, number_iterations):
        for i in range(number_iterations):
            count = 0
            data_counter = 0
            self.agent.lrp.reset_actions()
            # print(self.agent.depth)
            while(count < self.max):  # and count < 1000000
                if(count >= self.start):
                    # print("count is: " + str(count) +
                    #       ", p = " + str(self.agent.lrp.p))
                    # time.sleep(1)
                    self.action1_p[data_counter] += self.agent.lrp.p[1]
                    self.action0_p[data_counter] += self.agent.lrp.p[0]
                    self.action2_p[data_counter] += self.agent.lrp.p[2]
                    self.dist_est[self.agent.depth - 1] += 1
                    data_counter += 1
                self.evaluate()
                count += 1
                # if(max(self.agent.lrp.p) > 0.90):
                if(max(self.agent.lrp.p) > 0.98 and
                   count == self.max - 1):
                    # print("BOOOM DONE!!")
                    # print("************************************")
                    self.learned_best[self.agent.depth - 1] += 1
                    # break
            # self.action1_p = self.action1_p / (count - self.start)
        self.learned_best = self.learned_best / number_iterations
        self.dist_est = self.dist_est
        self.action1_p = self.action1_p / number_iterations
        self.action0_p = self.action0_p / number_iterations
        self.action2_p = self.action2_p / number_iterations
