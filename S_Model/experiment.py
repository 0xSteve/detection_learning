'''This module will contain one round of learning and updating for the
   mobile-agent. This will serve to separate concerns from the simulation
   and the mobile-agent implementation.'''
from agent import Agent
from senvironment import SEnvironment
import numpy as np


class Experiment(object):

    def __init__(self, depth, channel_depth, p_vector, precision=1):
        self.agent = Agent(precision, depth, channel_depth)
        self.environment = SEnvironment(p_vector)
        self.learned_p = np.zeros(int(len(p_vector) / precision))

    def evaluate(self):
        self.agent.move()
        self.agent.receive(self.environment.response(self.agent.send()))
        self.agent.next_action()

    def ensemble_evaluation(self, number_iterations):
        for i in range(number_iterations):
            count = 0
            self.agent.lrp.reset_actions()
            while(count < 10000):
                self.evaluate()
                self.learned_p[self.agent.depth] += 1
                count += 1
            print(self.learned_p)
            self.learned_p = self.learned_p / count
            print(self.learned_p)
