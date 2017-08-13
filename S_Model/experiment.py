'''This module will contain one round of learning and updating for the
   mobile-agent. This will serve to separate concerns from the simulation
   and the mobile-agent implementation.'''
from agent import Agent
from senvironment import SEnvironment


class Experiment(object):

    def __init__(self, depth, channel_depth, p_vector, precision=1):
        self.agent = Agent(precision, depth, channel_depth)
        self.environment = SEnvironment(p_vector)

    def evaluate(self):
        self.agent.receive(self.environment.response(self.agent.send()))
        self.agent.next_action()
