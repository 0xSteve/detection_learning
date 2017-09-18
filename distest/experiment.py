'''This module will contain one round of learning and updating for the
   mobile-agent. This will serve to separate concerns from the simulation
   and the mobile-agent implementation.'''
from agent import Agent
from senvironment import SEnvironment
import numpy as np


class Experiment(object):

    def __init__(self, depth, channel_depth, p_vector, precision=1,
                 isNS=False):
        self.agent = Agent(precision, depth, channel_depth)
        self.max = 10000
        self.start = 0
        self.expectation = 0
        self.precision = precision
        self.dist_count = np.zeros(int(len(p_vector) / precision))
        self.failure_rate = 0
        if(not isNS):
            self.environment = SEnvironment(p_vector)
            self.dist_est = np.zeros(int(len(p_vector) / precision))
            self.learned_best = np.zeros(int(len(p_vector) / precision))
            self.action1_p = np.zeros(self.max - self.start)
            self.action0_p = np.zeros(self.max - self.start)
            self.action2_p = np.zeros(self.max - self.start)
        else:
            self.environment = [SEnvironment(p_vector[0]),
                                SEnvironment(p_vector[1])]
            self.dist_est = np.zeros(int(len(p_vector[0]) / precision))
            self.learned_best = np.zeros(int(len(p_vector[0]) / precision))
            self.action1_p = [np.zeros(self.max - self.start),
                              np.zeros(self.max - self.start)]
            self.action0_p = [np.zeros(self.max - self.start),
                              np.zeros(self.max - self.start)]
            self.action2_p = [np.zeros(self.max - self.start),
                              np.zeros(self.max - self.start)]

        # The learned best depth is the index. Stored in the index is the
        # confidence that the index is the best.
        # I need a metric to attest to the increase rate of entering the do-
        # nothing state. To achieve this, I will append the probability of
        # choosing this action as I approach the end of the ensemble. In this
        # way we can see that the ergodic LA will tend towards this action as
        # it encounters the maximum.

    def evaluate(self):
        self.agent.move()
        self.agent.receive(self.environment.response(self.agent.send()))
        self.agent.next_action()

    def evaluate_ns(self, env_index):
        self.agent.move()
        self.agent.receive(
            self.environment[env_index].response(
                self.agent.send()))
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

    def ensemble_evaluation_ns(self, number_iterations):
        for k in range(len(self.environment)):
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
                        self.action1_p[k][data_counter] += self.agent.lrp.p[1]
                        self.action0_p[k][data_counter] += self.agent.lrp.p[0]
                        self.action2_p[k][data_counter] += self.agent.lrp.p[2]
                        self.dist_est[self.agent.depth - 1] += 1
                        data_counter += 1
                    self.evaluate_ns(k)
                    count += 1
                    # if(max(self.agent.lrp.p) > 0.90):
                    if(max(self.agent.lrp.p) > 0.98 and
                       count == self.max - 1):
                        self.learned_best[self.agent.depth - 1] += 1
                        # break
                # self.action1_p = self.action1_p / (count - self.start)
                self.learned_best = self.learned_best / number_iterations
                self.dist_est = self.dist_est
            self.action1_p[k] = self.action1_p[k] / (number_iterations)
            self.action0_p[k] = self.action0_p[k] / (number_iterations)
            self.action2_p[k] = self.action2_p[k] / (number_iterations)

    def distribution_estimate(self, number_iterations):
        '''Find the complete distribution of a stationary stochastic
           environment and report the expected number of movements to
           find the entire distribution.'''
        temp = self.dist_est
        count = 0
        for i in range(number_iterations):
            self.agent.lrp.reset_actions()
            # self.agent.depth = int(u(0, 70))
            self.agent.depth = self.agent.starting_depth
            while(sum(self.dist_count) < len(self.dist_count)):
                self.evaluate()
                index = self.agent.depth - 1
                self.dist_count[index] = 1
                temp[index] = self.environment.p[index * self.precision]
                count += 1
                # if(max(self.agent.lrp.p) > 0.90):
                if(self.agent.lrp.p[1] > 0.98):
                    break
            self.dist_count = np.zeros(len(self.dist_count))
            self.dist_est = temp
        self.expectation = count / number_iterations

    def true_max_estimate(self, number_iterations):
        '''Find the complete distribution of a stationary stochastic
           environment and report the expected number of movements to
           find the entire distribution.'''
        temp = self.dist_est
        moves = 0
        successes = 0
        failures = 0
        for i in range(number_iterations):
            count = 0
            self.agent.reset_actions()
            # self.agent.depth = int(u(0, 70))
            self.agent.depth = self.agent.starting_depth
            index = self.agent.depth - 1
            while(np.amax(temp) != np.amax(self.environment.p)):
                self.evaluate()
                index = self.agent.depth
                self.dist_count[index - 1] = 1
                if(self.agent.action != self.agent.last_action):
                    temp[index - 1] = self.environment.p[index * self.precision - 1]
                    # in this the dist_est becomes frequency of travels
                    self.dist_est += 1
                    count += 1
                # if(max(self.agent.lrp.p) > 0.90):
                if(self.agent.lrp.p[1] > 0.98):
                    break
            moves += count
            is_successful = "Success."
            successes += 1
            if(np.amax(temp) != np.amax(self.environment.p)):
                is_successful = "####FAILURE####"
                successes -= 1
                failures += 1
            print("(" + str(i) + "/" + str(number_iterations) + ")" +
                  " maximum found: " + str(np.amax(temp)) +
                  " status: " + is_successful)
            temp = np.zeros(len(temp))
            # if(np.amax(temp) == np.amax(self.environment.p)):
            #     self.expectation = moves / i
            #     break
        self.expectation = moves / number_iterations
        self.failure_rate = failures / (successes + failures)
