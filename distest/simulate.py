'''Underwater Link Finding simulation with a unimodal distribution.'''
import csv
from experiment import Experiment
from random import uniform as u
# First, we must read in the probability vector.
f = open('matlab files/bimodalNormalVector.csv', 'r')
p_vec = []
reader = csv.reader(f)
for row in reader:
    p_vec.append(float(row[0]))
p_vec.pop()
rando = int(u(0, 70))  # This will be the initial depth of the mobile-agent.
# Create a new experiment with a randomly seeded depth for the agent and
# precision of 1m, max depth of 70m.
experiment = Experiment(rando, 70, p_vec, 10)

experiment.true_max_estimate(150000)
w = open('matlab files/maxEstimate.csv', 'w')
for i in range(len(experiment.dist_est) - 1):
    w.write(str(experiment.dist_est[i]) + '\n')
w.write(str(experiment.dist_est[len(experiment.dist_est) - 1]))
w.close()
print("The expectation of moves to learn the distribution is: " +
      str(experiment.expectation))
print("The failure rate is: " +
      str(experiment.failure_rate))
print("Simulation complete!")
