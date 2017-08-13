'''Underwater Link Finding simulation with a unimodal distribution.'''
import csv
from experiment import Experiment
from random import uniform as u
# First, we must read in the probability vector.
f = open('unimodalNormalVector.csv', 'r')
p_vec = []
reader = csv.reader(f)
for row in reader:
    p_vec.append(float(row[0]))
p_vec.pop()
rando = int(u(0, 70))  # This will be the initial depth of the mobile-agent.
# Create a new experiment with a randomly seeded depth for the agent and
# precision of 1m, max depth of 70m.
experiment = Experiment(rando, 70, p_vec, 10)

experiment.evaluate()
