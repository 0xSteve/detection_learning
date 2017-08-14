'''Underwater Link Finding simulation with a unimodal distribution.'''
import csv
from experiment import Experiment
from random import uniform as u
import numpy as np
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
experiment = Experiment(30, 70, p_vec, 10)

experiment.ensemble_evaluation(3000)
print("*****************************************************************************")
print("the max of the learned probability vector is: " + str(max(experiment.learned_p)))
print("the max of the actual probability vector is: " + str(max(p_vec)))
a = np.array(experiment.learned_p)
b = np.array(p_vec)
print(len(experiment.learned_p))
print(np.where(a == a.max()))
print(np.where(b == b.max()))
print(a)
print(a[np.where(a == a.max())])
