'''Underwater Link Finding simulation with a unimodal distribution.'''
import csv
from experiment import Experiment
from random import uniform as u
import numpy as np
# First, we must read in the probability vector.
f = open('bimodalNormalVector.csv', 'r')
p_vec = []
reader = csv.reader(f)
for row in reader:
    p_vec.append(float(row[0]))
p_vec.pop()
f = open('otherNormalVector.csv', 'r')
p_vec2 = []
reader = csv.reader(f)
for row in reader:
    p_vec2.append(float(row[0]))
p_vec2.pop()
p_vec = [p_vec, p_vec2]
rando = int(u(0, 70))  # This will be the initial depth of the mobile-agent.
# Create a new experiment with a randomly seeded depth for the agent and
# precision of 1m, max depth of 70m.
experiment = Experiment(rando, 70, p_vec, 10, True)

experiment.ensemble_evaluation_ns(20)
print("********************************************************************")
a = np.array(experiment.learned_best)
b = np.array(p_vec)
print("The location of the best depth is: " +
      str(np.where(a == a.max())))
print("the confidence of the learned best depth is: " +
      str(max(experiment.learned_best)))
print("The maximums of the environment vector are at indices: " +
      str(np.where(b == b.max())))
# print("the max of the actual probability vector is: " + str(max(p_vec)))

# Write to a file to send to matlab.

# with open('learnedBest.csv', 'w') as csvfile:
#     w = csv.writer(csvfile, dialect='excel')
#     for i in range(len(experiment.learned_best)):
#         w.writerow(str(experiment.learned_best[i]))
w = open('learnedBest.csv', 'w')
for i in range(len(experiment.learned_best) - 1):
    w.write(str(experiment.learned_best[i]) + '\n')
w.write(str(experiment.learned_best[len(experiment.learned_best) - 1]))
w.close()

# w = open('learnedDist.csv', 'w')
# for i in range(len(experiment.dist_est) - 1):
#     w.write(str(experiment.dist_est[i]) + '\n')
# w.write(str(experiment.dist_est[len(experiment.dist_est) - 1]))
# w.close()

# w = open('Action1Probability.csv', 'w')
# for i in range(len(experiment.action1_p) - 1):
#     w.write(str(experiment.action1_p[i]) + '\n')
# w.write(str(experiment.action1_p[len(experiment.action1_p) - 1]))
# w.close()
# w = open('Action0Probability.csv', 'w')
# for i in range(len(experiment.action0_p) - 1):
#     w.write(str(experiment.action0_p[i]) + '\n')
# w.write(str(experiment.action0_p[len(experiment.action0_p) - 1]))
# w.close()
# w = open('Action2Probability.csv', 'w')
# for i in range(len(experiment.action2_p) - 1):
#     w.write(str(experiment.action2_p[i]) + '\n')
# w.write(str(experiment.action2_p[len(experiment.action2_p) - 1]))
# w.close()
# print("Simulation complete!")

w = open('Action1Probability.csv', 'w')
for k in range(2):
    for i in range(len(experiment.action1_p[k]) - 1):
        w.write(str(experiment.action1_p[k][i]) + '\n')
    w.write(str(experiment.action1_p[k][len(experiment.action1_p[k]) - 1]))
w.close()
w = open('Action0Probability.csv', 'w')
for k in range(2):
    for i in range(len(experiment.action0_p[k]) - 1):
        w.write(str(experiment.action0_p[k][i]) + '\n')
    w.write(str(experiment.action0_p[k][len(experiment.action0_p[k]) - 1]))
w.close()
w = open('Action2Probability.csv', 'w')
for k in range(2):
    for i in range(len(experiment.action2_p[k]) - 1):
        w.write(str(experiment.action2_p[k][i]) + '\n')
    w.write(str(experiment.action2_p[k][len(experiment.action2_p[k]) - 1]))
w.close()
print("Simulation complete!")
