'''A simple module to see what my CSV from matlab looks like in python.'''
import csv
import numpy as np


a = []

with open('environments.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        a.append(row)

a = np.array(a)

print(a[:, 1])
