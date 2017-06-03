'''This module tunes the LRP to optimize for the environment'''
import numpy as np


def find_accuracy(la, env, teacher, a=0.9, b=0.1):
    ensemble_average = 0
    ensemble_size = 10000
    la.a = a
    la.b = b
    for i in range(ensemble_size):
        ensemble_average += simulate_once(la, env, teacher)

    ensemble_average = ensemble_average / ensemble_size
    return ensemble_average


def find_optimal_a(la, env, teacher, desired_accuracy=0.95,
                   low=0.0001, high=0.999999):
    L = low
    H = high

    while(percent_diff(L, H) >= 0.05):
        a = (L + H) / 2
        accuracy = find_accuracy(la, env, teacher, a)
        if(accuracy >= desired_accuracy):
            print("H = a, a = " + str(a))
            H = a
        else:
            print("H = a, a = " + str(a))
            L = a
    return H


def find_optimal_b(la, env, teacher, desired_accuracy=0.95,
                   low=0.0001, high=0.999999):
    L = low
    H = high

    while(percent_diff(L, H) >= 0.05):
        b = (L + H) / 2
        accuracy = find_accuracy(la, env, teacher, la.a, b)
        if(accuracy >= desired_accuracy):
            print("H = b, b = " + str(b))
            H = b
        else:
            print("L = b, b = " + str(b))
            L = b
    return H


def percent_diff(value1, value2):
    return abs(value1 - value2) / ((value1 + value2) / 2)


def simulate_once(la, env, teacher):
    # bestdepth = np.zeros(len(la.p))
    while(True):
            # Define m as the next action predicting the depth of the object.
            m = la.next_action()
            # Define req as the next detectable object depth.
            req = teacher.request()
            # reward if m = req.
            resp = env.response(m, req)
            if(not resp):
                la.do_reward(m)
            else:
                la.do_penalty(m)
            if(max(la.p) > 0.98):
                # The best depth counting from 0.
                # Break at 98% convergence to a single depth.
                break
    return round(la.p[np.argmax(teacher.E)] / sum(la.p), 0)
