'''Some helper functions.'''


def make_p(count):
    '''A helper function that generates count based on the number of
    actions.'''
    a = []
    for i in range(count):
        a.append(1 / count)
    return a


def subtract_nonzero(array, amount):
    for i in range(len(array)):
        if(array[i] > 0):
            array[i] -= amount
    return array


def make_dp(count):
    a = []
    for i in range(count):
        a.append(count)
    return a


def cdf(p_vector):
    '''get the cumulative distribution vector for a given input vector.'''
    cdf = []
    sigma = 0
    for i in range(len(p_vector)):
        sigma += p_vector[i]
        cdf.append(sigma)
    return cdf


def get_index(desired_action, cdf_array):
    index = 0  # Return the first action as default.
    for i in range(len(cdf_array)):
        # Not actually looking for p looking for the CDF.
        if(desired_action < cdf_array[i]):
            index = i
            break
    return index


def reward_function(a1, a2):
    pass
    # return int(not abs(a1 - a2) < 2)
    # return int(not (a1 == a2))
