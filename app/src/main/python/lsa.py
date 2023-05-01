import numpy as np
from scipy.optimize import linear_sum_assignment

def lsa(cost):
    # np_cost = np.array(cost)
    # print(np.array(cost))
    row_ind, col_ind = linear_sum_assignment(cost, True)
    final = list(list(zip(row_ind, col_ind)))
    return final