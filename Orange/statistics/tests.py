import math
import numpy as np
import scipy

def wilcoxon_rank_sum(d1, d2):
    # TODO Check this function!!!
    N1, N2 = np.sum(d1[1, :]), np.sum(d2[1, :])
    ni1, ni2 = d1.shape[1], d2.shape[1]
    i1 = i2 = 0
    R = 0
    rank = 0
    while i1 < ni1 and i2 < ni2:
        if d1[0, i1] < d2[0, i2]:
            R += (rank + (d1[1, i1] - 1) / 2) * d1[1, i1]
            rank += d1[1, i1]
            i1 += 1
        elif d1[0, i1] == d2[0, i2]:
            br = d1[1, i1] + d2[1, i2]
            R += (rank + (br - 1) / 2) * d1[1, i1]
            rank += br
            i1 += 1
            i2 += 1
        else:
            rank += d2[1, i2]
            i2 += 1
    if i1 < ni1:
        s = np.sum(d1[1, i1:])
        R += (rank + (s - 1) / 2) * s
    U = R - N1 * (N1 + 1) / 2
    m = N1 * N2 / 2
    var = m * (N1 + N2 + 1) / 6
    z = abs(U - m) / math.sqrt(var)
    p = 2 * (1 - scipy.special.ndtr(z))
    return z, p
