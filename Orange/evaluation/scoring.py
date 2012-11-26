import numpy as np


# Classification accuracy
def CA(data, predictions):
    Y = np.reshape(data.Y, -1)
    if len(Y) != len(predictions):
        raise ValueError("inconsistent size of data and predictions")
    return 1.0 * sum(Y == predictions) / len(predictions)


# Area under ROC for binary class
def AUC_binary(data, prob):
    P, N = np.sum(data.Y == 1), np.sum(data.Y == 0)
    if P == 0 or N == 0:
        raise ValueError("no positive or no negative values")
    dx, dy = 1.0 / N, 1.0 / P
    pred = sorted([(p[1], y == 1) for y, p in zip(data.Y[:, 0], prob)], reverse=True)
    x, y = 0, 0
    auc = 0
    for p, ok in pred:
        x2, y2 = x, y
        if ok:
            y2 += dy
        else:
            x2 += dx
        auc += 0.5 * (y2 + y) * (x2 - x)
        x, y = x2, y2
    return auc
