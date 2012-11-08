import numpy as np

# Classification accuracy
def CA(data, predictions):
    Y = np.reshape(data.Y,-1);
    if len(Y) != len(predictions):
        raise ValueError("inconsistent size of data and predictions")
    return 1.0*sum(Y == predictions) / len(predictions)