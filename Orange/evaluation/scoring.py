import numpy as np

# Classification accuracy
def CA(data, predictions):
    Y = np.reshape(data.Y,-1);
    if len(Y) != len(predictions):
        raise ValueError("inconsistent size of data and predictions")
    correct = 0
    for i in range(len(predictions)):
        if Y[i]==predictions[i]: correct+=1
    return 1.0*correct/len(predictions)