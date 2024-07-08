##RMSE
import numpy as np
import pandas as pd

def RMSE(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# Example usage
predictions = np.array([10,20,30])
targets = np.array([12,18,30])
print(RMSE(predictions, targets))

##ADE
def ADE(predictions, targets):
    return np.mean(np.abs(predictions - targets))

# Example usage
predictions = np.array([10,20,30])
targets = np.array([12,18,30])
print(ADE(predictions, targets))


##FDE
def FDE(fpositions, tpositions):
    return np.mean(np.abs(fpositions - tpositions))


##Negative Log Likelihood

##def


##KL Divergence
