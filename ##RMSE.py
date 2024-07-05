##RMSE
import numpy as np
import pandas as pd

def RMSE(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# Example usage
predictions = np.array([10,20,30])
targets = np.array([12,18,28])
print(RMSE(predictions, targets))