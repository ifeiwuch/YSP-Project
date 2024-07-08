import numpy as np
import pandas as pd

#ADE
def ADE(xpredictions, ypredictions, xtargets, ytargets):
    xerror = (xpredictions - xtargets)**2
    yerror = (ypredictions - ytargets)**2
    return (np.mean(xerror) + np.mean(yerror))/2 

#RMSE
def RMSE(xpredictions, ypredictions, xtargets, ytargets):
    print("RMSE: ")
    return np.sqrt(ADE(xpredictions, ypredictions, xtargets, ytargets))

#FDE
def FDE(xpredictions, ypredictions, xtargets, ytargets):
    xerror = (xpredictions - xtargets)**2
    yerror = (ypredictions - ytargets)**2
    print("FDE: ")
    return np.sqrt(np.mean(xerror + yerror))

##Negative Log Likelihood
def NLL(predictions, targets):
    return -np.mean(targets*np.log(predictions))

##KL Divergence
def KLD(xpredictions, ypredictions, xtargets, ytargets):
    print("KLD: ")
    KLx = np.sum(xpredictions*np.log(xpredictions/xtargets))
    KLy = np.sum(ypredictions*np.log(ypredictions/ytargets))
    return (KLx + KLy)/2

# Example usage
predictions = np.array([10,20,30])
targets = np.array([12,18,30])
xpredictions = np.array([10,20,30])
xtargets = np.array([12,18,30])
ypredictions = np.array([10,20,30])
ytargets = np.array([12,18,30])
print(RMSE(xpredictions, ypredictions, xtargets, ytargets))
print(ADE(xpredictions, ypredictions, xtargets, ytargets))
print(FDE(xpredictions, ypredictions, xtargets, ytargets))
print(KLD(xpredictions, ypredictions, xtargets, ytargets))    
