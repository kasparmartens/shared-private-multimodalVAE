import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def calculate_Rsq_torch(y, y_pred):
    with torch.no_grad():
        SSR = ((y - y_pred) ** 2).mean(dim=0)
        SST = ((y - y.mean(dim=0)) ** 2).mean(dim=0)
        Rsq = 1 - SSR / SST
    return Rsq.cpu().numpy()

def calculate_Rsq_numpy(y, y_pred):
    SSR = ((y - y_pred) ** 2).mean(axis=0)
    SST = ((y - y.mean(axis=0)) ** 2).mean(axis=0)
    Rsq = 1 - SSR / SST
    return Rsq

def calculate_Rsq(y, y_pred):
    if isinstance(y_pred, np.ndarray):
        return calculate_Rsq_numpy(y, y_pred)
    elif isinstance(y_pred, torch.Tensor):
        return calculate_Rsq_torch(y, y_pred)
    else:
        raise NotImplementedError
    
def corr(v1, v2):
    n = v1.shape[0]
    # v1, v2 = df1.values, df2.values
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n)
