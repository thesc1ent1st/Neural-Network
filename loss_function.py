import numpy as np


def mse_loss(y_true, y_pred):
    """calculate loss, 'mean square error'"""
    return np.mean((y_true - y_pred)**2, dtype=np.float64)


y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))
