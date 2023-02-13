import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Kalman_prediction(A, B, R, μ, Σ, u):
    '''
    The prediction step of Kalman Filter. Calculating the mean and covariance matrix at time step t+1.
    '''
    assert A.shape == (2,2)
    assert B.shape == (2,2)
    assert R.shape == (2.2)

    μ_hat = np.dot(A, μ)+np.dot(B, u)
    Σ_hat = np.dot(np.dot(A, Σ), A.T) + R

    return μ_hat, Σ_hat