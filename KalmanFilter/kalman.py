import numpy as np


class KalmanFilter:
    '''
    Kalman Filter algorithm
    '''

    def __init__(self, A, B, R, C=np.zeros((1,2)), Q=np.zeros((2,2))):
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert A.shape[0] == B.shape[0] and A.shape[0] == R.shape[0]

        self.A = A
        self.B = B
        self.R = R

        self.C = C
        self.Q = Q

    def predict(self, μ, Σ, u=np.zeros((2,1))):
        assert self.A.shape[1] == μ.shape[0]
        assert self.B.shape[1] == u.shape[0]
        μ_hat = np.dot(self.A, μ)
        Σ_hat = np.dot(np.dot(self.A, Σ), self.A.T)+self.R

        return μ_hat, Σ_hat

    def measure(self):
        pass