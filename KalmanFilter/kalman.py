import numpy as np


class KalmanFilter:
    '''
    Kalman Filter algorithm
    '''

    def __init__(self, A, B, R, C=np.zeros((1,2)), Q=np.zeros((1,1))):

        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(Q, np.ndarray)
        assert A.shape[0] == B.shape[0] and A.shape[0] == R.shape[0]
        assert C.shape[0] == Q.shape[0] and Q.shape[1] == Q.shape[0]

        self.A = A
        self.B = B
        self.R = R

        self.C = C
        self.Q = Q

    def predict(self, μ, Σ, u=np.zeros((2,1))):
        assert self.A.shape[1] == μ.shape[0]
        assert self.B.shape[1] == u.shape[0]
        μ_hat = np.dot(self.A, μ) + np.dot(self.B, u)
        Σ_hat = np.dot(np.dot(self.A, Σ), self.A.T) + self.R

        return μ_hat, Σ_hat
    
    def sense(self, x):
        assert isinstance(x, np.ndarray)
        assert self.C.shape[1] == x.shape[0]

        z = np.dot(self.C, x)+np.random.multivariate_normal(mean=np.zeros((self.C.shape[0])), cov=self.Q)
        return z

    def measure(self, μ_hat, Σ_hat, z):
        # Calculating Kalman Gain
        K = np.dot(np.dot(self.C, Σ_hat), self.C.T) + self.Q
        K = np.linalg.inv(K)
        K = np.dot(np.dot(Σ_hat, self.C.T), K)

        # Measurement
        μ = μ_hat + np.dot(K, z-np.dot(self.C, μ_hat))
        Σ = np.dot(np.eye(μ.shape[0])-np.dot(K, self.C), Σ_hat)

        return μ, Σ