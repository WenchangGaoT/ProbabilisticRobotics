import numpy as np
from kalman import KalmanFilter


if __name__ == '__main__':

    # Q1: Kalman filter prediction

    A = np.array([[1, 1], [0, 1]], dtype=np.float32)
    B = np.array([[0, 0], [0, 0]], dtype=np.float32)
    R = np.array([[0.25, 0.5], [0.5, 1]],  dtype=np.float32)

    kf = KalmanFilter(A, B, R)

    μ = np.zeros((2, 1), dtype=np.float32)
    Σ = np.zeros((2, 2), dtype=np.float32)

    for t in range(1, 6):
        μ, Σ = kf.predict(μ, Σ)
        print(μ)
        print(Σ)

        # TODO: Plot
