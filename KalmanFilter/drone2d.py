from kalman import KalmanFilter
import numpy as np


if __name__ == '__main__':
    A = np.array([[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    B = np.array([[0.5, 0, 1, 0], [0, 0.5, 0, 1]], dtype=np.float32)
    R = np.dot(np.array([[0.5, 0, 1, 0], [0, 0.5, 0, 1]]), np.array([[0.5, 0, 1, 0], [0, 0.5, 0, 1]]).T)
    C = np.array([[1], [1], [0], [0]], dtype=np.float32)
    Q = np.array()

    kf = KalmanFilter(A, B, R, C, Q)

    for t in range(1, 31):
        pass