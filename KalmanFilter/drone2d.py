from kalman import KalmanFilter
from plot import plot_2d, create_gif
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    B = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], dtype=np.float32)
    R = np.dot(np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]]), np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]]).T)
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    Q = np.eye(2)*8

    kf = KalmanFilter(A, B, R, C, Q)

    μ = np.zeros((4,1), dtype=np.float32)
    Σ = np.zeros((4,4), dtype=np.float32)

    x = np.zeros((4,1), dtype=np.float32)

    file_list = []

    # fig, ax = plt.subplots(figsize=(6, 6))

    for t in range(1, 31):
        μ_hat, Σ_hat = kf.predict(μ, Σ)

        noise = np.random.normal(0, 1, 2)
        x[0] += 0.5*noise[0]
        x[1] += 0.5*noise[1]
        x[2] += noise[0]
        x[3] += noise[1]

        z = kf.sense(x)
        μ, Σ = kf.measure(μ_hat, Σ_hat, z)

        plot_2d(x, μ, Σ,t, './drone2dplots/t=%d.jpg'%t)
        file_list.append('./drone2dplots/t=%d.jpg'%t)

        print(x)
        print(z)
    # plt.savefig('./drone2dplots/2d.jpg')
    # plt.show()
    create_gif(file_list)
