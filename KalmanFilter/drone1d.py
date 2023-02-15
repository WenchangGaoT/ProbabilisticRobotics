import numpy as np
from kalman import KalmanFilter
from plot import plot_uncertainty_ellipse, plot_histogram


def sample_with_failure(kf, p=0):
    μ = np.zeros((2,1), dtype=np.float32)
    Σ = np.zeros((2,2), dtype=np.float32)
    z = kf.sense(μ)
    x = np.zeros((2,1), dtype=np.float32)

    for t in range(1, 21):
        μ, Σ = kf.predict(μ, Σ)
        # z = kf.sense(μ)
        x += np.array([[0.5], [1]], dtype=np.float32)*np.random.normal(0, 1)
        if np.random.rand() > p: 
            z = kf.sense(x)
            μ, Σ = kf.measure(μ, Σ, z)
    
    return x[0]-μ[0]


if __name__ == '__main__':

    # Q1: Prediction
    print('Question 1: Prediction')
    print()

    A = np.array([[1, 1], [0, 1]], dtype=np.float32)
    B = np.array([[0, 0], [0, 0]], dtype=np.float32)
    R = np.array([[0.25, 0.5], [0.5, 1]],  dtype=np.float32)
    C = np.array([[1, 0]], dtype=np.float32)
    Q = np.array([[8]], dtype=np.float32)

    kf = KalmanFilter(A, B, R, C, Q)

    # Starting from 0,0 with no uncertainty
    μ = np.zeros((2, 1), dtype=np.float32)
    Σ = np.zeros((2, 2), dtype=np.float32)

    μ_list = []
    Σ_list = []

    for t in range(1, 6):
        μ, Σ = kf.predict(μ, Σ)
        print(f'predicted μ at time step {t} is {μ}')
        print(f'predicted Σ at time step {t} is {Σ}')
        print()

        μ_list.append(μ)
        Σ_list.append(Σ)
    # Plot the uncertainty ellipse 
    plot_uncertainty_ellipse(μ_list, Σ_list)

    print('-----------------')
    print()

    
    # Question2: Measurement
    print('Question 2: Measurement')
    print()

    # Q2.2
    print('Q2.2:')
    z = 10.
    μ_5, Σ_5 = kf.measure(μ, Σ, z)
    print(f'mu_5 is: {μ_5}')
    print(f'Sigma_5 is: {Σ_5}')
    print()
    # print(f'Kalman gain is:{K}')

    # Q2.3
    print('Q2.3:')
    p_list = [0.1, 0.5, 0.9]
    N = 10000
    for p in p_list:
        samples = []
        for n in range(N):
            samples.append(sample_with_failure(kf, p))
        samples = np.array(samples)
        print(np.mean(samples))
        # samples = np.abs(samples)
        plot_histogram('./drone1dplots/p_failure=%f.jpg'%p, samples, N)


    print('-----------')
    print()


    # Question 3: Movement
    print('Question 3: Movement')


    B = np.array([[0.5], [1]])
    μ = np.array([[5.], [1.]])
    Σ = np.zeros((2, 2))
    u = np.array([[1.]])

    kf = KalmanFilter(A, B, R, C, Q)
    μ_hat, Σ_hat =  kf.predict(μ, Σ, u)
    print(f'Expected mean: {μ_hat}')
    print(f'Estimated covariance matrix: {Σ_hat}')


