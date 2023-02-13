import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from kalman import KalmanFilter


def plot_uncertainty_ellipse(μ_list, Σ_list, N=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    facecolors = ['red', 'blue', 'yellow', 'green', 'orange']

    dataset = np.random.multivariate_normal(np.reshape(μ_list[1], (2,)), Σ_list[1], N)
    x = dataset[:, 0]
    y = dataset[:, 1]
        
    pts = ax.scatter(x, y, s=0.5)

    for t in range(5):
        # ax.scatter(0, 0)

        μ = μ_list[t]
        μ = np.reshape(μ, (2,))
        Σ = Σ_list[t]

        facecolor = facecolors[t]

        pearson = Σ[0, 1]/np.sqrt(Σ[0, 0] * Σ[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, facecolor='none', label='t=%d'%(t+1), edgecolor=facecolor)

        scale_x = np.sqrt(Σ[0][0])
        scale_y = np.sqrt(Σ[1][1])
        mean_x = μ[0]
        mean_y = μ[1]

        transf = transforms.Affine2D() \
                    .rotate_deg(45) \
                    .scale(scale_x, scale_y) \
                    .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        # print(dataset.shape)
        pts.set_visible(False)
        ax.add_patch(ellipse)

    ax.legend()
    plt.savefig('./drone1dplots/uncertainty_ellipses.jpg')
    plt.show()

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
    # TODO
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
        pass


    print('-----------')
    print()
    # Question 3: Movement
    print('Question 3: Movement')

