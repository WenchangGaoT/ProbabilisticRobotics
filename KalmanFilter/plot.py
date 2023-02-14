import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os


def plot_uncertainty_ellipse(μ_list, Σ_list, N=100):
    if not os.path.exists('./drone1dplots'): os.mkdir('./drone1dplots')
    fig, ax = plt.subplots(figsize=(6, 6))
    facecolors = ['red', 'blue', 'yellow', 'green', 'orange']

    dataset = np.random.multivariate_normal(np.reshape(μ_list[1], (2,)), Σ_list[1], N)
    x = dataset[:, 0]
    y = dataset[:, 1]
        
    pts = ax.scatter(x, y, s=0.5)
    
    ax.set_xlabel(r'$x_t$')
    ax.set_ylabel(r'$\dot{x}_t$')
    ax.set_title('Uncertainty Ellipses')

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


def plot_histogram(fname, μ, N):
    if not os.path.exists('./drone1dplots'): os.mkdir('./drone1dplots')

    print(μ.shape)
    μ = np.reshape(μ, (N,))

    # plt.set_xlabel('absolute value of error')
    # plt.set_ylabel('')

    plt.hist(μ, 100)
    plt.savefig(fname)
    plt.show()