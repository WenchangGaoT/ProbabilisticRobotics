import numpy as np


def simple_similarity(img1, img2):
    diff = np.sum(np.abs(img1-img2))
    return 1-diff/(np.sum(img1)+np.sum(img2))

class ParticleFilter:

    def __init__(self, M, env, metric=simple_similarity):
        self.M = M
        self.env = env
        self.metric = metric

        self.weights = np.ones((M))
        self.particles = []
        x_range, y_range = env.x_range, env.y_range
        for _ in range(M):
            x = np.random.uniform(-x_range, x_range)
            y = np.random.uniform(-y_range, y_range)
            self.particles.append((x, y))

    def resample(self):
        
    
