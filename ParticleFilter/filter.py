import numpy as np
from environment import Environment


def simple_similarity(img1, img2):
    diff = np.sum(np.abs(img1-img2))
    return 1-diff/(np.sum(img1)+np.sum(img2))

class ParticleFilter:

    def __init__(self, M, env, metric=simple_similarity):
        assert isinstance(env, Environment)
        self.M = M
        self.env = env
        self.metric = metric

        self.weights = np.zeros((M))
        self.particles = []
        self.x_range, self.y_range = env.x_range, env.y_range
        for _ in range(M):
            x = np.random.uniform(-self.x_range, self.x_range)
            y = np.random.uniform(-self.y_range, self.y_range)
            self.particles.append((x, y))

    def resample(self, weights):
       x = np.random.uniform(0, weights[-1])
       for i, wi in enumerate(weights):
           if x <= wi:
               return i 
           
    def filter(self, u, z):
        σ = self.env.σ_movement
        new_particles = []
        weights = []
        for m in range(self.M):
            # Sampling new x
            new_x = self.particles[m][0]+u[0]+np.random.normal(0, σ)
            new_x = max(-self.x_range, new_x)
            new_x = min(self.x_range, new_x)
            new_y = self.particles[m][1]+u[1]+np.random.normal(0, σ)
            new_y = max(-self.y_range, new_y)
            new_y = min(self.y_range, new_y)
            new_particles.append((new_x, new_y))

            img = self.env.generate_ref(new_x, new_y)
            w = self.metric(z, img)
            self.weights[m] = w
            w = w if len(weights) == 0 else w+weights[-1]
            weights.append(w)
        
        for m in range(self.M):
            self.particles[m] = new_particles[self.resample(weights)]


if __name__ == '__main__':
    env = Environment()
    filter = ParticleFilter(1000, env)
    T = 5000
    for t in range(T):
        z = env.generate_observation()
        env.render(filter.particles, filter.weights)
        u = env.generate_movement()
        env.step(u[0], u[1])
        filter.filter(u, z)
        