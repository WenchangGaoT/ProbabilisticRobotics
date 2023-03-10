import numpy as np
import cv2
from environment import Environment
import matplotlib.pyplot as plt


def L1_similarity(img1, img2):
    # print(img1.dtype)
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    diff = np.sum(np.abs(img1-img2))
    # print(diff)
    # diff = np.sum(img1-img2)
    return 1-diff/(np.sum(img1)+np.sum(img2))

def L2_similarity(img1, img2):
    img1, img2 = np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)
    diff = np.sum((img1-img2)**2, axis=-1)
    diff = np.sqrt(diff)
    return 1-np.sqrt(diff)/(np.sum(img1)+np.sum(img2))

def hist_similarity(img1, img2):
    histRange = (0, 256)

    b_hist1 = cv2.calcHist(img1, [0], None, [256], (0, 256), accumulate=False)
    g_hist1 = cv2.calcHist(img1, [1], None, [256], histRange, accumulate=False)
    r_hist1 = cv2.calcHist(img1, [2], None, [256], histRange, accumulate=False)

    b_hist2 = cv2.calcHist(img2, [0], None, [256], (0, 256), accumulate=False)
    g_hist2 = cv2.calcHist(img2, [1], None, [256], histRange, accumulate=False)
    r_hist2 = cv2.calcHist(img2, [2], None, [256], histRange, accumulate=False)

    diff = np.sum(np.abs(b_hist1-b_hist2))+np.sum(np.abs(g_hist1-g_hist2))+np.sum(np.abs(r_hist1-r_hist2))
    return 1-diff/(np.sum(img1)+np.sum(img2))

def correlation(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [12, 12, 12],
        [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [12, 12, 12],
        [0, 256, 0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def chi_square(img1, img2):
    pass




class ParticleFilter:

    def __init__(self, M, env, metric=L1_similarity):
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
       x = np.random.uniform(0, weights[self.M-1])
    #    print(weights[self.M-1])
       for i, wi in enumerate(weights):
           if x <= wi:
               return i 
           
    def filter(self, u, z):
        ?? = self.env.??_movement
        new_particles = []
        weights = []
        for m in range(self.M):
            # Sampling new x
            new_x = self.particles[m][0]+u[0]+np.random.normal(0, ??)
            new_x = max(-self.x_range, new_x)
            new_x = min(self.x_range, new_x)
            new_y = self.particles[m][1]+u[1]+np.random.normal(0, ??)
            new_y = max(-self.y_range, new_y)
            new_y = min(self.y_range, new_y)
            new_particles.append((new_x, new_y))

            img = self.env.generate_ref(new_x, new_y)
            w = self.metric(z, img)
            w = w if w >= 0 else 0
            # Hist_similarity(z, img)
            # print(w)
            self.weights[m] = w
            # if w < 0: print('AHHHHH')
            w = w if m == 0 else w+weights[m-1]
            weights.append(w)

        # print(self.weights)

        for m in range(self.M):
            self.particles[m] = new_particles[self.resample(weights)]


if __name__ == '__main__':
    env = Environment(m=40, map='./pics/MarioMap.png', sigma_movement=0.25)
    filter = ParticleFilter(2000, env, metric=correlation)
    biases, vars = [], []
    rates = []
    T = 20
    for t in range(T):
        z = env.generate_observation()
        cv2.imshow("obs", np.transpose(z, (1, 0, 2)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        fname = '%d.png' % t

        env.render(filter.particles, filter.weights, fname=fname)
        u = env.generate_movement()
        bias = env.compute_bias(filter.particles)
        var = env.compute_variance(filter.particles)
        biases.append(bias)
        vars.append(var)
        rates.append(env.correct_rate(filter.particles))

        env.step(u[0], u[1])
        filter.filter(u, z)
    
    # plt.plot(range(1, 21), biases, label='bias')
    # plt.plot(range(1, 21), vars, label='variance')
    plt.plot(range(1, 21), rates, label='CorrectRate')
    plt.legend()
    plt.show()
        