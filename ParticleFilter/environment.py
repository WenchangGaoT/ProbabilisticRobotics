import cv2
import numpy as np

UNIT = 50


class Environment:

    def __init__(self, m=25, sigma_movement=1, map='./pics/BayMap.png', mode='random', is_accurate=True):
        self.map = cv2.imread(map)

        self.σ_movement = sigma_movement
        self.m = m

        self.map_width, self.map_height = self.map.shape[0], self.map.shape[1]
        self.x_range = 0.5*self.map_width/UNIT
        self.y_range = 0.5*self.map_height/UNIT
        print(self.x_range)

        self.mode = mode

        self.drone_x = np.random.uniform(0, 1)*self.x_range
        self.drone_x = self.drone_x if np.random.uniform(0, 1) < 0.5 else -self.drone_x
        self.drone_y = np.random.uniform(0, 1)*self.y_range
        self.drone_y = self.drone_y if np.random.uniform(0, 1) < 0.5 else -self.drone_y

        self.is_accurate = is_accurate

    def drone2map(self, x, y):
        return (int(x*UNIT+0.5*self.map_width), int(y*UNIT+0.5*self.map_height))
    
    def generate_observation(self):
        '''
        Generating reference image with size m*m
        ----
        returns:
            reference_img: ndarray of shape (m, m, 3)
        '''
        map_x, map_y = self.drone2map(self.drone_x, self.drone_y)
        left_x, upper_y = max(int(map_x-0.5*self.m), 0), max(int(map_y-0.5*self.m), 0)
        right_x, lower_y = min(int(left_x+self.m), self.map_width), min(int(upper_y+self.m), self.map_height)
        if right_x-left_x < self.m: left_x = right_x-self.m
        if lower_y-upper_y < self.m: upper_y = lower_y-self.m

        ref_img = self.map[left_x:right_x, upper_y:lower_y, :]

        # print(left_x, right_x)
        # print(ref_img.shape)
        # cv2.imshow("reference", ref_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return ref_img
    
    def generate_ref(self, x, y):
        map_x, map_y = self.drone2map(x, y)
        left_x, upper_y = max(int(map_x-0.5*self.m), 0), max(int(map_y-0.5*self.m), 0)
        right_x, lower_y = min(int(left_x+self.m), self.map_width), min(int(upper_y+self.m), self.map_height)
        if right_x-left_x < self.m: left_x = right_x-self.m
        if lower_y-upper_y < self.m: upper_y = lower_y-self.m

        ref_img = self.map[left_x:right_x, upper_y:lower_y, :]

        return ref_img 
    
    def generate_movement(self):
        '''
        Generating random movement at each time step
        ----

        returns:
            [dx, dy]T: ndarray of shape (2,)
        ''' 
        dx = np.random.uniform(0, 1)
        dy = np.sqrt(1-dx**2)
        dy = dy if np.random.uniform(0, 1)>0.5 else -dy
        return np.array([dx, dy], dtype=np.float32)
    
    def step(self, dx, dy):
        self.drone_x = self.drone_x+dx+np.random.normal(0, self.σ_movement)
        self.drone_x = max(-self.x_range, self.drone_x)
        self.drone_x = min(self.x_range, self.drone_x)

        self.drone_y = self.drone_y+dy+np.random.normal(0, self.σ_movement)
        self.drone_y = max(-self.y_range, self.drone_y)
        self.drone_y = min(self.y_range, self.drone_y)

    def render(self, particles, beliefs, fname='1.jpg', save=False):
        # print(self.drone2map(self.drone_x, self.drone_y))
        map = self.map.copy()
        x, y = self.drone2map(self.drone_x, self.drone_y)
        img = cv2.circle(map, (y, x), 10, color=(0, 0, 0), thickness=-1)

        for id, particle in enumerate(particles):
            # print(id)
            # print(self.drone2map(particle[0], particle[1]))
            x, y = self.drone2map(particle[0], particle[1])
            img = cv2.circle(img, (y, x), 3, color=(0, 255, 0), thickness=-1)
        # print(np.sum(img != self.map))
        if save: cv2.imwrite('./BayMap_plots/'+fname, img)
        cv2.imshow("whole map", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    env = Environment()
    # print(env.generate_ref())
    env.generate_observation()
    env.render([(0, 0)], [1])
    mv = env.generate_movement()
    env.step(mv[0], mv[1])
    env.render([(0.1, 0.1)], [1])
