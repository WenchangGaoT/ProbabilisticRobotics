import cv2
import numpy as np
import math
import os


UNIT = 50


class Environment:

    def __init__(self, m=25, sigma_movement=1, map='./pics/BayMap.png', mode='random', is_accurate=True):
        self.map = cv2.imread(map)
        self.map = np.transpose(self.map, (1, 0, 2))

        self.plot_dir = './BayPlots' if map == './pics/BayMap.png' \
                                    else './CityPlots' if map == './pics/CityMap.png' \
                                    else './MarioPlots'
        
        if not os.path.isdir(self.plot_dir): os.mkdir(self.plot_dir)


        self.σ_movement = sigma_movement
        self.m = m

        self.map_width, self.map_height = self.map.shape[0], self.map.shape[1]
        self.x_range = 0.5*self.map_width/UNIT
        self.y_range = 0.5*self.map_height/UNIT
        # print(self.x_range)
        # print(self.map_width)

        self.mode = mode

        # self.drone_x = np.random.uniform(0, 1)*self.x_range
        # self.drone_x = self.drone_x if np.random.uniform(0, 1) < 0.5 else -self.drone_x
        # self.drone_y = np.random.uniform(0, 1)*self.y_range
        # self.drone_y = self.drone_y if np.random.uniform(0, 1) < 0.5 else -self.drone_y
        self.drone_x = np.random.uniform(-self.x_range, self.x_range)
        self.drone_y = np.random.uniform(-self.y_range, self.y_range)

        self.is_accurate = is_accurate

    def drone2map(self, x, y):
        return (math.trunc(x*UNIT+0.5*self.map_width), math.trunc(y*UNIT+0.5*self.map_height))
    
    def generate_observation(self):
        '''
        Generating reference image with size m*m
        ----
        returns:
            reference_img: ndarray of shape (m, m, 3)
        '''
        # map_x, map_y = self.drone2map(self.drone_x, self.drone_y)
        # # print("map_x, map_Y:", map_x, map_y)
        # map_x = math.trunc(map_x/self.m)
        # map_y = math.trunc(map_y/self.m)
        # left_x = map_x*self.m
        # upper_y = map_y*self.m

        map_x, map_y = self.drone2map(self.drone_x, self.drone_y)
        left_x = max(math.trunc(map_x-0.5*self.m), 0)
        right_x = min(left_x+self.m, self.map_width)
        upper_y = max(math.trunc(map_y-0.5*self.m), 0)
        lower_y = min(upper_y+self.m, self.map_height)

        left_x = left_x if right_x-left_x==self.m else right_x-self.m
        upper_y = upper_y if lower_y-upper_y==self.m else lower_y-self.m

        ref_img = self.map[left_x:right_x, upper_y:lower_y, :]

        return ref_img
    
    def generate_ref(self, x, y):
        # print("x, y:", x, y)
        # map_x, map_y = self.drone2map(x, y)
        # # print("map_x, map_y", map_x, map_y)
        # map_x = math.trunc(map_x/self.m)
        # map_y = math.trunc(map_y/self.m)
        # left_x = map_x*self.m
        # upper_y = map_y*self.m

        # ref_img = self.map[left_x:left_x+self.m, upper_y:upper_y+self.m, :]
        # print(ref_img.shape)

        map_x, map_y = self.drone2map(x, y)
        left_x = max(math.trunc(map_x-0.5*self.m), 0)
        right_x = min(left_x+self.m, self.map_width)
        upper_y = max(math.trunc(map_y-0.5*self.m), 0)
        lower_y = min(upper_y+self.m, self.map_height)

        left_x = left_x if right_x-left_x==self.m else right_x-self.m
        upper_y = upper_y if lower_y-upper_y==self.m else lower_y-self.m

        ref_img = self.map[left_x:right_x, upper_y:lower_y, :]

        return ref_img 
    
    def generate_movement(self):
        '''
        Generating random movement at each time step
        ----

        returns:
            [dx, dy]T: ndarray of shape (2,)
        ''' 
        dx = np.random.uniform(-1, 1)
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

    def render(self, particles, beliefs, fname='1.jpg'):
        # print(self.drone2map(self.drone_x, self.drone_y))
        map = self.map.copy()
        # x, y = self.drone2map(self.drone_x, self.drone_y)
        # img = cv2.circle(map, (y, x), 10, color=(0, 0, 0), thickness=-1)

        for id, particle in enumerate(particles):
            x, y = self.drone2map(particle[0], particle[1])
            # print('circle x, y: ', x, y)
            map = cv2.circle(map, (y, x), 2+math.floor(beliefs[id]*5), color=(0, 255, 0), thickness=-1)

        x, y = self.drone2map(self.drone_x, self.drone_y)
        map = cv2.circle(map, (y, x), 10, color=(0, 0, 0), thickness=-1)

        img = np.transpose(map, (1, 0, 2))

        cv2.imshow("whole map", img)
        k = cv2.waitKey(0)
        print(k)
        if k == 115:
            cv2.imwrite(os.path.join(self.plot_dir, fname), img)

        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    env = Environment()
    env.generate_observation()
    env.render([(0, 0)], [1])
    mv = env.generate_movement()
    env.step(mv[0], mv[1])
    env.render([(0.1, 0.1)], [1])
