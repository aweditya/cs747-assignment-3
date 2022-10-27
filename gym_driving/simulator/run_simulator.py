from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

def tiling(lo, hi, bins, offsets):
    split_points = []
    ndim = len(bins)
    for dim in range(ndim):
        split_points.append(np.linspace(lo[dim], hi[dim], bins[dim] + 1)[1:-1] + offsets[dim])

    return split_points

def create_tilings(lo, hi, bins_and_offsets):
    tilings = []
    for bins, offsets in bins_and_offsets:
        tilings.append(tiling(lo, hi, bins, offsets))
    
    return tilings

def single_tiling_code(state, tiling):
    ndim = len(state)
    discretised_state = []
    for dim in range(ndim):
        discretised_state.append(int(np.digitize(state[dim], tiling[dim])))

    return tuple(discretised_state)

def tile_coding(state, tilings):
    encoding = [single_tiling_code(state, tiling) for tiling in tilings]

    return encoding

class Task1():
    def get_feature(self, state):
        x = state[0]
        y = state[1]
        vel = state[2]
        angle = (state[3] + 360) % 360

        x_bin = np.linspace(-400, 400, 800 // 25 + 1)
        y_bin = np.linspace(-400, 400, 800 // 25 + 1)
        vel_bin = np.linspace(0, 20, 20 * 2 + 1)
        angle_bin = np.linspace(0, 360, 360 // 3 + 1)

        feature_x = np.zeros(800 // 25)
        feature_x[np.where(x_bin <= x)[0][-1]] = 1
        
        feature_y = np.zeros(800 // 25)
        feature_y[np.where(y_bin <= y)[0][-1]] = 1

        feature_vel = np.zeros(20 * 2)
        feature_vel[np.where(vel_bin <= vel)[0][-1]] = 1

        feature_angle = np.zeros(360 // 3)
        feature_angle[np.where(angle_bin <= angle)[0][-1]] = 1

        feature = np.concatenate([feature_x, feature_y, feature_vel, feature_angle])
        return feature
    
    # def get_feature(self, state):
    #     state[0] /= 350
    #     state[1] /= 350
    #     state[2] /= 20
    #     state[3] /= 360
    #     return state

    def __init__(self):
        """
        Can modify to include variables as required
        """
        super().__init__()

        # self.d = 4 
        self.d = (800 // 25) + (800 // 25) + (20 * 2) + (360 // 3)
        self.w_steer = np.zeros((3, self.d))
        self.w_acc = np.zeros((5, self.d))

        # Eligibility traces
        self.z_steer = np.zeros(self.d)
        self.z_acc = np.zeros(self.d)

    def linear_sarsa_lambda(self, s, a, s_dash, a_dash, r, alpha=0.125, gamma = 1.0, Lambda = 0.9):
        delta_steer = r + gamma * self.w_steer[a_dash[0]] @ s_dash - self.w_steer[a[0]] @ s
        self.z_steer = gamma * Lambda * self.z_steer + s
        self.w_steer[a[0]] += alpha * delta_steer * self.z_steer

        delta_acc = r + gamma * self.w_acc[a_dash[1]] @ s_dash - self.w_acc[a[1]] @ s
        self.z_acc = gamma * Lambda * self.z_acc + s
        self.w_acc[a[1]] += alpha * delta_acc * self.z_acc

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        epsilon = 1e-1
        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action_steer = np.random.randint(0, 3)
        else:
            q_steer = self.w_steer @ self.get_feature(state)
            # action_steer = np.random.choice(np.flatnonzero(q_steer == np.max(q_steer)))
            action_steer = np.argmax(q_steer)

        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action_acc = np.random.randint(0, 5)
        else:
            q_acc = self.w_acc @ self.get_feature(state)
            # action_acc = np.random.choice(np.flatnonzero(q_acc == np.max(q_acc)))
            action_acc = np.argmax(q_acc)

        action = np.array([action_steer, action_acc])  
        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            s = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # Linear SARSA(λ)
            self.z_steer = np.zeros(self.d)
            self.z_acc = np.zeros(self.d)

            a = self.next_action(self.get_feature(s))
            # a = self.next_action(s)
            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                s_dash, r, terminate, reached_road, info_dict = simulator._step(a)
                fpsClock.tick(FPS)

                cur_time += 1

                a_dash = self.next_action(s_dash)
                self.linear_sarsa_lambda(self.get_feature(s), a, self.get_feature(s_dash), a_dash, r)
                # self.linear_sarsa_lambda(s, a, s_dash, a_dash, r)

                s = s_dash  
                a = a_dash

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = None
        action_acc = None

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
