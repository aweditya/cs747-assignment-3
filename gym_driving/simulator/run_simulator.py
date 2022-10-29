from secrets import choice
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

        x_bin = np.linspace(-400, 400, 10 + 1)
        y_bin = np.linspace(-400, 400, 10 + 1)
        vel_bin = np.linspace(0, 20, 20 // 5 + 1)
        angle_bin = np.linspace(0, 360, 10 + 1)

        feature_x = np.zeros(10)
        feature_x[np.where(x_bin <= x)[0][-1]] = 1
        
        feature_y = np.zeros(10)
        feature_y[np.where(y_bin <= y)[0][-1]] = 1

        feature_vel = np.zeros(20 // 5)
        feature_vel[np.where(vel_bin <= vel)[0][-1]] = 1

        feature_angle = np.zeros(10)
        feature_angle[np.where(angle_bin <= angle)[0][-1]] = 1

        feature = np.concatenate([feature_x, feature_y, feature_vel, feature_angle])
        # feature = np.concatenate([feature_x, feature_y, feature_angle])
        return feature

    def get_high_level_features(self, state):
        x = state[0]
        y = state[1]
        vel = state[2]
        angle = state[3]

        dist_goal = math.sqrt((350 - x) ** 2 + y ** 2)
        angle_goal_bottom = math.atan((y - 75) / (350 - x)) * 180 / (2 * np.pi)
        angle_goal_top = math.atan((y + 75) / (350 - x)) * 180 / (2 * np.pi)

        # print(dist_goal, angle_goal_bottom, angle_goal_top)
        return [dist_goal / (350 * math.sqrt(5)), angle_goal_bottom / 90, angle_goal_top / 90, angle / 180]

    def __init__(self):
        """
        Can modify to include variables as required
        """
        super().__init__()

        self.d = 4 
        # self.d = 10 + 10 + 4 + 10
        self.w_steer = np.random.normal(size=(3, self.d))
        self.w_acc = np.random.normal(size=(5, self.d))

        # Eligibility traces
        self.z_steer = np.zeros(self.d)
        self.z_acc = np.zeros(self.d)

        # Parametric policy
        self.pi_steer = np.ones(3) / 3
        self.pi_acc = np.ones(5) / 5


    def reinforce(self, history, alpha=0.1):
        w_steer_new = np.copy(self.w_steer)
        w_acc_new = np.copy(self.w_acc)

        s = history['s']
        a = history['a']
        r = history['r']

        T = len(s)
        G = 0.0
        for k in range(T):
            G += r[k]
        
        grad_ln_pi_steer = np.zeros(np.shape(self.w_steer))
        grad_ln_pi_acc = np.zeros(np.shape(self.w_acc))
        for t in range(T):
            choice_steer = self.w_steer @ self.get_feature(s[t])
            choice_steer -= np.max(choice_steer)
            self.pi_steer = np.exp(choice_steer) / np.sum(np.exp(choice_steer))
            for action_steer in range(3):
                if action_steer == a[t][0]:
                    grad_ln_pi_steer[action_steer] = (1 - self.pi_steer[action_steer]) * self.get_feature(s[t])
                else:
                    grad_ln_pi_steer[action_steer] = -self.pi_steer[action_steer] * self.get_feature(s[t])

            w_steer_new += alpha * G * grad_ln_pi_steer

            choice_acc = self.w_acc @ self.get_feature(s[t])
            choice_acc -= np.max(choice_acc)
            self.pi_acc = np.exp(choice_acc) / np.sum(np.exp(choice_acc))
            for action_acc in range(5):
                if action_acc == a[t][1]:
                    grad_ln_pi_acc[action_acc] = (1 - self.pi_acc[action_acc]) * self.get_feature(s[t])
                else:
                    grad_ln_pi_acc[action_acc] = -self.pi_acc[action_acc] * self.get_feature(s[t])

            w_acc_new += alpha * G * grad_ln_pi_acc

            G -= r[t]

        self.w_steer = w_steer_new
        self.w_acc = w_acc_new

    def linear_sarsa_lambda(self, s, a, s_dash, a_dash, r, terminate, alpha=0.125, gamma = 1.0, Lambda = 0.0):
        s = self.get_high_level_features(s)
        s_dash = self.get_high_level_features(s_dash)

        if terminate:
            delta_steer = r - self.w_steer[a[0]] @ s
            delta_acc = r - self.w_acc[a[1]] @ s
        else:
            delta_steer = r + gamma * self.w_steer[a_dash[0]] @ s_dash - self.w_steer[a[0]] @ s
            delta_acc = r + gamma * self.w_acc[a_dash[1]] @ s_dash - self.w_acc[a[1]] @ s

        self.z_steer = gamma * Lambda * self.z_steer + s
        self.w_steer[a[0]] += alpha * delta_steer * self.z_steer

        self.z_acc = gamma * Lambda * self.z_acc + s
        self.w_acc[a[1]] += alpha * delta_acc * self.z_acc

    # Linear SARSA(λ) action logic
    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        epsilon = 0.0
        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action_steer = np.random.randint(0, 3)
        else:
            q_steer = self.w_steer @ self.get_high_level_features(state)
            # action_steer = np.random.choice(np.flatnonzero(q_steer == np.max(q_steer)))
            action_steer = np.argmax(q_steer)

        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action_acc = np.random.randint(0, 5)
        else:
            q_acc = self.w_acc @ self.get_high_level_features(state)
            # action_acc = np.random.choice(np.flatnonzero(q_acc == np.max(q_acc)))
            action_acc = np.argmax(q_acc)

        action = np.array([action_steer, action_acc])  
        return action

    # REINFORCE action logic
    # def next_action(self, state):
    #     """
    #     Input: The current state
    #     Output: Action to be taken
    #     TO BE FILLED
    #     """

    #     # Replace with your implementation to determine actions to be taken
    #     choice_steer = self.w_steer @ self.get_feature(state)
    #     choice_steer -= np.max(choice_steer)
    #     self.pi_steer = np.exp(choice_steer) / np.sum(np.exp(choice_steer))

    #     choice_acc = self.w_acc @ self.get_feature(state)
    #     choice_acc -= np.max(choice_acc)
    #     self.pi_acc = np.exp(choice_acc) / np.sum(np.exp(choice_acc))
        
    #     action_steer = np.random.choice(a=np.arange(3), p=self.pi_steer)
    #     action_acc = np.random.choice(a=np.arange(5), p=self.pi_acc)

    #     action = np.array([action_steer, action_acc])  
    #     return action

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

            # REINFORCE 
            history = {}
            history['s'] = []
            history['a'] = []
            history['r'] = []            

            a = self.next_action(s)
            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                # a = self.next_action(s)
                s_dash, r, terminate, reached_road, info_dict = simulator._step(a)
                fpsClock.tick(FPS)

                cur_time += 1

                a_dash = self.next_action(s_dash)
                self.linear_sarsa_lambda(s, a, s_dash, a_dash, r, terminate)

                s = s_dash  
                a = a_dash

                history['s'].append(s)
                history['a'].append(a)
                history['r'].append(r)

                if terminate:
                    road_status = reached_road
                    break

            # self.reinforce(history)

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
        action_steer = 0
        action_acc = 0

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
