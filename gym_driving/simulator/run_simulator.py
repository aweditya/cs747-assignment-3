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
TRAIN_EPISODES = 5000

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

        # Steps to reach road:
        # 1) Turn to the x-axis (TURN_TO_X_AXIS)
        # 2) Move to the x-axis (MOVE_TO_X_AXIS)
        # 3) Turn and move towards the road (MOVE_TO_ROAD)
        self.behaviour = 'TURN_TO_X_AXIS'

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        x = state[0]
        y = state[1]
        angle = (state[3] + 360) % 360

        action_steer = 1
        action_acc = 2
    
        if self.behaviour == 'TURN_TO_X_AXIS':
            if y >= 10:
                if abs(angle - 270) < 3:
                    self.behaviour = 'MOVE_TO_X_AXIS'
                    action_acc = 3
                
                else:
                    if abs(angle - 273) < abs(angle - 267):
                        action_steer = 0
                    else:
                        
                        action_steer = 2
            
            elif y <= -10:
                if abs(angle - 90) < 3:
                    self.behaviour = 'MOVE_TO_X_AXIS'
                    action_acc = 3
                
                else:
                    if abs(angle - 93) < abs(angle - 87):
                        action_steer = 0
                    else:
                        
                        action_steer = 2
                
            else:
                self.behaviour = 'MOVE_TO_ROAD'

        elif self.behaviour == 'MOVE_TO_X_AXIS':
            if abs(y) >= 10:
                action_acc = 4

            else:
                action_acc = 0
                self.behaviour = 'MOVE_TO_ROAD'

        elif self.behaviour == 'MOVE_TO_ROAD':
            angle_road_top = math.atan2(-50 - y, 350 - x) * 180 / math.pi
            angle_road_top = (angle_road_top + 360) % 360

            angle_road_bottom = math.atan2(50 - y, 350 - x) * 180 / math.pi
            angle_road_bottom = (angle_road_bottom + 360) % 360

            if angle > angle_road_top or angle < angle_road_bottom:
                    action_acc = 4
            else:
                if abs(angle - angle_road_top) < abs(angle - angle_road_bottom):
                    action_steer = 2
                else:
                    action_steer = 0

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
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
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

            self.behaviour = 'TURN_TO_X_AXIS'

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():
    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

        # Steps to reach road:
        # 1) If there is an obstacle to the top/bottom, first turn right/left (TURN_AWAY_FROM_MUDPIT)
        # 2) Once you point in the current direction, move along the mudpit (MOVE_ALONG_MUDPIT)
        # 2) Once there is no obstacle to the top/bottom, turn to the x-axis (TURN_TO_X_AXIS)
        # 2) Move to the x-axis (MOVE_TO_X_AXIS)
        # 3) Turn and move towards the road (MOVE_TO_ROAD)
        self.behaviour = 'MOVE_AROUND_MUDPIT'
        self.mud_pits = []

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        # print(state, self.behaviour)
        x = state[0]
        y = state[1]
        angle = (state[3] + 360) % 360

        action_steer = 1
        action_acc = 2

        [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4] = self.mud_pits
        [ran_cen_1x, ran_cen_1y] = ran_cen_1 # Quadrant 1
        [ran_cen_2x, ran_cen_2y] = ran_cen_2 # Quadrant 4
        [ran_cen_3x, ran_cen_3y] = ran_cen_3 # Quadrant 2
        [ran_cen_4x, ran_cen_4y] = ran_cen_4 # Quadrant 3

        if self.behaviour == 'MOVE_AROUND_MUDPIT':
            if x >= 0 and y >= 0: # Quadrant 1
                if y >= 10:
                    if abs(x - ran_cen_1x) <= 110: # Obstacle in the way
                        if abs(angle - 180) < 3: # If you are in the right direction, move ahead
                            action_acc = 4

                        else: # Turn to the right direction
                            if abs(angle - 183) < abs(angle - 177):
                                action_steer = 0

                            else:                        
                                action_steer = 2

                    else: # No obstacle in the way
                        self.behaviour = 'TURN_TO_X_AXIS'
                        action_acc = 0
                else:
                    self.behaviour = 'MOVE_TO_ROAD'
                    action_acc = 0
                    
            elif x >= 0 and y < 0: # Quadrant 4
                if y <= -10:
                    if abs(x - ran_cen_2x) <= 110: # Obstacle in the way
                        if abs(angle - 180) < 3: # If you are in the right direction, move ahead
                            action_acc = 4

                        else: # Turn to the right direction
                            if abs(angle - 183) < abs(angle - 177):
                                action_steer = 0

                            else:                        
                                action_steer = 2

                    else: # No obstacle in the way
                        self.behaviour = 'TURN_TO_X_AXIS'
                        action_acc = 0

                else:
                    self.behaviour = 'MOVE_TO_ROAD'
                    action_acc = 0

            elif x < 0 and y >= 0: # Quadrant 2
                if y >= 10:
                    if abs(x - ran_cen_3x) <= 110: # Obstacle in the way
                        if abs(angle) < 3: # If you are in the right direction, move ahead
                            action_acc = 3

                        else: # Turn to the right direction
                            if abs(angle - 3) < abs(angle - 357):
                                action_steer = 0

                            else:                        
                                action_steer = 2

                    else: # No obstacle in the way
                        self.behaviour = 'TURN_TO_X_AXIS'
                        action_acc = 0
                
                else:
                    self.behaviour = 'MOVE_TO_ROAD'
                    action_acc = 0

            elif x < 0 and y < 0: # Quadrant 3
                if y <= -10:
                    if abs(x - ran_cen_4x) <= 110: # Obstacle in the way
                        if abs(angle) < 3: # If you are in the right direction, move ahead
                            action_acc = 3

                        else: # Turn to the right direction
                            if abs(angle - 3) < abs(angle - 357):
                                action_steer = 0

                            else:                        
                                action_steer = 2

                    else: # No obstacle in the way
                        self.behaviour = 'TURN_TO_X_AXIS'
                        action_acc = 0
                else:
                    self.behaviour = 'MOVE_TO_ROAD'
                    action_acc = 0
        
        elif self.behaviour == 'TURN_TO_X_AXIS':
            if y >= 10:
                if abs(angle - 270) < 3:
                    self.behaviour = 'MOVE_TO_X_AXIS'
                    action_acc = 4
                
                else:
                    if abs(angle - 273) < abs(angle - 267):
                        action_steer = 0

                    else:
                        action_steer = 2
            
            elif y <= -10:
                if abs(angle - 90) < 3:
                    self.behaviour = 'MOVE_TO_X_AXIS'
                    action_acc = 4
                
                else:
                    if abs(angle - 93) < abs(angle - 87):
                        action_steer = 0
                    else:
                        
                        action_steer = 2
                
            else:
                self.behaviour = 'MOVE_TO_ROAD'
                action_acc = 0

        elif self.behaviour == 'MOVE_TO_X_AXIS':
            if abs(y) >= 10:
                action_acc = 4

            else:
                action_acc = 0
                self.behaviour = 'MOVE_TO_ROAD'

        elif self.behaviour == 'MOVE_TO_ROAD':
            if ran_cen_1y - 110 < 50:
                angle_road_bottom = math.atan2(ran_cen_1y - 110 - y, 350 - x) * 180 / math.pi

            else:
                angle_road_bottom = math.atan2(50 - y, 350 - x) * 180 / math.pi

            if ran_cen_2y + 110 > -50:
                angle_road_top = math.atan2(ran_cen_2y + 110 - y, 350 - x) * 180 / math.pi

            else:
                angle_road_top = math.atan2(-50 - y, 350 - x) * 180 / math.pi

            angle_road_bottom = (angle_road_bottom + 360) % 360
            angle_road_top = (angle_road_top + 360) % 360
            
            # print(angle, angle_road_top, angle_road_bottom)
            if angle > 180:
                if angle > angle_road_top:
                    self.behaviour = 'FINISH'
                        
                else:
                    action_steer = 2
                
            else:
                if angle < angle_road_bottom:
                    self.behaviour = 'FINISH'

                else:
                    action_steer = 0
        
        elif self.behaviour == 'FINISH':
            action_acc = 4

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
            s = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            self.mud_pits = ran_cen_list
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                a = self.next_action(s)
                s_dash, r, terminate, reached_road, info_dict = simulator._step(a)
                fpsClock.tick(FPS)

                cur_time += 1
                s = s_dash

                if terminate:
                    road_status = reached_road
                    break

            self.behaviour = 'MOVE_AROUND_MUDPIT'

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
