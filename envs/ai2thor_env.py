import sys
import os
# 假设你的项目根目录是 /home/roxxi/CODE/dreamerv3-torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gym
import numpy as np
from gym import spaces
from thortils import launch_controller
import cv2
import time
from enum import Enum
import random
import copy
from PIL import Image


from ai2thor_utils.ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2thor_utils.ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                                              action_to_index, index_to_action, inverted_action_mapping,
                                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                                              room_this_point_belongs_to, get_rooms_ground_truth)

from thortils.utils.math import sep_spatial_sample
import thortils as tt
from thortils.map3d import Map3D, Mapper3D
from thortils.utils.visual import GridMapVisualizer, Visualizer2D
from thortils.agent import thor_reachable_positions
from thortils.scene import proper_convert_scene_to_grid_map
from thortils.vision import thor_topdown_img
from thortils.agent import thor_agent_pose, thor_pose_as_tuple
from thortils.controller import _resolve
from thortils.navigation import get_shortest_path_to_object, _round_pose
from thortils.utils import PriorityQueue, normalize_angles, euclidean_dist

class AI2ThorEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # initialize AI2-THOR controller
        self.controller = None
        self.config = config
        self.atu = AI2THORUtils()
        self.rnc = RobotNavigationControl()
        self.nu = NavigationUtils()
        self.mapper = None
        self.action_list = config.action_list  # ["MoveAhead", "RotateLeft", "RotateRight"]
        self.action_space = spaces.Discrete(len(self.action_list))
        # define observation space as a dictionary format, compatible with Dreamer3
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(config.size[1], config.size[0], 3), dtype=np.uint8
            ),
            "is_first": spaces.Box(low=False, high=True, shape=(1,), dtype=bool),
            "is_terminal": spaces.Box(low=False, high=True, shape=(1,), dtype=bool)
        })
        self.max_steps = getattr(config, "time_limit", 200)

        # new data
        self.habitat = self.atu.load_proctor_habitat(int(self.config.habitat_id)) 
        # print("ROXXI: habitat: ", self.habitat)
        self.reachable_positions = None # reachable positions in the habitat
        self.start_point = None # random start point (pos, rtn)
        self.cur_pos = None  # current position (pos, rtn) like ((2.0, 0.9009993672370911, 2.75), (-0.0, 135.0, 0.0))
        self.rooms_in_habitat = None # rooms in the habitat
        self.current_target_point = None # current target point (room center)
        self.initial_path_length = None 
        self.current_path_length = 1000 # distance to the target point
        self.initial_path = None
        self.current_path = None   # for visualization
        self.dreamer_path = None   # for visualization
        self.step_count = 0  # for reward function


    def set_random_start_position(self):
        sep = 1
        num_stops = 10
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]


        if self.controller is None:
            self.controller = launch_controller({"scene": self.habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
            self.rnc.set_controller(self.controller)
        else:
            self.controller.reset(self.controller.scene)
            self.rnc.reset_state()
        
        ## visualize current map
        # topdown_img = thor_topdown_img(self.controller)
        # img = Image.fromarray(topdown_img)
        # img.show()

        rnd = random.Random(int(time.time() * 100)) 

        self.reachable_positions = tt.thor_reachable_positions(self.controller)
        # print("ROXXI: reachable_positions: ", self.reachable_positions)
        # self.reachable_positions = self.update_reachable_positions()
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops,rnd=rnd)
        # print("ROXXI: placements: ", placements)

        # Choose one placement in the set of placements and then plan path from that placement to
        # the middle of the room. If planning path is not possible, then choose another one.
        path_planned = False
        while not path_planned:
            p = random.sample(placements, 1)[0]
            # print("ROXXI: p: ", p)
            # p = placements.pop()

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            # print("ROXXI: yaw: ", yaw)
            place_with_rtn = (p[0], p[1], yaw)
            #print("Placement: ", place_with_rtn)
            # print("ROXXI: place_with_rtn: ", place_with_rtn)
            self.rnc.teleport_to(place_with_rtn)

            self.start_point = self.rnc.get_agent_pos_and_rotation()
            # # print current position
            # pos = get_agent_pos_and_rotation(self)
            # print("ROXXI: current agent position: ", pos)
            # pose = thor_agent_pose(self.controller)
            # print("ROXXI: current agent pose: ", pose)



            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            point_for_room_search = (p[0], "", p[1])
            # In this habitat we have these rooms
            self.rooms_in_habitat = get_rooms_ground_truth(self.habitat)
            # print("ROXXI: rooms_in_habitat: ", self.rooms_in_habitat)
            room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
            # print("ROXXI: room_of_placement: ", room_of_placement)  
            # ROXXI: room_of_placement:  ('Bedroom', [(0.0, 1.778), (0.0, 5.333), (3.555, 5.333), (3.555, 0.0), (1.778, 0.0), (1.778, 1.778)], <POINT (1.955 3.022)>)
            room_centre = room_of_placement[2]
       

            self.current_target_point = room_centre
            # print("ROXXI: start_point: ", self.start_point )
            # print("ROXXI: current_target_point: ", self.current_target_point)

            try:
                self.cur_pos = self.rnc.get_agent_pos_and_rotation()
                self.initial_path_length = self.nu.get_path_cost_to_target_point(self.cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions)
                # update initial_path every reset, update when episode starts
                self.initial_path = self.nu.get_path_to_target_point(self.cur_pos,
                                                                     self.current_target_point,
                                                                     self.reachable_positions)
               
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                print(f"ERROR: {e}")
                continue

            self.current_path_length = self.initial_path_length
            path_planned = True


    def reset(self):
        self.set_random_start_position()  # 每个episode更新initial_path

        self.dreamer_path = []  # 每个episode开始时清空  for visualization
        self.step_count = 0
        
        event = self.controller.step(action="Pass")

        raw_image = event.cv2img  # shape: (H, W, 3), dtype: uint8, BGR channel
        target_size = (self.observation_space["image"].shape[1], self.observation_space["image"].shape[0])  # (width, height)
        if raw_image.shape[:2] != (target_size[1], target_size[0]):
            image = cv2.resize(raw_image, target_size, interpolation=cv2.INTER_AREA)
        else:
            image = raw_image
        # print("ROXXI: image shape: ", image.shape)
        obs = {
            "image": image
        }
        obs["is_first"] = True # if reset, is_first is True
        obs["is_terminal"] = False  # if reset, is_terminal is False
        return obs

    def step(self, action):
        action_idx = int(action)
        action_name = self.action_list[action_idx]
        info = {"target": False, "dreamer_path": self.dreamer_path, "initial_path": self.initial_path, "habitat_id": self.config.habitat_id}

        # event = self.controller.step(action=action_name)
        self.rnc.execute_action(action_name, degrees=45)
        event = self.controller.last_event

 
        self.step_count += 1
        reward, done, info = self._compute_reward_done_sparse(event, action_name, info, self.step_count)
        obs = self._get_obs(event)
        
        debug = 0
        if debug:
            print("----------------------------------------------------")
            print(f"ROXXI: action_name: {action_name}")
            print(f"ROXXI: current_path_length: {self.current_path_length}")
            print(f"ROXXI: initial_path_length: {self.initial_path_length}")
            # print("ROXXI: initial_path: ", self.initial_path)
            # print("ROXXI: current_path: ", self.current_path)
            # print(f"ROXXI: tar_pos: {[self.current_target_point.x, self.current_target_point.y]}")
            # print(f"ROXXI: cur_pos: {[self.cur_pos[0][0], self.cur_pos[0][2]]}")
            # ROXXI: tar_pos: [5.136, 6.848]
            # ROXXI: cur_pos: [1.310661792755127, 7.853559494018555]
            print(f"ROXXI: reward: {reward}")



        return obs, reward, done, info
        # obs: dict, reward: float, done: bool, info: dict
        # obs: {image, is_first, is_terminal}  all np.ndarray

    def _get_obs(self, event):
        raw_image = event.cv2img  # shape: (H, W, 3), dtype: uint8, BGR channel
        target_size = (self.observation_space["image"].shape[1], self.observation_space["image"].shape[0])  # (width, height)
        if raw_image.shape[:2] != (target_size[1], target_size[0]):
            image = cv2.resize(raw_image, target_size, interpolation=cv2.INTER_AREA)
        else:
            image = raw_image
        # print("ROXXI: image shape: ", image.shape)
        cur_xy = (self.nu.normalize_to_grid(self.cur_pos)[0][0], self.nu.normalize_to_grid(self.cur_pos)[0][2])
        self.dreamer_path.append(cur_xy)  # update dreamer_path every step
        # print("ROXXI: dreamer_path: ", self.dreamer_path)
        obs = {
            "image": image
        }
        obs["is_first"] = False  # if not reset, is_first is False
        obs["is_terminal"] = False  # if not done, is_terminal is False
        return obs

    def _compute_reward_done_sparse(self, event, action_name, info, step_count):
        reward = -1

        time_penalty_weight = 1.0      

        self.current_path_length = self.get_current_path_length()   

        # time_penalty = time_penalty_weight * np.log(1 + step_count) * -1
        # reward += time_penalty


        done = self.current_path_length <= 0.0
        # print(f"reward:{reward}, "
        #       f"done:{done}")
        if done:
            reward += 1000
            info["target"] = True
            print("ROXXI: We are in the center of the room!--------------------------------------------")
        return reward, done, info

    def _compute_reward_done_dense(self, event, action_name, info, step_count):
        """
        1. distance reward: y = 1 - x (x = current distance / initial distance)
        2. angle reward: y = cos(a) (a = current angle - target angle)
        3. time penalty: p1 = -time_penalty * step_count
        4. target reward
        """
        reward = 0.0
        distance_reward_weight = 3.0      
        angle_reward_weight = 1.5         
        time_penalty_weight = 1.0         
        
        # get current position and target position
        self.cur_pos = self.rnc.get_agent_pos_and_rotation()
        current_xy = (self.cur_pos[0][0], self.cur_pos[0][2])
        target_xy = (self.current_target_point.x, self.current_target_point.y)
        
        # 1. distance reward: y = 1 - x (x = current distance / initial distance)
        distance_reward = 0.0
        if self.initial_path_length == 0:  # already in the target point
            target_reached = True
        else:
            # use A* path length as distance measure
            # try:
            self.current_path_length = self.get_current_path_length()
            distance_ratio = self.current_path_length / self.initial_path_length
            distance_reward = 1.0 - distance_ratio  
            # except:
            # # if A* fails, use Euclidean distance
            # initial_distance = np.sqrt((self.start_point[0][0] - target_xy[0])**2 + 
            #                         (self.start_point[0][2] - target_xy[1])**2)
            # current_distance = np.sqrt((current_xy[0] - target_xy[0])**2 + (current_xy[1] - target_xy[1])**2)
            # distance_ratio = current_distance / initial_distance
            # distance_reward = 1.0 - distance_ratio
         
        reward += distance_reward * distance_reward_weight
        
        # 2. angle reward: y = cos(a) (a = current angle - target angle)
        current_yaw = self.cur_pos[1][1]
        target_yaw = np.arctan2(target_xy[1] - current_xy[1], target_xy[0] - current_xy[0])
        target_yaw = np.degrees(target_yaw)
        if target_yaw < 0:
            target_yaw += 360
        
        angle_diff = abs(current_yaw - target_yaw)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        angle_reward = np.cos(np.radians(angle_diff))
        reward += angle_reward * angle_reward_weight  # angle reward weight
        
        # 3. time penalty: p1 = -time_penalty * step_count
        # use logarithmic time penalty to avoid too much penalty
        time_penalty = time_penalty_weight * np.log(1 + step_count) * -1
        reward += time_penalty
        
        
        # 4. target reward
        target_reached = False
        if self.current_path_length <= 0.0:
            target_reached = True
        
        # handle termination
        done = False
        if target_reached:
            done = True
            reward += 1000.0  
            info["target"] = True
            # print("ROXXI: ------------------------------We are in the center of the room!---------------------------------")
        
        # debug information
        reward_debug = 0  # enable debug information
        # if step_count % 10 == 0 and reward_debug:  # print every 10 steps
        if done and reward_debug:
            status = "Target Reached" if target_reached else  ""
            print(f"----------------Step {step_count}-- {status}-------------------")
            print(f"distance_reward={distance_reward*distance_reward_weight:.3f}")
            print(f"angle_reward={angle_reward*angle_reward_weight:.3f}")
            print(f"time_penalty={time_penalty:.3f},")
            print(f"total_reward={reward:.3f}")
        
        # add reward components to info for external recording
        info["reward_components"] = {
            "distance_reward": distance_reward * distance_reward_weight,
            "angle_reward": angle_reward * angle_reward_weight,
            "time_penalty": time_penalty,
        }
        
        return reward, done, info


    def _compute_reward_done_collided(self, event, action_name, info, step_count):
        """
        new reward function design:
        1. distance reward: y = 1 - x (x = current distance / initial distance)
        2. angle reward: y = cos(a) (a = current angle - target angle)
        3. time penalty: p1 = -time_penalty * step_count
        4. forward bonus: +forward_bonus
        !!! if use this collided setting, please set reward_set = "collided" in tools.py.  (to be modified) ！！！
        """
        # initialize reward
        reward = 0.0
        distance_reward_weight = 3.0      
        angle_reward_weight = 1.5         
        time_penalty_weight = 1.0         
        forward_bonus_weight = 0.5        
        
        # get current position and target position
        self.cur_pos = self.rnc.get_agent_pos_and_rotation()
        current_xy = (self.cur_pos[0][0], self.cur_pos[0][2])
        target_xy = (self.current_target_point.x, self.current_target_point.y)
        
        # 1. distance reward: y = 1 - x (x = current distance / initial distance)
        distance_reward = 0.0
        if self.initial_path_length == 0:  # already in the target point
            target_reached = True
        else:
            # use A* path length as distance measure
            # try:
            self.current_path_length = self.get_current_path_length()
            distance_ratio = self.current_path_length / self.initial_path_length
            distance_reward = 1.0 - distance_ratio  
            # except:
            # if A* fails, use Euclidean distance
            # initial_distance = np.sqrt((self.start_point[0][0] - target_xy[0])**2 + 
            #                         (self.start_point[0][2] - target_xy[1])**2)
            # current_distance = np.sqrt((current_xy[0] - target_xy[0])**2 + (current_xy[1] - target_xy[1])**2)
            # distance_ratio = current_distance / initial_distance
            # distance_reward = 1.0 - distance_ratio
    
        reward += distance_reward * distance_reward_weight 
        
        # 2. angle reward: y = cos(a) (a = current angle - target angle)
        current_yaw = self.cur_pos[1][1]
        target_yaw = np.arctan2(target_xy[1] - current_xy[1], target_xy[0] - current_xy[0])
        target_yaw = np.degrees(target_yaw)
        if target_yaw < 0:
            target_yaw += 360
        
        angle_diff = abs(current_yaw - target_yaw)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        angle_reward = np.cos(np.radians(angle_diff))
        reward += angle_reward * angle_reward_weight  # angle reward weight
        
        # 3. time penalty: p1 = -time_penalty * step_count
        # use logarithmic time penalty to avoid too much penalty
        time_penalty = time_penalty_weight * np.log(1 + step_count) * -1
        reward += time_penalty

        # # 4. still penalty: if the position does not change for a period of time, then give penalty
        # # 记录agent最近的若干步的位置
        # if not hasattr(self, 'position_history'):
        #     self.position_history = []
            # # record current position (only x, z coordinates, ignore y height and orientation)
        # current_pos_2d = (round(self.cur_pos[0][0], 2), round(self.cur_pos[0][2], 2))
        # self.position_history.append(current_pos_2d)
        # # only keep the last N steps' positions
        # N = 3
        # if len(self.position_history) > N:
        #     self.position_history.pop(0)
        # # check if the position has changed for the last N steps (threshold can be adjusted)
        # still_count = 0
        # if len(self.position_history) == N:
        #     # calculate the distance between all positions and the first position
        #     base_pos = self.position_history[0]
        #     threshold = 0.05  # 5cm以内算作没动
        #     if all(np.linalg.norm(np.array(pos) - np.array(base_pos)) < threshold for pos in self.position_history):
        #         still_count = 1
        # # still penalty: if the position does not change for a period of time, then give penalty
        # still_penalty = (still_penalty_weight * still_count) + still_count * (time_penalty / time_penalty_weight)
        # reward -= still_penalty
        
        # 5. forward bonus
        if action_name == "MoveAhead":
            reward += forward_bonus_weight * 1# forward bonus
        
        # 6. 碰撞惩罚
        collided = False
        if not event.metadata["lastActionSuccess"]:  # if the move fails, it is usually because of collision
            # print("ROXXI: agent collided with something!")
            collided = True
        # # use raycast to check if there is a wall in front
        #     query = self.controller.step(
        #         action="GetCoordinateFromRaycast",
        #         x=0.5,  # raycast from the center of the agent pov
        #         y=0.5
        #     )
        #     if query.metadata["actionReturn"] is not None:  
        #         print("Agent 撞墙了！")


        # 6. target reward
        target_reached = False
        if self.current_path_length <= 0.0:
            target_reached = True
        
        # handle termination
        done = False
        if collided:
            done = True
            reward -= 50.0   # reduce collision penalty
            info["collided"] = True
            # print("ROXXI: ------------------------------Agent collided with something!---------------------------------")
        if target_reached:
            done = True
            reward += 200.0  # reduce target reward, but still keep large
            info["target"] = True
            # print("ROXXI: ------------------------------We are in the center of the room!---------------------------------")
        
        # debug information
        reward_debug = 0  # enable debug information
        # if step_count % 10 == 0 and reward_debug:  # print every 10 steps
        if done and reward_debug:
            status = "Target Reached" if target_reached else ("Collided" if collided else "")
            print(f"----------------Step {step_count}-- {status}-------------------")
            print(f"distance_reward={distance_reward*distance_reward_weight:.3f}")
            print(f"angle_reward={angle_reward*angle_reward_weight:.3f}")
            print(f"time_penalty={time_penalty:.3f},")
            print(f"forward_bonus={forward_bonus_weight if action_name=='MoveAhead' else 0:.3f}")
            print(f"total_reward={reward:.3f}")
        
            # add reward components to info for external recording
        info["reward_components"] = {
            "distance_reward": distance_reward * distance_reward_weight,
            "angle_reward": angle_reward * angle_reward_weight,
            "time_penalty": time_penalty,
            "forward_bonus": forward_bonus_weight if action_name == "MoveAhead" else 0.0,
            "total_reward": reward
        }
        
        return reward, done, info






#-----------------------------------------------------------------------------------------------------------------------
    # This function will calculate path length to the desired point from the current position.
    def get_current_path_length(self):
        try:
            self.cur_pos = self.rnc.get_agent_pos_and_rotation()
            self.current_path_length = self.nu.get_path_cost_to_target_point(self.cur_pos,
                                                                             self.current_target_point,
                                                                             self.reachable_positions)
            self.current_path = self.nu.get_path_to_target_point(self.cur_pos,
                                                                     self.current_target_point,
                                                                     self.reachable_positions)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Using previous current_path_length: ", self.current_path_length)
            self._bad_spot = True

        return self.current_path_length

    # draw topdown frame
    def get_top_down_frame(self, dreamer_path, initial_path, floor_cut=0.1):
        topdown_img = thor_topdown_img(self.controller)
        img = Image.fromarray(topdown_img)
        # img.show()
        img_original = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
        
        if self.mapper is None:
            self.mapper = Mapper3D(self.controller)
        else:
            self.mapper.update(self.controller.last_event)

        self.mapper.automate(num_stops=20, sep=1.5)
        grid_map = self.mapper.get_grid_map(floor_cut=floor_cut, debug=False)



        dreamer_path_grid = []
        for pos in dreamer_path:
            dreamer_path_grid.append(grid_map.to_grid_pos(*pos))
        initial_path_grid = []
        for pos in initial_path:
            initial_path_grid.append(grid_map.to_grid_pos(*pos))

        # print("ROXXI: dreamer_path_grid: ", dreamer_path_grid)
        # print("ROXXI: initial_path_grid: ", initial_path_grid)
        
        # show grid map
        viz = GridMapVisualizer(grid_map=grid_map, res=30)
        img = viz.render()
       
        # 画出dreamer_path
        img = viz.highlight(img, dreamer_path_grid,
                            color=(255, 0, 255), show_progress=True)  # purple
        # 画出initial_path
        img = viz.highlight(img, initial_path_grid,
                            color=(0, 255, 255), show_progress=True)  # blue/yellow
        # 画出目标位置
        img = viz.highlight(img, [grid_map.to_grid_pos(self.current_target_point.x, self.current_target_point.y)],
                            color=(0, 0, 255), show_progress=True)  # red
        # 画出初始位置
        img = viz.highlight(img, [grid_map.to_grid_pos(self.start_point[0][0], self.start_point[0][2])],
                            color=(255, 255, 0), show_progress=True)  # yellow
        # viz.show_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img, img_original



#-----------------------------------------------------------------------------------------------------------------------
    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value


