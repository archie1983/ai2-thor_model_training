import gym
import numpy as np
from gym import spaces
from thortils import launch_controller
import cv2
import time
from enum import Enum
import random

from ai2thor_utils.ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2thor_utils.ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                                              action_to_index, index_to_action, inverted_action_mapping,
                                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                                              room_this_point_belongs_to, get_rooms_ground_truth)

from thortils.utils.math import sep_spatial_sample
import thortils as tt

from thortils.agent import thor_agent_pose, thor_pose_as_tuple
from thortils.controller import _resolve
from thortils.navigation import get_shortest_path_to_object, _round_pose
# TODO: "get_shortest_path_to_object" is A* path?
from thortils.utils import PriorityQueue, normalize_angles, euclidean_dist

# TODO: 1. one step is one action?  multiple steps IN ONE ROOM with same start point is one episode?
# TODO: 2. reset when episode done?
# TODO: 3. How to know what function I can use?
# TODO: 4. GPU Memory cose?

class AI2ThorEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 初始化 AI2-THOR 控制器
        self.controller = None
        self.config = config
        self.atu = AI2THORUtils()
        self.rnc = RobotNavigationControl()
        self.nu = NavigationUtils()

        self.action_list = config.action_list  # ["MoveAhead", "RotateLeft", "RotateRight"]
        self.action_space = spaces.Discrete(len(self.action_list))
        # 定义观测空间为字典格式，与Dreamer3兼容
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(config.size[1], config.size[0], 3), dtype=np.uint8
            ),
            "is_first": spaces.Box(low=False, high=True, shape=(1,), dtype=bool),
            "is_terminal": spaces.Box(low=False, high=True, shape=(1,), dtype=bool)
        })
        self.max_steps = getattr(config, "time_limit", 200)
        self.current_step = 0

        # 新增的数据
        self.habitat = self.atu.load_proctor_habitat(int(self.config.habitat_id)) # what is habitat means?
        # print("ROXXI: habitat: ", self.habitat)
        self.reachable_positions = None # 该habitat的可达位置列表
        self.start_point = None # 随机选取的起点（pos， rtn）
        self.rooms_in_habitat = None # 该habitat的房间列表
        self.current_target_point = None # 当前目标点（房间中心）
        self.initial_path_length = None # TODO A* path length?
        self.current_path_length = 1000 # 当前里终点的距离


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

            self.start_point = self.get_agent_pos_and_rotation()
            # # 打印当前位置
            # pos = get_agent_pos_and_rotation(self)
            # print("ROXXI: 当前agent位置: ", pos)
            # pose = thor_agent_pose(self.controller)
            # print("ROXXI: 当前agent姿态: ", pose)



            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            point_for_room_search = (p[0], "", p[1])
            # In this habitat we have these rooms
            self.rooms_in_habitat = get_rooms_ground_truth(self.habitat)
            # print("ROXXI: rooms_in_habitat: ", self.rooms_in_habitat)
            room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
            # print("ROXXI: room_of_placement: ", room_of_placement)
            room_centre = room_of_placement[2]
       

            self.current_target_point = room_centre
            # print("ROXXI: start_point: ", self.start_point )
            # print("ROXXI: current_target_point: ", self.current_target_point)

            try:
                cur_pos = self.rnc.get_agent_pos_and_rotation()
                # self.initial_path_length = self.get_path_cost_to_target_point(self.start_point, self.current_target_point)
                self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions)
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                print(f"ERROR: {e}")
                continue

            self.current_path_length = self.initial_path_length
            path_planned = True


    def reset(self):
        self.set_random_start_position()
        # print("ROXXI: finish set random start position")

        # if self.controller is None:
        #     self.controller = launch_controller({"scene": self.habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
        # else:
        #     self.controller.reset(self.controller.scene)

        self.current_step = 0
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
        
        # event = self.controller.step(action=action_name)
        self.rnc.execute_action(action_name)
        event = self.controller.last_event

        reward, done = self._compute_reward_done(event)
        obs = self._get_obs(event)
        self.current_step += 1
        info = {}

        # print("ROXXI: obs:", obs)
        return obs, reward, done, info
        # obs: dict, reward: float, done: bool, info: dict
        # obs: {image, is_first, is_terminal}  all np.ndarray

    def _get_obs(self, event):
        # 返回 obs 字典，Dreamer3 兼容        
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
        obs["is_first"] = False  # if not reset, is_first is False
        obs["is_terminal"] = False  # if not done, is_terminal is False
        return obs

    def _compute_reward_done(self, event):
        # print(f"Initial length:{self.initial_path_length}, "
        #       f"current length:{self.get_current_path_length()}, "
        #       f"length difff:{self.initial_path_length - self.get_current_path_length()}, ")

        if self.initial_path_length is 0:  #  initial position is the target point
            reward = 1.0  # biggest reward
        else:
            reward = (self.initial_path_length - self.get_current_path_length()) / self.initial_path_length
        done = self.current_path_length <= 0.0

        # print(f"reward:{reward}, "
        #       f"done:{done}")
        if done:
            print("ROXXI: We are in the center of the room!--------------------------------------------")
        return reward, done

#-----------------------------------------------------------------------------------------------------------------------
    # This function will calculate path length to the desired point from the current position.
    def get_current_path_length(self):
        try:
            cur_pos = self.rnc.get_agent_pos_and_rotation()
            self.current_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                             self.current_target_point,
                                                                             self.reachable_positions)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Using previous current_path_length: ", self.current_path_length)
            self._bad_spot = True

        return self.current_path_length

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

if __name__ == "__main__":
    import argparse

    class DummyConfig:
        # 你可以根据实际情况修改这些参数
        size = (64, 64)
        scene = "FloorPlan1"
        action_list = ["MoveAhead", "RotateLeft", "RotateRight"]
        time_limit = 10

    config = DummyConfig()
    env = AI2ThorEnv(config)

    print("AI2-THOR Controller参数：")
    print("Scene:", config.scene)
    print("Action List:", config.action_list)
    print("Image Size:", config.size)
    print("Max Steps:", config.time_limit)
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    obs = env.reset()
    print("初始观测obs keys:", obs.keys())
    print("初始观测image shape:", obs["image"].shape)

    for i in range(5):
        action = env.action_space.sample()
        action_idx = np.argmax(action)
        print(f"\nStep {i+1}: 执行动作 {config.action_list[action_idx]}")
        obs, reward, done, info = env.step(action)
        print("观测image shape:", obs["image"].shape)
        print("reward:", reward, "done:", done)
        if done:
            print("Episode done, 重置环境")
            obs = env.reset()