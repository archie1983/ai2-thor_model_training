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
        self.mapper = None
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

        # 新增的数据
        self.habitat = self.atu.load_proctor_habitat(int(self.config.habitat_id)) # what is habitat means?
        # print("ROXXI: habitat: ", self.habitat)
        self.reachable_positions = None # 该habitat的可达位置列表
        self.start_point = None # 随机选取的起点（pos， rtn）
        self.cur_pos = None  #  当前位置（pos， rtn）如 ((2.0, 0.9009993672370911, 2.75), (-0.0, 135.0, 0.0))
        self.rooms_in_habitat = None # 该habitat的房间列表
        self.current_target_point = None # 当前目标点（房间中心）
        self.initial_path_length = None 
        self.current_path_length = 1000 # 当前里终点的距离
        self.initial_path = None
        self.current_path = None 
        self.dreamer_path = None


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
        
        # 可视化当前地图
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
                # 每个reset才更新一次，episode重新开始才更新
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
        self.set_random_start_position()

        event = self.controller.step(action="Pass")
        self.dreamer_path = []  # 每个episode开始时清空

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
        obs["dreamer_path"] = self.dreamer_path
        obs['initial_path'] = self.initial_path
        return obs

    def step(self, action):
        action_idx = int(action)
        action_name = self.action_list[action_idx]
        info = {"target": False}

        # event = self.controller.step(action=action_name)
        self.rnc.execute_action(action_name)
        event = self.controller.last_event
        # self.get_top_down_frame()

        reward, done, info = self._compute_reward_done(event, action_name, info)
        obs = self._get_obs(event)
        


        debug = 0
        if debug:
            print("----------------------------------------------------")
            print(f"ROXXI: current_path_length: {self.current_path_length}")
            print(f"ROXXI: initial_path_length: {self.initial_path_length}")
            print("ROXXI: initial_path: ", self.initial_path)
            print("ROXXI: current_path: ", self.current_path)
            print(f"ROXXI: tar_pos: {[self.current_target_point.x, self.current_target_point.y]}")
            print(f"ROXXI: cur_pos: {[self.cur_pos[0][0], self.cur_pos[0][2]]}")
            # print(f"ROXXI: start_point: {self.start_point}")
            print(f"ROXXI: reward: {reward}")



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
        cur_xy = (self.nu.normalize_to_grid(self.cur_pos)[0][0], self.nu.normalize_to_grid(self.cur_pos)[0][2])
        self.dreamer_path.append(cur_xy)  # 每个step更新一次dreamer_path
        # print("ROXXI: dreamer_path: ", self.dreamer_path)
        obs = {
            "image": image
        }
        obs["is_first"] = False  # if not reset, is_first is False
        obs["is_terminal"] = False  # if not done, is_terminal is False
        obs["dreamer_path"] = self.dreamer_path
        obs['initial_path'] = self.initial_path
        return obs

    def _compute_reward_done(self, event, action_name, info):
        # print(f"Initial length:{self.initial_path_length}, "
        #       f"current length:{self.get_current_path_length()}, "
        #       f"length difff:{self.initial_path_length - self.get_current_path_length()}, ")

        if self.initial_path_length == 0:  #  initial position is the target point
            reward = 1.0  # biggest reward
        else:
            self.current_path_length = self.get_current_path_length()
            reward = (self.initial_path_length - self.current_path_length) / self.initial_path_length
        if action_name == "MoveAhead":
            reward += 1


        done = self.current_path_length <= 0.0

        # print(f"reward:{reward}, "
        #       f"done:{done}")
        if done:
            info["target"] = True
            print("ROXXI: We are in the center of the room!--------------------------------------------")
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

    def get_top_down_frame(self, dreamer_path, initial_path, floor_cut=0.1):
        topdown_img = thor_topdown_img(self.controller)
        img = Image.fromarray(topdown_img)
        img.show()
        
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

        print("ROXXI: dreamer_path_grid: ", dreamer_path_grid)
        print("ROXXI: initial_path_grid: ", initial_path_grid)
        
        # show grid map
        viz = GridMapVisualizer(grid_map=grid_map, res=30)
        img = viz.render()
        # 画出dreamer_path
        img = viz.highlight(img, dreamer_path_grid,
                            color=(255, 0, 255), show_progress=True)  # purple
        # 画出initial_path
        img = viz.highlight(img, initial_path_grid,
                            color=(0, 255, 255), show_progress=True)  # blue
        # 画出目标位置
        img = viz.highlight(img, [grid_map.to_grid_pos(self.current_target_point.x, self.current_target_point.y)],
                            color=(255, 255, 0), show_progress=True)  # yellow
  
        viz.show_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 1)  # flip horizontally
        return img



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
    import numpy as np
    import yaml

    def load_config_from_yaml(yaml_path, key='ai2thor'):
        with open(yaml_path, 'r') as f:
            configs = yaml.safe_load(f)
        # 合并 defaults 和 key 部分
        config = configs.get('defaults', {}).copy()
        config.update(configs.get(key, {}))
        return config


    # 读取配置
    config_dict = load_config_from_yaml("./configs.yaml", key='ai2thor')
    # print("加载的配置：", config_dict)

    # 你可以用 types.SimpleNamespace 或自定义 Config 类来转成属性访问
    from types import SimpleNamespace
    config = SimpleNamespace(**config_dict)

    env = AI2ThorEnv(config)

    print("AI2-THOR Controller参数：------------------------------")
    print("Scene:", getattr(config, "scene", None))
    print("Action List:", getattr(config, "action_list", None))
    print("Image Size:", getattr(config, "size", None))
    print("Max Steps:", getattr(config, "time_limit", None))
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    obs = env.reset()
    print("初始观测obs keys:", obs.keys())
    print("初始观测image shape:", obs["image"].shape)

    # 获取并显示top-down视角
    top_down_img = env.get_top_down_frame()
    top_down_img.show(title="Top-Down View (Reset)")

    for i in range(5):
        action = env.action_space.sample()
        action_idx = np.argmax(action)
        print(f"\nStep {i+1}: 执行动作 {config.action_list[action_idx]}")
        obs, reward, done, info = env.step(action)
        print("观测image shape:", obs["image"].shape)
        print("reward:", reward, "done:", done)

        # 获取并显示top-down视角
        top_down_img = env.get_top_down_frame()
        top_down_img.show(title=f"Top-Down View (Step {i+1})")

        if done:
            print("Episode done, 重置环境")
            obs = env.reset()
            top_down_img = env.get_top_down_frame()
            top_down_img.show(title="Top-Down View (Reset)")