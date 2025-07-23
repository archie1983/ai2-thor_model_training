import gym
import numpy as np
from gym import spaces
from thortils import launch_controller
import cv2
import random
from enum import Enum
from ai2_thor_model_training.training_data_extraction import (
    RobotNavigationControl, action_mapping, action_to_index, 
    index_to_action, inverted_action_mapping, AI2THORUtils, 
    get_path_length, get_centre_of_the_room, room_this_point_belongs_to, 
    get_rooms_ground_truth
)
from thortils.utils.math import sep_spatial_sample
import thortils as tt
from thortils.agent import thor_agent_pose, thor_pose_as_tuple
from thortils.controller import _resolve
from thortils.navigation import get_shortest_path_to_object, get_navigation_actions, _round_pose, _same_pose, transform_pose, _valid_pose, _cost



from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany, PriorityQueue, normalize_angles, euclidean_dist

class AI2ThorEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 初始化 AI2-THOR 控制器
        # self.controller = launch_controller({"scene": config.scene, "VISIBILITY_DISTANCE": 3.0, "headless": False})

        self.rnc = RobotNavigationControl()
        self.controller = None
        self.atu = AI2THORUtils()
        self.rooms_in_habitat = None
        self.current_path_length = 1000
        self.initial_path_length = 0
        self.current_target_point = None
         # If we get into a bad spot from which for whatever reason we can't plan a path out, then we'll set this to
        # True and based on it will teleport to a new place when we see this set.
        self._bad_spot = False

        # when we select a random position and plan path to the room centre, we will assign a value to this parameter
        # with the A* path length from that random position to the desired point. This will help calculate reward from all
        # further points.
        self.initial_path_length = 0

        # We will need to keep track of the target point that we want to reach because we will be re-planning path to it
        # from all sorts of different points.
        self.current_target_point = None

        self.habitat_id = 420

        # # Dreamer stuff
        # self._size = size
        # self._repeat = repeat





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

    ##
    # Load the given habitat- load it, and put agent in a random place
    ##
    def load_habitat(self, habitat_id):
        # load required habitat
        # print("ROXXI: haba: ", habitat_id)
        habitat = self.atu.load_proctor_habitat(int(habitat_id))

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat,
                                                 "VISIBILITY_DISTANCE": 3.0,
                                                 "headless": False,
                                                 "IMAGE_WIDTH": 64,
                                                 "IMAGE_HEIGHT": 64
                                                 # "RENDER_DEPTH": False,
                                                 # "RENDER_INSTANCE_SEGMENTATION": False,
                                                 # "RENDER_IMAGE": True
                                                 # "IMAGE_WIDTH": 64,
                                                 # "IMAGE_HEIGHT": 64
                                                 })
            self.rnc.set_controller(
                self.controller)  # This allows our control scripts to interact with AI2-THOR environment
        else:
            self.controller.reset(habitat)
            # self.reset_state()
            self.rnc.reset_state()
            # self.rnc.set_controller(self.controller)

        # In this habitat we have these rooms
        self.rooms_in_habitat = get_rooms_ground_truth(habitat)

        # Take a snapshot of all available positions- these won't change while we're in this habitat,
        # so no need to re-do them everytime we plan a path.
        self.grid_size = self.controller.initialization_parameters["gridSize"]
        self.reachable_positions = self.update_reachable_positions()

        # Now place the robot in a random position and figure out the target from there.
        self.choose_random_placement_in_habitat()

    # Here we will select a number of random placements and then choose one to navigate from it
    # to some goal.
    def choose_random_placement_in_habitat(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 20
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have

        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        rnd = random.Random(seed)

        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        # reachable_positions = tt.thor_reachable_positions(self.controller)
        # self.reachable_positions
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops,
                                        rnd=rnd)

        # print(placements)

        # Choose one placement in the set of placements and then plan path from that placement to
        # the middle of the room. If planning path is not possible, then choose another one.
        path_planned = False
        while not path_planned:
            p = random.choice(list(placements)) 

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            #print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            point_for_room_search = (p[0], "", p[1])
            room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
            room_centre = room_of_placement[2]

            # Now plan path to the centre of the room
            #            try:
            #                path_and_plan = self.get_path_to_target_point(room_centre)
            #            except ValueError as e:
            #                # If the path could not be planned, then drop it and carry on with the next one
            #                print(f"ERROR: {e}")
            #                continue

            self.current_target_point = room_centre
            try:
                self.initial_path_length = self.get_path_cost_to_target_point(self.current_target_point)
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                print(f"ERROR: {e}")
                continue

            # print("PATH & PLAN: ", path_and_plan)
            #            path = path_and_plan[0]
            #            plan = path_and_plan[1]
            # self.prev_pose = thor_agent_pose(self.controller)  # This is where we are before the plan started
            # place_with_rtn
            # thor_pose_as_tuple(self.prev_pose)
            #print("AE poses: place_with_rtn: ", place_with_rtn, " p: ", p, " self.rnc.get_agent_pos_and_rotation(): ",
            #      self.rnc.get_agent_pos_and_rotation())
            #            cur_pos = self.rnc.get_agent_pos_and_rotation()
            #            self.initial_path_length = get_path_length(path, cur_pos)

            # at this point current path length is the initial path length. We will re-calculate current path length
            # many times and reward will be calculated using it.
            self.current_path_length = self.initial_path_length

            path_planned = True

    def get_path_cost_to_target_point(self, target_point):
        # where we start
        start_point = self.rnc.get_agent_pos_and_rotation()  # (start_position, start_rotation)
        # where we want to get to
        destination = ((target_point.x, 0.9009993672370911, target_point.y),
                       start_point[1])  # the defined 2D coordinates and same rotation as start position
        # Normalize angles in start and goal to be within 0 to 360 (see top comments)
        # Also, round the poses so that we don't have irrational numbers in them that would be hard to look up
        # e.g. (10.75, 8.25) instead of (10.86666666, 8.3333333333).
        # Also angles need to be discrete values in [0, 45, 90, 135, 180, 225, 270, 315]
        start_point = _round_pose((start_point[0], normalize_angles(start_point[1])))
        destination = _round_pose((destination[0], normalize_angles(destination[1])))
        start_point = self.normalize_to_grid(start_point)
        destination = self.normalize_to_grid(destination)
        # positions that we can reach
        reachable_positions = set(self.reachable_positions)
        # The priority queue. We will keep poses in it with the estimates of their distances to the goal stored as priorities.
        worklist = PriorityQueue()
        # First pose will be the start and the estimate to the goal is its priority.
        # Make sure that all poses in the worklist and elsewhere are rounded though,
        # else they may not match later when we look them up and lead to no path found.
        start_node = self.AStarNode(start_point, h=euclidean_dist(start_point[0], destination[0]))
        # start_node.f doesn't take necessary turns into account, but as a heuristic it is admissible
        worklist.push(start_node, start_node.f)

        # cost[n] is the cost of the cheapest path from start to n currently known, where n is the pose
        cost = {}
        # Obviously from start to start the cost is 0
        cost[start_node.get_ai2thor_pose_and_rtn()] = 0
        # keep track of visited poses
        visited = set()

        # AE: Start the A* exploration. Obviously at first we will have the start node there with the estimate to the goal.
        while not worklist.isEmpty():
            current_node = worklist.pop()
            # If we've already visited this pose, then we can skip it and look at the next one
            if current_node in visited:
                continue
            # AE: If we're close enough to the end, then stop exploration and work backwards to reconstruct plan or
            # estimate path cost.
            if euclidean_dist(destination[0], current_node.get_ai2thor_pose()) <= 0.5:
                return cost[current_node.get_ai2thor_pose_and_rtn()]

            # AE: Look at all defined actions and try each of them from the current pose and see what happens
            for action in self.NavigationAction:
                nx, ny = action.apply(current_node.x, current_node.y)
                # If we end up in a legal place, then generate a new node and add it to the priority queue
                if (nx, ny) in reachable_positions:
                    # generate a new node from this action. There may be different nodes for the same location
                    # on the grid because they may have different yaw rotations and even different parents.
                    # So in the worst case there can be a node for <each grid location> * <all possible yaw rotations> * <each grid location as a parent>
                    next_node = action.apply_to_node(current_node, destination)
                    # The new nodes cost (the g value) has already been computed when it was generated, we can add it to the
                    # cost dictionary for this position and rotation if it's not already there.
                    #
                    # AE: Now we check if this pose, that we get with the chosen action, already exists in the cost set.
                    # If it does, then we'll get some number from cost.get(next_pose, float("inf"), otherwise we'll get
                    # infinity. If we got some number, but our new calculation is better than the old one, then we update
                    # the cost set with the new cost for the given pose.
                    if next_node.g < cost.get(next_node.get_ai2thor_pose_and_rtn(), float("inf")):
                        # AE: update the cost for this pose that we achieve from old pose with the selected action
                        cost[next_node.get_ai2thor_pose_and_rtn()] = next_node.g
                        # AE: push the newly discovered pose to our priority queue, giving the priority of its cost + euclidean
                        # distance from it to the goal as a heuristic (underestimate of the cost of the rest of the path).
                        worklist.push(next_node, next_node.f)
                        # AE: Keep track of where we came from so that we can reconstruct plan
                        # No need here, because each node contains a link to its parent
                        # comefrom[next_pose] = (current_pose, action)

            visited.add(current_node)

        # AE: If we're here, then that means, we could not find a path.
        # AE: print warning and return something.
        raise ValueError("Plan not found from {} to {}".format(start_point, destination))
        # return float("inf")


    def reset(self):
        self.load_habitat(self.habitat_id)
        event = self.controller.last_event

        obs = self._get_obs(event)
        return obs

    def step(self, action):
        action_idx = int(action)
        action_name = self.action_list[action_idx]
        
        # event = self.controller.step(action=action_name)
        self.rnc.execute_action(action_name)
        event = self.controller.last_event

        obs = self._get_obs(event)
        reward, done = self._compute_reward_done(event)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        info = {}
        return obs, reward, done, info

    def _get_obs(self, event):
        # 返回 obs 字典，Dreamer3 兼容        
        raw_image = event.cv2img  # shape: (H, W, 3), dtype: uint8, BGR channel
        target_size = (self.observation_space["image"].shape[1], self.observation_space["image"].shape[0])  # (width, height)
        # print("ROXXI: raw_image.shape[:2]: ", raw_image.shape[:2], " target_size: ", target_size)
        if raw_image.shape[:2] != (target_size[1], target_size[0]):
            image = cv2.resize(raw_image, target_size, interpolation=cv2.INTER_AREA)
        else:
            image = raw_image
        
        obs = {
            "image": image
        }
        # Dreamer3 需要 is_first/is_terminal 字段
        obs["is_first"] = np.array([self.current_step == 0], dtype=bool)
        obs["is_terminal"] = np.array([False], dtype=bool)  # 终止时在 step 里处理
        return obs

    def _compute_reward_done(self, event):
        reward = (self.initial_path_length - self.get_current_path_length()) / self.initial_path_length
        done = self.current_path_length <= 0.0
        return reward, done


    # This function will calculate path length to the desired point from the current position.
    def get_current_path_length(self):
        try:
            self.current_path_length = self.get_path_cost_to_target_point(self.current_target_point)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Using previous current_path_length: ", self.current_path_length)
            self._bad_spot = True

        return self.current_path_length


     # Get all reachable positions and store them in a variable.
    
    def update_reachable_positions(self):
        reachable_positions = [
            tuple(map(lambda x: roundany(x, self.grid_size), pos))
            for pos in thor_reachable_positions(self.controller)]
        # print(reachable_positions, self.grid_size)
        return reachable_positions

    def normalize_to_grid(self, pose, step=0.25):
        def round_to_step(val):
            return round(val / step) * step

        def normalize_yaw(yaw):
            """
            Normalize yaw angle to the nearest multiple of 45 in [0, 315].
            Input yaw can be any real number (positive or negative).
            """
            yaw = yaw % 360  # Bring into [0, 360)
            return round(yaw / 45) * 45 % 360

        location = pose[0]
        rotation = pose[1]
        return ((round_to_step(location[0]), round_to_step(location[1]), round_to_step(location[2])),
                (0.0, normalize_yaw(rotation[1]), 0.0))
        
    class AStarNode:
        def __init__(self, pose, g=0, h=0, parent=None):
            self.x = pose[0][0]
            self.y = pose[0][2]
            self.yaw = pose[1][1]
            self.g = g  # Cost from start to current node
            self.h = h  # Heuristic cost estimate to goal
            self.f = g + h  # Total cost
            self.parent = parent

        def update_heuristic(self, new_h):
            self.h = new_h
            self.f = self.h + self.g

        def update_real_cost(self, new_g):
            self.g = new_g
            self.f = self.h + self.g

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

        def __hash__(self):
            return hash((self.x, self.y))

        def get_xy(self):
            return (self.x, self.y)

        def get_ai2thor_pose(self):
            return (self.x, 0.9009993672370911, self.y)

        def get_ai2thor_pose_and_rtn(self):
            return ((self.x, 0.9009993672370911, self.y), (0, self.yaw, 0))
    
    class NavigationAction(Enum):
        NORTH = (0, -0.25, 0)
        SOUTH = (0, 0.25, 180)
        EAST = (0.25, 0, 90)
        WEST = (-0.25, 0, 270)
        NORTHEAST = (0.25, -0.25, 45)
        NORTHWEST = (-0.25, -0.25, 315)
        SOUTHEAST = (0.25, 0.25, 135)
        SOUTHWEST = (-0.25, 0.25, 225)

        def __init__(self, dx, dy, yaw):
            self._dx = dx
            self._dy = dy
            self._yaw = yaw

        @property
        def dx(self):
            return self._dx

        @property
        def dy(self):
            return self._dy

        @property
        def new_yaw(self):
            return self._yaw

        # Apply the action to some X and Y coordinates.
        # This will allow us to test if the new X and Y is within reachable positions
        def apply(self, x, y):
            return x + self._dx, y + self._dy

        # Apply the action to a node. This will allow us to create a new node from a previous node
        # including X and Y coordinates and also the Yaw rotation.
        def apply_to_node(self, node, destination):
            full_pose = node.get_ai2thor_pose_and_rtn()
            # TODO: Are we updating numerical values here or the fields of the other node?
            # print("AE: full_pose[0][0]: ", full_pose[0][0], node.get_ai2thor_pose())
            new_x = full_pose[0][0] + self._dx
            new_y = full_pose[0][2] + self._dy
            # print("AE: full_pose[0][0]: ", full_pose[0][0], node.get_ai2thor_pose())

            # Cost will always be 1 for the movement ahead and some value to account for the required turns
            # And we want to calculate it before we update the yaw for the full pose which will form the new
            # node.
            turning_deg_required = abs(full_pose[1][1] - self._yaw)
            # If more than 180, then turn the other way
            if turning_deg_required > 180:
                turning_deg_required -= 180
            # We turn in 45 degree increments, so this many turns we will need
            turns_required = turning_deg_required // 45

            # Cost will always be 1 for the movement ahead and some value to account for the required turns
            new_cost = 1 + turns_required
            # now that new cost has been calculated, we can assign the new yaw to the pose fields.
            new_yaw = self._yaw
            new_full_pose = ((new_x, full_pose[0][1], new_y), (0.0, new_yaw, 0.0))
            new_node = AI2ThorEnv.AStarNode(new_full_pose, node.g + new_cost,
                                            euclidean_dist(new_full_pose[0], destination[0]), node)

            return new_node
    
    
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
    print("obs keys:", obs.keys())
    print("image shape:", obs["image"].shape, "dtype:", obs["image"].dtype)
    print("is_first:", obs.get("is_first"), "is_terminal:", obs.get("is_terminal"))

    for i in range(5):
        action = env.action_space.sample()
        action_idx = np.argmax(action)
        print(f"\nStep {i+1}: 执行动作 {config.action_list[action_idx]}")
        obs, reward, done, info = env.step(action)
        print("obs keys:", obs.keys())
        print("image shape:", obs["image"].shape, "dtype:", obs["image"].dtype)
        print("is_first:", obs.get("is_first"), "is_terminal:", obs.get("is_terminal"))
        print("reward:", reward, "done:", done)
        if done:
            print("Episode done, 重置环境")
            obs = env.reset()