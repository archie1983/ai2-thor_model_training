import prior
import random
from IPython import get_ipython
from . import NavigationTrainingDataManagement

from thortils import (launch_controller,
                      convert_scene_to_grid_map, proper_convert_scene_to_grid_map,
                      proper_convert_scene_to_grid_map_and_poses)

from thortils.navigation import get_shortest_path_to_object_type, get_shortest_path_to_object
from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany
from thortils.controller import _resolve
from thortils.object import thor_closest_object_of_type, thor_visible_objects
from . import RobotNavigationControl, RoomType
from thortils.scene import ThorSceneInfo
from thortils.map3d import Mapper3D

import thortils as tt

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
import copy
from . import (get_rooms_ground_truth,
               get_visible_objects_from_collection,
               get_all_objects, get_all_objects_of_type,
               get_path_length, get_centre_of_the_room,
               room_this_point_belongs_to, angle_to_turn_to_face_p2_from_p1)


class DiffusionTrainingDataExtractor:
    def __init__(self, data_store_dir="harvested_data"):
        self.HABITAT_SET_PREFIX = "train"  # "val" "test"
        self.data_store_dir = data_store_dir
        self.dataset = None
        self.controller = None
        self.rnc = RobotNavigationControl()

        self.last_start_position = None
        self.last_goal_position = None
        self.rooms_in_habitat = None

        self.habitat_mgmt = NavigationTrainingDataManagement(self.data_store_dir)
        self.NUMBER_OF_HABITATS_IN_BATCH = 1  # 55 # how many habitats in one go do we want to explore
        self.NUMBER_OF_EXPLORATIONS_PER_HABITAT = 1  # insane number - we're never going to get 1000, but this way it ensures that we get all there is

        self.room_cnts = []

    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            #print(self.dataset)
        return self.dataset

    def reset_state(self):
        self.has_top_down_camera = False
        self.top_down_camera_id = 0

    # If any questions, please check navigation_training_data_extractor.py
    # Any code without annotations probably came from navigation_training_data_extractor.py
    # The reason of without annotations just for clean and short
    def ae_load_proctor_habitat(self, habitat_id):
        dataset = self.getDataSet()

        data_split = self.HABITAT_SET_PREFIX
        self.habitat_id = habitat_id

        print("Loading : " + data_split + "[" + str(habitat_id) + "]")
        house = dataset[data_split][habitat_id]
        self.rooms_in_habitat = get_rooms_ground_truth(house)

        print("ROOMS:", len(self.rooms_in_habitat))
        self.room_cnts.append(len(self.rooms_in_habitat))

        return house

    def ae_process_proctor_habitat(self, habitat, habitat_id):
        self.habitat_mgmt.start_habitat(habitat_id)

        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
            self.rnc.set_controller(self.controller) # This allows our control scripts to interact with AI2-THOR environment
            self.mapper = Mapper3D(self.controller)
            self.rnc.set_mapper3D(self.mapper) # This allows taking FPV pictures of robot
        else:
            self.controller.reset(habitat)
            self.reset_state()
            self.rnc.reset_state()
            #self.rnc.set_controller(self.controller)

        self.do_habitat_explorations()
        self.habitat_mgmt.end_habitat()

    def do_habitat_explorations(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        rnd = random.Random(seed)

        reachable_positions = tt.thor_reachable_positions(self.controller)
        # Sample from reachable positions, max up to 300 points
        placements = self.sample_by_point(reachable_positions, 300, rnd=rnd)
        print('The number of reachable positions:', len(reachable_positions))
        print('The number of placements: ', len(placements))

        explorations_processed = 0
        for p in placements:
            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)

            point_for_room_search = (p[0], "", p[1])

            room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
            room_centre = room_of_placement[2]
            print("Placement: ", place_with_rtn)
            print("ROOM of placement: ", room_of_placement)
            print("Turn: ", angle_to_turn_to_face_p2_from_p1(p, (room_centre.x, room_centre.y)))

            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # Start new exploration data storage
            habitat_data_store = self.habitat_mgmt.start_new_exploration()  # get the directory for the new exploration.
            self.mapper.set_scene_id("", habitat_data_store)

            # Now plan path to the centre of the room
            try:
                path_and_plan = self.get_path_to_target_point(room_centre)
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                print(f"ERROR: {e}")
                continue

            print("PATH & PLAN: ", path_and_plan)
            path = path_and_plan[0]
            plan = path_and_plan[1]

            # Walk through the plan
            # Store pictures
            self.rnc.follow_planned_path_and_store_data(path, plan, self.habitat_mgmt)

            # close off current exploration
            self.habitat_mgmt.end_current_exploration()
            explorations_processed += 1
            if (explorations_processed >= self.NUMBER_OF_EXPLORATIONS_PER_HABITAT):
                break

    def get_path_to_target_point(self, target_point):
        (start_position, start_rotation) = self.rnc.get_agent_pos_and_rotation()

        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)

        self.last_goal_position = {'x': target_point.x, 'y': 0.9009993672370911, 'z': target_point.y}
        target_position = (self.last_goal_position['x'], self.last_goal_position['y'], self.last_goal_position['z'])
        print('Target Position: ', target_position)
        print('Target Point: ', target_point)

        keywords = {'v_angles': [0], 'return_plan': True, 'diagonal_ok': True}
        return get_shortest_path_to_object(self.controller, "TargetPoint", start_position, start_rotation, target_position=target_position, **keywords)

    def sample_by_point(self, candidates, num_samples, rnd):
        n = min(len(candidates), num_samples)
        return rnd.sample(candidates, n)

    def process_1_batch_of_habitats(self):
        highest_habitat_index = self.habitat_mgmt.last_extracted_habitat()
        print("Highest habitat explored: " + str(highest_habitat_index))
        processed_habitats_in_this_batch = 0

        while processed_habitats_in_this_batch < self.NUMBER_OF_HABITATS_IN_BATCH:

            habitat_id = highest_habitat_index + 1 + processed_habitats_in_this_batch
            habitat = self.ae_load_proctor_habitat(habitat_id)
            processed_habitats_in_this_batch += 1

            if not habitat:
                continue

            self.ae_process_proctor_habitat(habitat, habitat_id)

        print(np.mean(self.room_cnts), np.median(self.room_cnts))
        print(self.room_cnts)

        if (self.controller != None):
            self.controller.stop()

