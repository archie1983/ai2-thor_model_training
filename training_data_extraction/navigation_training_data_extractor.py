import os
import pickle
import prior
import random

from . import NavigationTrainingDataManagement

from thortils import (launch_controller,
                      convert_scene_to_grid_map, proper_convert_scene_to_grid_map, proper_convert_scene_to_grid_map_and_poses)

from thortils.navigation import get_shortest_path_to_object_type, get_shortest_path_to_object
from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany
from thortils.controller import _resolve
from thortils.object import thor_closest_object_of_type, thor_visible_objects
from . import RobotNavigationControl
from thortils.scene import ThorSceneInfo
from thortils.map3d import Mapper3D

from thortils.utils.math import sep_spatial_sample, euclidean_dist
import thortils as tt

import matplotlib.pyplot as plt
from PIL import Image
import copy
from ai2_thor_utils import (get_rooms_ground_truth,
                            get_visible_objects_from_collection,
                            get_all_objects, get_all_objects_of_type,
                            get_path_length)
##
# This class will load a scene from ProcThor and then start harvesting dat from
# it that can be used for training our neural networks (the intuition CNN
# the navigation diffuser).
##
class NavigationTrainingDataExtractor:
    def __init__(self, data_store_dir = "harvested_data"):
        self.HABITAT_SET_PREFIX = "train" # "val" "test"
        self.data_store_dir = data_store_dir
        self.dataset = None
        self.controller = None
        self.rnc = RobotNavigationControl()

        self.last_start_position = None
        self.last_goal_position = None

        self.habitat_mgmt = NavigationTrainingDataManagement(self.data_store_dir)
        self.NUMBER_OF_HABITATS_IN_BATCH = 1
        self.NUMBER_OF_EXPLORATIONS_PER_HABITAT = 3

    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            #print(self.dataset)
        return self.dataset

    ##
    # Load a PROCTHOR scene specified by the habitat_id. They will all be loaded
    # from train, val or test splits depending on the HABITAT_SET_PREFIX variable.
    ##
    def ae_load_proctor_habitat(self, habitat_id):
        dataset = self.getDataSet()

        data_split = self.HABITAT_SET_PREFIX
        self.habitat_id = habitat_id

        print("Loading : " + data_split + "[" + str(habitat_id) + "]")
        house = dataset[data_split][habitat_id]
        rooms = get_rooms_ground_truth(house)
        print("ROOMS:" + str(rooms))

        # For now accept every habitat as good. We may want to introduce some
        # logic here at a later time to only explore suitable rooms using some
        # criteria that I don't know yet.
        habitat_ok = True

        #print(self.controller.scene)
        #self.scene_info = ThorSceneInfo("FloorPlan01-default", self.controller.last_event.metadata['objects'])

        if habitat_ok:
            return house
        else:
            return None

    # Load an AI2-THOR controller with the chosen habitat and start exploration
    def ae_process_proctor_habitat(self, habitat, habitat_id):

        self.habitat_mgmt.start_habitat(habitat_id)

        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
            self.rnc.set_controller(self.controller) # This allows our control scripts to interact with AI2-THOR environment
            self.mapper = Mapper3D(self.controller)
            self.rnc.set_mapper3D(self.mapper) # This allows taking FPV pictures of robot
        else:
            self.controller.reset(habitat)
            #self.rnc.set_controller(self.controller)

        self.do_all_habitat_explorations()
        self.habitat_mgmt.end_habitat()

    # Generate all random points that we want to generate and navigate from there
    # to whatever target we want (door, or middle of room, or whatever)
    def do_all_habitat_explorations(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 20
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [30]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have

        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        rnd = random.Random(seed)

        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        reachable_positions = tt.thor_reachable_positions(self.controller)
        placements = sep_spatial_sample(reachable_positions, sep, num_stops,
                                        rnd=rnd)

        print(placements)

        explorations_processed = 0
        for p in placements:
            # append a rotation to the place. We will want to change this to face
            # what we want to face
            place_with_rtn = p + (0,)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # Start new exploration data storage
            habitat_data_store = self.habitat_mgmt.start_new_exploration() # get the directory for the new exploration.
            self.mapper.set_scene_id("", habitat_data_store)

            explorations_processed += 1
            if (explorations_processed >= self.NUMBER_OF_EXPLORATIONS_PER_HABITAT):
                break

        ## This will teleport us randomly to different positions. Maybe we don't even
        # need num_rotates because we will want to be facing general direction of navigation
        # anyway, We will therefore need some routine to generate a vector (euclidian vector)
        # to the target. But that's for some other day. Gotta sleep now.
        #
        # Anyway, from each random position we want to navigate to somewhere (middle of room
        # or door, or whatever).
        #for pos in tqdm(placements, desc="Running Explorations"):
        #    #for _ in range(num_rotates):
        #    event = tt.thor_place_agent_randomly(self.controller,
        #                                         pos=pos,
        #                                         v_angles=v_angles,
        #                                         h_angles=h_angles,
        #                                         rnd=rnd)

    # Navigate to a door - any door, at this point I'm just trying out a concept.
    def navigate_to_door(self):
        doors = self.find_all_doors()
        door_of_interest = doors[0]
        path_and_plan = self.get_path_to_actual_object(door_of_interest, is_door=True)
        path = path_and_plan[0]
        plan = path_and_plan[1]

        for element in path:
            print(element)
        #print(path

        self.rnc.follow_planned_path(path)

    def get_path_to(self, object_type):
        #return get_shortest_path_to_object_type(controller, object_id, start_position, start_rotation, **{"return_plan": return_plan})
        (start_position, start_rotation) = self.rnc.get_agent_pos_and_rotation()

        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)

        obj = thor_closest_object_of_type(self.controller, object_type)
        #print(obj)
        self.last_goal_position = obj["position"]

        return get_shortest_path_to_object_type(self.controller, object_type, start_position, start_rotation)

    ##
    # Retursn a list of all doors in the scene
    ##
    def find_all_doors(self):
        #"Doorway"
        #objects = get_all_objects(self.rnc.controller, True)
        #door_ids = self.scene_info.objects_of_type("Doorway")
        door_objs = get_all_objects_of_type(self.rnc.controller, "Doorway")
        #print(door_objs)
        return door_objs

    def bring_me_this_from_actual_objs(self, what_to_bring):

        actual_objects_to_look_at = thor_visible_objects(self.rnc.controller)

        for obj in actual_objects_to_look_at:
            print(obj["objectType"])
            if obj["objectType"] == what_to_bring:
                needed_obj = obj
                print(obj["objectType"])
                break

        #path = self.get_path_to(object_to_look_at)
        path = self.get_path_to_actual_object(needed_obj)
        #path = self.get_path_to("Fridge")
        #print(str(path))

        return path

    ##
    # For display purposes - the top down view of the habitat
    ##
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        return Image.fromarray(top_down_frame)

    ##
    # Sets up the self.last_start_position and self.last_goal_position which is necessary for
    # visualizing path.
    #
    # If is_door is set to True, then we are navigation to a door and then we want to arrive at its
    # center which is usually not its position. Door position is typically the hinge-side of frame.
    ##
    def get_path_to_actual_object(self, needed_obj, is_door=False):
        #return get_shortest_path_to_object_type(controller, object_id, start_position, start_rotation, **{"return_plan": return_plan})
        (start_position, start_rotation) = self.rnc.get_agent_pos_and_rotation()

        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)

        obj = needed_obj #thor_closest_object_of_type(self.controller, object_type)
        #print(obj)
        if is_door:
            self.last_goal_position = obj["axisAlignedBoundingBox"]["center"]
            target_position = (self.last_goal_position['x'], self.last_goal_position['y'], self.last_goal_position['z'])
            #print("# AE: target_position: ", target_position)
            #print("# AE: obj_pos: ", obj["position"])
        else:
            self.last_goal_position = obj["position"]
            target_position = None

        #print("# AE: start_pose: ", (start_position, start_rotation))

        #print("AE: v_angles: " + str(v_angles) + " # h_angles: " + str(h_angles)
        #        + " # movement_params: " + str(movement_params) + " # goal_distance: " + str(goal_distance)
        #        + " # diagonal_ok: " + str(diagonal_ok) + " # positions_only: " + str(positions_only)
        #        + " # return_plan: " + str(return_plan) + " # as_tuples: " + str(as_tuples))

        #keywords = {'v_angles': [30], 'return_plan': True}
        keywords = {'v_angles': [0], 'return_plan': True, 'diagonal_ok': True}
        return get_shortest_path_to_object(self.controller, obj["objectId"], start_position, start_rotation, target_position=target_position, **keywords)

    def get_current_pose(self):
        return self.rnc.get_agent_pos_and_rotation()

    ##
    # Plot a path on the top-down view of the habitat
    ##
    def visualise_path(self, path):
        grid_size = self.controller.initialization_parameters["gridSize"]

        reachable_positions = [
            tuple(map(lambda x: roundany(x, grid_size), pos))
            for pos in thor_reachable_positions(self.controller)]

        x_max = max([pos[0] for pos in reachable_positions])
        z_max = max([pos[1] for pos in reachable_positions])
        x_min = min([pos[0] for pos in reachable_positions])
        z_min = min([pos[1] for pos in reachable_positions])

        start = self.last_start_position
        goal = self.last_goal_position

        fig, ax = plt.subplots()

        # another way how to plot the path
        #x = [p[0]["x"] for p in path]
        #z = [p[0]["z"] for p in path]
        #ax.scatter(x, z, s=300, c='gray', zorder=1)

        # setting up for the top-down picture of the habitat
        print(str(x_min-grid_size) + " " + str(x_max+grid_size) + " " + str(z_min-grid_size) + " " + str(z_max+grid_size))
        img = self.get_top_down_frame()
        ex_mul = 7
        ax.imshow(img, extent=[x_min-ex_mul*grid_size, x_max+ex_mul*grid_size, z_min-ex_mul*grid_size, z_max+ex_mul*grid_size])

        # set up for the path print
        lim_mul = 4
        ax.set_xlim(x_min-lim_mul*grid_size, x_max+lim_mul*grid_size)
        ax.set_ylim(z_min-lim_mul*grid_size, z_max+lim_mul*grid_size)

        # start pos
        xs = start["x"]
        zs = start["z"]
        ax.scatter([xs], [zs], s=100, c='red', zorder=4)

        # goal
        xg = goal["x"]
        zg = goal["z"]
        ax.scatter([xg], [zg], s=100, c='green', zorder=4)

        # path
        for step in path:
            x = step[0]["x"]
            z = step[0]["z"]
            ax.scatter([x], [z], s=30, zorder=2, c="blue")

        plt.axis('off')
        plt.show()

    def get_controller(self):
        return self.controller

    ##
    # Gets all door paths, evaluates their lengths and sorts them by length,
    # finally returns the sorted list.
    ##
    def get_all_doors_sorted_by_distance_from_current_pose(self):
        doors = self.find_all_doors()
        all_door_paths = []
        current_pose = self.get_current_pose()
        for door in doors:
            path_and_plan = self.get_path_to_actual_object(door, is_door=True)
            path = path_and_plan[0]
            path_cost = get_path_length(path, current_pose)
            all_door_paths.append((path_cost, path))
            #print(str(get_path_length(path, current_pose)))

        result = sorted(all_door_paths, key=lambda x: x[0], reverse=False) # sort the result by path cost
        #for i in range(len(result)):
        #    print(str(result[i]))
        return result

    def process_1_batch_of_habitats(self):
        highest_habitat_index = self.habitat_mgmt.last_extracted_habitat()
        print("Highest habitat explored: " + str(highest_habitat_index))
        processed_habitats_in_this_batch = 0

        while processed_habitats_in_this_batch < self.NUMBER_OF_HABITATS_IN_BATCH:

            habitat_id = highest_habitat_index + 1
            habitat = self.ae_load_proctor_habitat(habitat_id)

            if not habitat:
                continue

            self.ae_process_proctor_habitat(habitat, habitat_id)
            processed_habitats_in_this_batch += 1


if __name__ == "__main__":
    ntde = NavigationTrainingDataExtractor()
    #doors = ntde.find_all_doors()
    ntde.process_1_batch_of_habitats()

    path_and_plan = ntde.get_path_to_actual_object(doors[0], is_door=True)
    path = path_and_plan[0]
    plan = path_and_plan[1]

    ntde.visualise_path(path)
    ntde.navigate_to_door()
