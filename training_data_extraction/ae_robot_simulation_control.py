# Importing the necessary libraries for AI2-THOR to run

#!pip install --upgrade ai2thor ai2thor-colab &> /dev/null
import ai2thor
import ai2thor_colab
import time, os, cv2
import math
from typing import Dict, List
from PIL import Image

from ai2thor.controller import Controller
from ai2thor_colab import (
    plot_frames,
    show_objects_table,
    side_by_side,
    overlay,
    show_video
)

import prior

from thortils import thor_teleport2d
from thortils.controller import _resolve
from thortils.agent import thor_agent_pose, thor_pose_as_tuple

from . import (get_path_length, convert_pose_set2tuple, normalize_colors)

# Class for controlling robot navigation. This is where we will have all the navigation commands.
# This has NOT yet got the LLM connected, but merely a set of tools to move the robot and to interact
# with the simulation environment.
class RobotNavigationControl:
    is_DEBUG = False
    NUM_ANGLES = 3 # how many angles we want to capture from each location along the path

    # Set a controller for the robot navigation control to use so that it
    # can interact with the AI2-THOR environment
    def set_controller(self, controller):
        self.controller = controller
        self.prev_pose = None

    # Set a mapper3D object from Thortils so that we can take snapshots of the
    # FPV of the robot.
    def set_mapper3D(self, mapper):
        self.mapper = mapper

    def start_procthor(self):
        dataset = prior.load_dataset("procthor-10k")
        #dataset
        house = dataset["train"][0]
        type(house), house.keys(), house
        self.controller = Controller(scene=house)

    # Starts server
    def start_ai2_thor(self):
        ai2thor_colab.start_xserver()
        "AI2-THOR Version: " + ai2thor.__version__

    # Initialises controller
    def initialise_controller(self, third_party_cam = True, omni_view = True):
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=3,
            scene="FloorPlan415",
            # image modalities
            #renderDepthImage=False,
            #renderInstanceSegmentation=False,
            # camera properties
            width=600,
            height=600,
            fieldOfView=120,
            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            #rotateStepDegrees=15,
        )

        # If debug is enabled, then print scene name and a few other things.
        if (self.is_DEBUG):
            event = self.controller.step(action="RotateRight")
            metadata = event.metadata
            print(event, event.metadata.keys())
            print("sceneName : " + self.controller.last_event.metadata["sceneName"])
            print("agent_pos : " + str(self.controller.last_event.metadata["agent"]["position"]))
            print("agent_rtn : " + str(self.controller.last_event.metadata["agent"]["rotation"]))
            #print("actionReturn : " + controller.last_event.metadata["actionReturn"])

        # If we want a top-view camera for floor plan, then add it here
        if (third_party_cam):
            event = self.controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=-4.25, y=2, z=-2.5),
                rotation=dict(x=90, y=0, z=0),
                fieldOfView=120
            )

    # Get robot's current position and rotation in format that Thortils use
    def get_agent_pos_and_rotation(self):

        pos = (self.controller.last_event.metadata["agent"]["position"]["x"], self.controller.last_event.metadata["agent"]["position"]["y"], self.controller.last_event.metadata["agent"]["position"]["z"])
        rtn = (self.controller.last_event.metadata["agent"]["rotation"]["x"], self.controller.last_event.metadata["agent"]["rotation"]["y"], self.controller.last_event.metadata["agent"]["rotation"]["z"])

        return (pos, rtn)

    # Exposing plot_frames function from AI2-THOR
    def show_current_robot_view(self, ev):
        plot_frames(ev.third_party_camera_frames[0])

    # Return current controller
    def get_current_controller(self):
        return self.controller

    # Return 1st object ID of the required object type specified by name
    def get_obj_id(self, obj_name):
        obj_of_interest = self.validate_object_in_collection(obj_name, self.get_visible_objects())
        #print(obj_of_interest)
        return obj_of_interest['objectId']

    # Purely for debug - shows where the robot is, where the target is and the list of available positions
    # that can be navigated to.
    def print_world_state(self, target_name):
        obj_navigate_to = self.validate_object_in_collection(target_name, self.get_visible_objects())

        print("target pos : " + str(obj_navigate_to['position']))
        print("agent pos : " + str(self.controller.last_event.metadata["agent"]["position"]))

        #reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        #print("reachable cells : " + str(reachable_positions))
        rc = self.get_reachable_cells_2d()
        print("reachable cells : " + str(rc))

    # Takes the list of available positions to navigate to, throws out the vertical dimension of each of them
    # (key: y) and creates a new list of these cells containing only 2d information.
    def get_reachable_cells_2d(self):
        reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        positions_2d = []

        for pos in reachable_positions:
            positions_2d.append((pos["x"], pos["z"]))

        return positions_2d

    # Print a table of all objects in the scene
    def show_all_objects(self):
        show_objects_table(self.controller.last_event.metadata['objects'])
        print(self.controller.last_event.metadata['objects'])

    # Rotate left by given number of degrees degrees
    def rotate_left(self, deg):
        frames = []
        for _ in range(int(deg) // 5):
            frames.append(self.controller.step(action="RotateLeft", degrees=5).frame)
            time.sleep(0.05)

    # Rotate right by given number of degrees degrees
    def rotate_right(self, deg):
        frames = []
        for _ in range(int(deg) // 5):
            frames.append(self.controller.step(action="RotateRight", degrees=5).frame)
            time.sleep(0.05)

    # Move ahead by the given number of meters
    def move_ahead(self, distance = 0.25):
        frames = []
        for _ in range(int(distance * 100) // 5):
            frames.append(self.controller.step(action="MoveAhead", moveMagnitude=0.05).frame)
            time.sleep(0.05)

    # Rotate right or left depending on the degree (positive degree- right, negative - left)
    def rotate_by_degree(self, deg):
        if (deg < 0):
            self.rotate_left(abs(deg))
        else:
            self.rotate_right(abs(deg))

    # Store visible objects in the self.visible_objects collection and print them out if needed
    def get_visible_objects(self, print_objects = False):
        objects = self.controller.last_event.metadata['objects']
        visible_objects = []

        for obj in objects:
            if obj['visible']:
                if print_objects:
                    print(obj['objectType'] + " : " + str(obj['position']))
                visible_objects.append(obj)

        return visible_objects

    # Find the closest positon from the given reachable positions to the given object position using
    # Pythagorean theorem.
    def closest_position(self, object_position: Dict[str, float], reachable_positions: List[Dict[str, float]]) -> Dict[str, float]:
        out = reachable_positions[0]
        min_distance = float('inf')
        for pos in reachable_positions:
            # NOTE: y is the vertical direction, so only care about the x/z ground positions
            dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
            if dist < min_distance:
                min_distance = dist
                out = pos
        return out

    # Finds the given object name in the given collection of objects and if found, returns the actual object
    def validate_object_in_collection(self, obj_name, obj_collection):
        obj_names = sorted([obj["objectType"] for obj in obj_collection])

        try:
            assert obj_name in obj_names
        except AssertionError:
            print(obj_name + " is not visible!!!!!!!!!!!!!!!!")
            return None

        obj_of_interest = next(obj for obj in obj_collection if obj["objectType"] == obj_name)
        return obj_of_interest

    # Navigate to object defined by the name in the input
    def navigate_to_object(self, obj_name):
        #plot_frames(self.controller.last_event)
        obj_navigate_to = self.validate_object_in_collection(obj_name, self.get_visible_objects())

        # Can't navigate to an unknown object
        if (obj_navigate_to is None):
            return

        reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        pos_navigate_to = self.closest_position(obj_navigate_to['position'], reachable_positions)

        #print(pos_navigate_to)

        self.controller.step(action="Teleport", **pos_navigate_to)
        #plot_frames(self.controller.last_event)

    # Navigate to defined pose
    def navigate_to_pose(self, pose):
        #plot_frames(self.controller.last_event)

        # Can't navigate to an unknown pose
        if (pose is None):
            return

        (position, rotation) = pose
        #dict_pos = {'x': position[0], 'y': position[1], 'z': position[2]}

        #print("navigating to: ", pose)
        #self.controller.step(action="Teleport", **position)
        #self.controller.step(action="TeleportFull", **position, rotation=rotation['y'])
        self.controller.step(action="Teleport", position=position, rotation=rotation)
        #plot_frames(self.controller.last_event)
        img_uri = self.mapper.get_front_view()
        img_uri_sides = self.get_side_cameras_views(self.mapper.get_target_dir(), self.mapper.get_current_img_counter())
        img_uris = [img_uri]
        img_uris.extend(img_uri_sides)
        return img_uris

    ##
    # We may not always want the full URI from self.mapper.get_target_dir(), we may
    # want the relative path. This will give it to us.
    ##
    def relative_target_dir(self, uri):
        components = uri.split("/")
        return components[-2] + "/" + components[-1]

    ##
    # Reset some internal variables, e.g. the flag that we have a top-down camera
    ##
    def reset_state(self):
        self.side_cameras_exist = False
        self.left_cam_index = 0
        self.right_cam_index = 0

    ##
    # Get images from the side cameras, e.g., if we have 3 cameras,
    # then this would be from the left and right ones.
    #
    # This method rotates the robot to face in the required directions and takes pictures
    ##
    def get_side_cameras_views(self, img_dir, img_index):
        """Capture a 360-degree panorama by rotating the agent"""
        event = self.controller.last_event
        original_yaw = event.metadata["agent"]["rotation"]["y"]
        initial_position = event.metadata["agent"]["position"]
        initial_rotation = event.metadata["agent"]["rotation"]
        initial_standing = event.metadata["agent"]["isStanding"]  # Get current standing state

        img_urls = []

        # Capture frames at different angles. We already have the front view, so get the others,
        # that's why start with 1, not 0, but divide 360 still by the full number of angles.
        for i in range(1, self.NUM_ANGLES):
            angle = (360 / self.NUM_ANGLES) * i

            required_yaw = (original_yaw + angle) % 360
            # Teleport to the same position but with a different rotation
            self.controller.step(
                action="TeleportFull",
                position=initial_position,
                rotation=dict(x=0, y=required_yaw, z=0),
                horizon=initial_rotation["x"],
                standing=initial_standing  # Include standing parameter
            )

            # Get the frame from the default camera
            img = self.controller.last_event.cv2img
            #frames.append(frame)

            # store them
            os.makedirs(img_dir, exist_ok=True)
            img_url = os.path.join(img_dir, str(int(angle)) + "_" + str(img_index) + ".png")
            cv2.imwrite(img_url, img)
            img_urls.append(img_url)

        # Restore original position and rotation
        self.controller.step(
            action="TeleportFull",
            position=initial_position,
            rotation=initial_rotation,
            horizon=initial_rotation["x"],
            standing=initial_standing  # Include standing parameter
        )

        return img_urls

    ##
    # Get images from the side cameras, e.g., if we have 3 cameras,
    # then this would be from the left and right ones.
    #
    # This method uses 2 extra cameras attached to the robot
    ##
    def get_side_cameras_views_2(self, img_dir, img_index):
        if not hasattr(self, 'side_cameras_exist'):
            self.side_cameras_exist = False
            self.left_cam_index = 0
            self.right_cam_index = 0

        """Position cameras relative to the agent's current position and rotation"""
        event = self.controller.last_event
        agent_position = event.metadata["agent"]["position"]
        agent_rotation = event.metadata["agent"]["rotation"]

        # Camera height offset from agent position
        camera_height_offset = 0.1  # Slightly above agent's camera

        # Cameras at 120, and 240 degrees relative to agent)
        left_angle_offset = 240
        right_angle_offset = 120
        left_absolute_angle = (agent_rotation["y"] + left_angle_offset) % 360
        right_absolute_angle = (agent_rotation["y"] + right_angle_offset) % 360

        # Convert to radians for math calculations
        left_angle_rad = math.radians(left_absolute_angle)
        right_angle_rad = math.radians(right_absolute_angle)

        # Small offset from center of agent (e.g., 0.05 units)
        offset_distance = 0.05

        # Calculate camera position (small offset from agent center)
        left_camera_x = agent_position["x"] + offset_distance * math.sin(left_angle_rad)
        left_camera_y = agent_position["y"] + camera_height_offset
        left_camera_z = agent_position["z"] + offset_distance * math.cos(left_angle_rad)
        right_camera_x = agent_position["x"] + offset_distance * math.sin(right_angle_rad)
        right_camera_y = left_camera_y
        right_camera_z = agent_position["z"] + offset_distance * math.cos(right_angle_rad)

        print(left_camera_x, left_camera_y, left_camera_z, left_absolute_angle, left_angle_rad)

        # If cameras don't exist yet, then create them, otherwise update
        if not self.side_cameras_exist:
            # Add here cameras to cover whole 360 degrees around robot
            event = self.controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=left_camera_x, y=left_camera_y, z=left_camera_z),
                rotation=dict(x=0, y=left_absolute_angle, z=0),  # Camera at 240 degrees
                fieldOfView=120,
                orthographic=False
            )
            self.left_cam_index = len(event.third_party_camera_frames) - 1
            event = self.controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=right_camera_x, y=right_camera_y, z=right_camera_z),
                rotation=dict(x=0, y=right_absolute_angle, z=0),  # Camera at 120 degrees
                fieldOfView=120,
                orthographic=False
            )
            self.right_cam_index = len(event.third_party_camera_frames) - 1
            self.side_cameras_exist = True
        else:
            #UpdateThirdPartyCamera
            # If we already have side cameras, then we need to update them to move with the current location of robot
            self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self.left_cam_index,
                position=dict(x=left_camera_x, y=left_camera_y, z=left_camera_z),
                rotation=dict(x=0, y=left_absolute_angle, z=0)  # Camera at 240 degrees
            )

            self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self.right_cam_index,
                position=dict(x=right_camera_x, y=right_camera_y, z=right_camera_z),
                rotation=dict(x=0, y=right_absolute_angle, z=0),  # Camera at 120 degrees
            )

        # get the frames
        #event = self.controller.last_event
        left_frame = event.third_party_camera_frames[self.left_cam_index]
        right_frame = event.third_party_camera_frames[self.right_cam_index]

        os.makedirs(img_dir, exist_ok=True)

        # store them
        left_img_url = os.path.join(img_dir, "L_" + str(img_index) + ".png")
        right_img_url = os.path.join(img_dir, "R_" + str(img_index) + ".png")

        #cv2.imwrite(left_img_url, Image.fromarray(left_frame))
        #cv2.imwrite(right_img_url, Image.fromarray(right_frame))
        cv2.imwrite(left_img_url, normalize_colors(left_frame))
        cv2.imwrite(right_img_url, normalize_colors(right_frame))

        return [left_img_url, right_img_url]

    ##
    # Follow through a pre-planned path
    ##
    def follow_planned_path(self, path, plan, data_manager):
        #self.prev_pose = self.get_agent_pos_and_rotation() # This is where we are before the plan started
        self.prev_pose = thor_agent_pose(self.controller) # This is where we are before the plan started

        #print((len(path) == len(plan)))
        if (self.is_DEBUG):
            print(path)
            print(plan)

        # Looks like I will need a wider angle camera and a better path length estimate to take into account
        # smaller distances otherwise we get 0 length estimate when there is still a move left. Also turning
        # might need a different score.

        remaining_path = path
        img_uri = self.mapper.get_front_view()
        img_uri_sides = self.get_side_cameras_views(self.mapper.get_target_dir(), self.mapper.get_current_img_counter())
        img_uris = [img_uri]
        img_uris.extend(img_uri_sides)
        img_uris = [self.relative_target_dir(iu) for iu in img_uris]

        if (self.is_DEBUG):
            print("self.prev_pose", self.prev_pose)
        path_length_at_this_step = get_path_length(remaining_path, thor_pose_as_tuple(self.prev_pose))

        # Before we start traversing the path, the current pose is what we have in self.prev_pose
        pose = self.prev_pose

        for i in range(len(path)):
            step = plan[i] # current step is how to get from previous point to here
            # storing the current path metrics with the last taken picture. When i == 0, the picture
            # will be taken outside the loop and will be the very first view before the motion starts.
            print(pose, step[0], path_length_at_this_step, img_uris)

            # What are we storing in metrics:
            # step[0] : What action is best to take at this location
            # path_length_at_this_step : How long have we got to go before we have taken this action
            # img_uris : What does it look like at this point
            data_manager.add_metrics((pose, step[0], path_length_at_this_step, img_uris))
            # now update the pose and recalculate path length for the next step.
            # The very last pose will yield path length of 0 and loop will exit, but
            # that's ok because we have a final step after the loop that we gather as STOP action
            pose = path[i]
            path_length_at_this_step = get_path_length(remaining_path, thor_pose_as_tuple(pose))
            img_uris = self.navigate_to_pose(pose) # move to the next step and take a picture
            img_uris = [self.relative_target_dir(iu) for iu in img_uris]
            remaining_path = remaining_path[1:] # update remaining path

        print(pose, "STOP", 0, img_uris) # final step - we've arrived. Remaining path length = 0 and action = STOP
        data_manager.add_metrics((pose, "STOP", 0, img_uris))

        self.controller.step(action="Done")

    ##
    # Teleport back to the place where we were before last path plan was executed
    ##
    def return_to_prev_pose(self):
        print(self.prev_pose)
        if self.prev_pose is not None:
            self.navigate_to_pose(self.prev_pose)

    def execute_action_plan(self, plan):
        for act in plan:
            self.controller.step(action=act[0])
            time.sleep(0.2)

    # Print pose of the object defined by the name in the input
    def print_pose_of_object(self, obj_name):
        obj_of_interest = self.validate_object_in_collection(obj_name, self.controller.last_event.metadata['objects'])

        # Unknown object
        if (obj_of_interest is None):
            return

        print("position of " + obj_name + " : " + str(obj_of_interest["position"]))
        print("rotation of " + obj_name + " : " + str(obj_of_interest["rotation"]))

    # Print current pose of robot
    def print_current_pose_of_robot(self):
        print("agent_pos : " + str(self.controller.last_event.metadata["agent"]["position"]))
        print("agent_rtn : " + str(self.controller.last_event.metadata["agent"]["rotation"]))

    # Calculate angle from the target that we need to rotate by to face the target
    def get_angle_offset_from_target(self, obj_name):
        obj_of_interest = self.validate_object_in_collection(obj_name, self.controller.last_event.metadata['objects'])

        # Unknown object
        if (obj_of_interest is None):
            return None

        robot_position = self.controller.last_event.metadata["agent"]["position"]

        # Using formula tg(alpha) = a/b to find relative angle from robot to object
        #tg_alpha = (robot_position["z"] - obj_of_interest["position"]["z"]) / (robot_position["x"] - obj_of_interest["position"]["x"])
        z_diff = (robot_position["z"] - obj_of_interest["position"]["z"])
        x_diff = (robot_position["x"] - obj_of_interest["position"]["x"])
        #print(math.degrees(math.atan2(z_diff, x_diff)))
        return math.degrees(math.atan2(z_diff, x_diff))

    # Rotate to face the selected target
    def rotate_to_face_target(self, obj_name):
        angle_to_target = self.get_angle_offset_from_target(obj_name)

        if (angle_to_target is None):
            print(obj_name + " not visibile")
            return

        current_robot_yaw = self.controller.last_event.metadata["agent"]["rotation"]["y"]

        self.rotate_by_degree(angle_to_target - current_robot_yaw)
        #print("rotating by: " + str(angle_to_target - current_robot_yaw) + " " + str(angle_to_target) + " " + str(current_robot_yaw))

    # Get ceiling camera image -- can try using this if you don't like the 3rd camera set up earlier.
    def get_ceiling_image(self):
        # puts the camera in the ceiling, then puts it back with the robot
        event = self.controller.step('ToggleMapView')
        self.controller.step('ToggleMapView')
        return event.frame

    ##
    # Teleport to an arbitrary pose. The pose is expected to be in
    # the format of (x, y, th) where x and y are 2D coordinates but th
    # is a rotational component - degrees.
    ##
    def teleport_to(self, pose):
        thor_teleport2d(self.controller, pose)
        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)
