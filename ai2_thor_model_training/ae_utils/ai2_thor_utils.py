from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from . import RoomType
import math, cv2, prior, copy
#from thortils.utils.math import (euclidean_dist, to_deg)
#from thortils.agent import thor_pose_as_tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import traceback

##
# My own utilities functions for AI2-THOR. I couldn't find analogous functions in Thortils,
# so I wrote my own here. Eventually some of them should probably be pushed to Thortils
# project.
##
class AI2THORUtils:
    def __init__(self):
        self.dataset = None
        self.path_fig = None
        self.path_ax = None
        self.controller = None

    ##
    # Get Procthor-10k dataset
    ##
    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            # print(self.dataset)
        return self.dataset

    ##
    # Load a PROCTHOR scene specified by the habitat_id.
    ##
    def load_proctor_habitat(self, habitat_id, from_set="train"):
        dataset = self.getDataSet()
        self.habitat_id = habitat_id
        print("Loading : ", from_set ,"[" + str(habitat_id) + "]")
        #traceback.print_stack()
        house = dataset[from_set][habitat_id]
        return house

    ##
    # Extract visible objects from a collection of objects
    ##
    def get_visible_objects_from_collection(self, objects, print_objects = False):
        visible_objects = []

        for obj in objects:
            if obj['visible']:
                if print_objects:
                    print(obj['objectType'] + " : " + str(obj['position']))
                visible_objects.append(obj)

        return visible_objects

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Skip duplicates.
    ##
    def get_visible_object_names_from_collection_set(self, objects):
        objs_at_this_pos = set()

        vis_objs = self.get_visible_objects_from_collection(objects)

        for obj in vis_objs:
            objs_at_this_pos.add(obj['objectType'])

        return objs_at_this_pos

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Return as a comma separated list.
    ##
    def get_visible_object_names_from_collection_csv(self, objects):
        objs_at_this_pos = ""

        vis_objs = self.get_visible_objects_from_collection(objects)

        for obj in vis_objs:
            objs_at_this_pos += ", " + obj['objectType']

        if len(objs_at_this_pos) > 2:
            objs_at_this_pos = objs_at_this_pos[2:]

        return objs_at_this_pos

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Return as a comma separated list. Skip duplicates.
    ##
    def get_visible_object_names_from_collection_csv_unique(self, objects):
        objs_at_this_pos = ""

        vis_objs = self.get_visible_object_names_from_collection_set(objects)

        for obj in vis_objs:
            objs_at_this_pos += ", " + obj

        if len(objs_at_this_pos) > 2:
            objs_at_this_pos = objs_at_this_pos[2:]

        return objs_at_this_pos

    def set_controller(self, controller):
        self.controller = controller

    def visualise_path2(self, path, reachable_positions, unreachable_postions,
                        rooms_in_habitat, start, goal,
                        show_reachable_pos = False, show_unreachable_pos = False):
        grid_size = self.controller.initialization_parameters["gridSize"]

        all_corners = [corner for room_name, corners, center in rooms_in_habitat for corner in corners]
        #print("all_corners: ", all_corners)

        x_max = max([pos[0] for pos in all_corners])
        z_max = max([pos[1] for pos in all_corners])
        x_min = min([pos[0] for pos in all_corners])
        z_min = min([pos[1] for pos in all_corners])

        # Use interactive mode to prevent blocking
        plt.ion()  # Turn on interactive mode

        # Reuse existing figure or create new one
        if self.path_fig is None or not plt.fignum_exists(self.path_fig.number):
            self.path_fig, self.path_ax = plt.subplots(figsize=(10, 8))
            self.path_fig.canvas.manager.set_window_title("AI2-Thor Path Visualization")
        else:
            # Clear the existing plot
            self.path_ax.clear()

        # AE: Debug
        if show_reachable_pos:
            event = self.controller.step(action="GetReachablePositions")
            r_positions = event.metadata["actionReturn"]
            r_positions = [(pos['x'], pos['z']) for pos in r_positions]
            # #print("AE: positions: ", positions)
            #
            # reachable_x = [pos[0] for pos in r_positions]
            # reachable_y = [pos[1] for pos in r_positions]
            # self.path_ax.scatter(reachable_x, reachable_y, s=50, c='white', alpha=0.5, zorder=4, label='Pstep')
            #
            # #Get reachable positions
            # event = self.controller.step(action="GetReachablePositions")
            # r_positions = event.metadata["actionReturn"]
            #
            # print(f"Total reachable positions: {len(r_positions)}")
            # print(f"Sample positions: {r_positions[:5]}")
            #
            # #Check the bounds
            # x_coords = [pos['x'] for pos in r_positions]
            # z_coords = [pos['z'] for pos in r_positions]
            #
            # print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
            # print(f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
            #
            # #Get current agent position for reference
            # agent_pos = self.controller.last_event.metadata["agent"]["position"]
            # print(f"Agent position: x={agent_pos['x']:.3f}, z={agent_pos['z']:.3f}")
            #
            # for pos in reachable_positions:
            #    #print("pos: ", pos)
            #    self.path_ax.scatter(reachable_x, reachable_y, s=50, c='white', zorder=4, label='Pstep')

            reachable_x = [pos[0] for pos in reachable_positions]
            reachable_y = [pos[1] for pos in reachable_positions]
            #reachable_x = [pos[0] for pos in r_positions]
            #reachable_y = [pos[1] for pos in r_positions]
            self.path_ax.scatter(reachable_x, reachable_y, s=50, c='white', alpha=0.5, zorder=4, label='Pstep')
        if show_unreachable_pos:
            unreachable_x = [pos[0] for pos in unreachable_postions]
            unreachable_y = [pos[1] for pos in unreachable_postions]
            self.path_ax.scatter(unreachable_x, unreachable_y, s=25, c='yellow', alpha=0.5, zorder=4, label='Pstep')
        # Debug done

        # Setting up for the top-down picture of the habitat
        #print("AE: map size: grid_size: ", grid_size, " x_min: ", x_min, " x_max: ", x_max, " z_min: ", z_min, " z_max: ", z_max)
        #print("AE: start: ", start, " goal: ", goal)
        img = self.get_top_down_frame()

        # Problem 2: Fix coordinate alignment
        # AI2-Thor uses Y-axis flipped compared to matplotlib, so we need to flip the image
        #img_height, img_width = img.shape[:2]

        # Calculate proper extents based on the actual coordinate system
        #ex_mul = 7
        #extent = [x_min - ex_mul * grid_size, x_max + ex_mul * grid_size,
        #          z_max + ex_mul * grid_size, z_min - ex_mul * grid_size]  # Note: z_max first, then z_min

        extents = [
            [x_min, x_max, z_min, z_max],  # Normal
            [x_min, x_max, z_max, z_min],  # Z flipped
            [-x_max, -x_min, z_min, z_max],  # X flipped
            [-x_max, -x_min, z_max, z_min],  # Both flipped
        ]

        self.path_ax.imshow(img, extent=extents[0], origin='upper')  # Use origin='upper' to match coordinate system

        #self.path_ax.scatter([agent_pos['x']], [agent_pos['z']], s=25, c='magenta', zorder=4, label='Agent')

        # Set up for the path print
        #lim_mul = 4
        #self.path_ax.set_xlim(x_min - lim_mul * grid_size, x_max + lim_mul * grid_size)
        #self.path_ax.set_ylim(z_min - lim_mul * grid_size, z_max + lim_mul * grid_size)

        # Start pos (using z instead of y coordinate)
        xs = start[0][0]  # x coordinate
        zs = start[0][2]  # z coordinate (not y!)
        self.path_ax.scatter([xs], [zs], s=100, c='red', zorder=4, label='Start')

        # Goal
        xg = goal[0][0]  # x coordinate
        zg = goal[0][2]  # z coordinate (not y!)
        self.path_ax.scatter([xg], [zg], s=100, c='green', zorder=4, label='Goal')

        if path is not None:
            # Path
            for step in path:
                x = step[0]  # x coordinate
                z = step[1]  # z coordinate
                self.path_ax.scatter([x], [z], s=30, zorder=2, c="blue", alpha=0.7)

            # Optional: Draw path as connected lines
            if len(path) > 1:
                path_x = [step[0] for step in path]
                path_z = [step[1] for step in path]
                self.path_ax.plot(path_x, path_z, 'b-', alpha=0.5, linewidth=2, zorder=1)

        self.path_ax.legend()
        plt.axis('off')

        # Update the existing window
        self.path_fig.canvas.draw()
        self.path_fig.canvas.flush_events()
        plt.show(block=False)

        # Option 2: Save to file instead
        # plt.savefig(self.habitat_mgmt.get_current_top_view_fname(), bbox_inches='tight', dpi=150)

        # Option 3: Keep window open but don't block
        # plt.draw()

        return self.path_fig, self.path_ax  # Return for potential further manipulation

    ##
    # Plot a path on the top-down view of the habitat
    ##
    def visualise_path(self, path, reachable_positions, start, goal):
        grid_size = self.controller.initialization_parameters["gridSize"]

        x_max = max([pos[0] for pos in reachable_positions])
        z_max = max([pos[1] for pos in reachable_positions])
        x_min = min([pos[0] for pos in reachable_positions])
        z_min = min([pos[1] for pos in reachable_positions])

        fig, ax = plt.subplots()

        # another way how to plot the path
        #x = [p[0]["x"] for p in path]
        #z = [p[0]["z"] for p in path]
        #ax.scatter(x, z, s=300, c='gray', zorder=1)

        #print("ae: start, goal: ", start, goal)

        # setting up for the top-down picture of the habitat
        print("AE: map size: ", x_min-grid_size, x_max+grid_size, z_min-grid_size, z_max+grid_size)
        img = self.get_top_down_frame()
        ex_mul = 7
        ax.imshow(img, extent=[x_min-ex_mul*grid_size, x_max+ex_mul*grid_size, z_min-ex_mul*grid_size, z_max+ex_mul*grid_size])

        # set up for the path print
        lim_mul = 4
        ax.set_xlim(x_min-lim_mul*grid_size, x_max+lim_mul*grid_size)
        ax.set_ylim(z_min-lim_mul*grid_size, z_max+lim_mul*grid_size)

        # start pos
        xs = start[0][0]
        zs = start[0][2]
        ax.scatter([xs], [zs], s=100, c='red', zorder=4)

        # goal
        xg = goal[0][0]
        zg = goal[0][2]
        ax.scatter([xg], [zg], s=100, c='green', zorder=4)

        # path
        for step in path:
            x = step[0]
            z = step[1]
            ax.scatter([x], [z], s=30, zorder=2, c="blue")
        plt.axis('off')

#        if self.is_running_in_jupyter():
#            plt.show()
#        else:
#            plt.savefig(self.habitat_mgmt.get_current_top_view_fname())

        #plt.show(block=False)
        plt.pause(0.1)
        plt.draw()

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

        if not hasattr(self, 'has_top_down_camera'):
            self.has_top_down_camera = False
            self.top_down_camera_id = 0

        if not self.has_top_down_camera:
            #print("AE: Adding camera")
            # add the camera to the scene
            event = self.controller.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )
            self.top_down_camera_id = len(event.third_party_camera_frames) - 1
            self.has_top_down_camera = True
        else:
            #print("AE: updating camera")
            # If we already have a top-down view camera, then we need to update it to suit the current habitat
            event = self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self.top_down_camera_id,
                **pose,
                #position=position,
                #rotation=rotation
            )

        top_down_frame = event.third_party_camera_frames[self.top_down_camera_id]

        return Image.fromarray(top_down_frame)

##
# Calculates the angle that we need to turn in order to face p2 if we are
# standing at p1.
##
def angle_to_turn_to_face_p2_from_p1(p1, p2):
    # Define the coordinates of the two points
    (x1, y1) = p1
    (x2, y2) = p2

    # Calculate the vector components from point 1 to point 2
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the angle using atan2
    angle_to_face_point2 = math.atan2(dy, dx)

    # Convert the angle from radians to degrees if needed
    angle_degrees = math.degrees(angle_to_face_point2)

    # Normalize the angle to be between 0 and 360 degrees
    if angle_degrees < 0:
        angle_degrees += 360

    return angle_degrees

##
# Calculates Euclidean distance between two points on a plane. If we pass 3 coordinates, then we drop the
# middle one.
##
def euclidean_dist(p1, p2):
    if len(p1) > 2:
        p1 = (p1[0], p1[2])
    if len(p2) > 2:
        p2 = (p2[0], p2[2])
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

##
# Ground truth functions - data extracted from the actual room and point is
# tested to belong to the room polygon or not.
##
def is_point_inside_room_ground_truth(point_to_test, room_polygon):
    (x, y, z) = point_to_test
    point = Point(x, z)
    polygon = Polygon(room_polygon)
    return polygon.contains(point)

##
# Ground truth functions - data extracted from the actual room and point is
# tested to belong to the room polygon or not.
##
def what_room_is_point_in_ground_truth(rooms, point):
    for room in rooms:
        if is_point_inside_room_ground_truth(point, room[1]):
            return RoomType.interpret_label(room[0])
    return RoomType.interpret_label("NONE")

##
# Returns the room that the given point belongs to.
##
def room_this_point_belongs_to(rooms, point):
    for room in rooms:
        if is_point_inside_room_ground_truth(point, room[1]):
            return room

    return None

##
# Ground truth functions - extracted room polygon is analyzed to get
# its centroid (middle point).
##
def get_centre_of_the_room(room_polygon):
    polygon = Polygon(room_polygon)
    return polygon.centroid

##
# Ground truth functions - data extracted from the actual room and point is
# tested to belong to the room polygon or not.
##
def get_rooms_ground_truth(house):
    rooms = []
    #print(house)
    #print("\n")
    #print(house["rooms"])
    for room in house["rooms"]:
        room_poly = [(corner["x"], corner["z"]) for corner in room["floorPolygon"]]
        #print(room["roomType"] + " # " + str(room["floorPolygon"]))
        #print(room["roomType"] + " ?? " + str(room_poly))
        rooms.append((room["roomType"], room_poly, get_centre_of_the_room(room_poly)))

    return rooms

##
# Return a room floor polygon for the room specifified by the id from the specified habitat
##
def get_room_poly_by_room_id(house, id):
    room_poly = None
    for room in house["rooms"]:
        if room["id"] == "room|" + str(id):
            room_poly = [(corner["x"], corner["z"]) for corner in room["floorPolygon"]]
    return room_poly

def is_full_house(rooms):
    existing_room_names = set()
    for room in rooms:
        rl = room[0].upper()
        if rl == "LIVINGROOM":
            rl = "LIVING ROOM"
        existing_room_names.add(rl)

    return set(RoomType.all_labels()) == existing_room_names

##
# Extract visible objects from a collection of objects
##
def get_visible_objects_from_collection(objects, print_objects = False):
    visible_objects = []

    for obj in objects:
        if obj['visible']:
            if print_objects:
                print(obj['objectType'] + " : " + str(obj['position']))
            visible_objects.append(obj)

    return visible_objects

##
# Extract visible objects (but only their names) from a collection of objects.
# Skip duplicates.
##
def get_visible_object_names_from_collection_set(objects):
    objs_at_this_pos = set()

    vis_objs = get_visible_objects_from_collection(objects)

    for obj in vis_objs:
        objs_at_this_pos.add(obj['objectType'])

    return objs_at_this_pos

def get_all_objects(event_or_controller):
    event = _resolve(event_or_controller)
    thor_objects = thor_get(event, "objects")
    result = []
    for obj in thor_objects:
        if obj["visible"]:
            result.append(obj)
    return result

# Store visible objects in the self.visible_objects collection and print them out if needed
def get_all_objects(event_or_controller, print_objects = False):
    objects = event_or_controller.last_event.metadata['objects']

    if print_objects:
        for obj in objects:
            print(obj['objectType'] + " : " + str(obj['position']))

    return objects

def get_all_objects_of_type(event_or_controller, obj_type_of_interest):
    objects = get_all_objects(event_or_controller)
    objects_of_type = []

    for obj in objects:
        if obj["objectType"] == obj_type_of_interest:
            objects_of_type.append(obj)

    return objects_of_type

def get_objects_of_multiple_types(event_or_controller, obj_types_of_interest):
    objects = get_all_objects(event_or_controller)
    objects_of_type = []

    for obj in objects:
        if obj["objectType"] in obj_types_of_interest:
            objects_of_type.append(obj)

    return objects_of_type

##
# Converts pose from the format of ({'x': 2.25, 'y': 0.9001, 'z': 11.75}, {'x': 0.0, 'y': 225.0, 'z': 0.0})
# into ((2.25, 0.9001, 11.75)(0, 225, 0))
#
# There is a Thortils version of this in agent.thor_pose_as_tuple
##
def convert_pose_set2tuple(pose_as_set):
    return ((pose_as_set[0]['x'], pose_as_set[0]['y'], pose_as_set[0]['z']),
            (pose_as_set[1]['x'], pose_as_set[1]['y'], pose_as_set[1]['z']))

def thor_pose_as_tuple(pose_or_component):
    """
    Returns tuple representation of given pose
    or pose component (position or rotation).
    """
    if type(pose_or_component) == tuple:
        position, rotation = pose_or_component
        return (position["x"], position["y"], position["z"]),\
            (rotation["x"], rotation["y"], rotation["z"])
    else:
        return (pose_or_component["x"],
                pose_or_component["y"],
                pose_or_component["z"])

##
# Calculates a cost (or length) of a path.
##
def get_path_length_old(path, current_pose):
    # A path is a list of tuples. Each tuple contains two dictionary.
    # First dictionary is position (e.g. {'x': 2.25, 'y': 0.9001, 'z': 11.75}) where
    # 'x' is the X position and 'z' is the Y position. The 'y' position doesn't change
    # because that is effectively robot's height (or related variable).
    #
    # The second dictionary is rotation (e.g. {'x': 0.0, 'y': 225.0, 'z': 0.0}) where
    # 'x' and 'z' components don't change for our purposes because our robot only does
    # 'y' rotation which corresponds to yaw.
    #
    # So to calculate path length, the proposal is in each step to calculate distances
    # between positions (previous and current) and sum them up. The rotation step needs
    # to be taken into account too. That we could evaluate as yaw angle difference between
    # previous and current rotation in degrees and multiply by 1/180.0. That gives a score
    # between 0 and 1 because no turn should be greater than 180 degrees.
    prev_pos = {'x': current_pose[0][0], 'y': current_pose[0][1], 'z': current_pose[0][2]} #None
    prev_rtn = {'x': current_pose[1][0], 'y': current_pose[1][1], 'z': current_pose[1][2]} #None
    cur_pos = None
    cur_rtn = None
    total_distance = 0
    for i in range(len(path)):
        (cur_pos, cur_rtn) = path[i]
        if prev_pos is not None:
            # Use Pythagoras theorem for step distance evaluation: a^2 + b^2 = c^2. c = sqrt(a^2 + b^2)
            step_distance = ((prev_pos['x'] - cur_pos['x'])**2 + (prev_pos['z'] - cur_pos['z'])**2)**0.5
            rtn_distance = (prev_rtn['y'] - cur_rtn['y'])
            if rtn_distance > 180.0:
                rtn_distance -= 360.0 # If greater than 180, then the true rotation will be less than 180. abs(x - 360.0)
            total_distance += (step_distance + abs(rtn_distance/180.0))
            #total_distance += step_distance

        prev_pos = cur_pos
        prev_rtn = cur_rtn

    return total_distance

def get_path_length(path, current_pose):
    prev_pos = (current_pose[0][0], current_pose[0][1], current_pose[0][2]) #None
    prev_rtn = (current_pose[1][0], current_pose[1][1], current_pose[1][2]) #None
    cur_pos = None
    cur_rtn = None
    total_distance = 0
    for i in range(len(path)):
        (cur_pos, cur_rtn) = path[i]
        cur_pos = thor_pose_as_tuple(cur_pos)
        cur_rtn = thor_pose_as_tuple(cur_rtn)
        if prev_pos is not None:
            # Use Pythagoras theorem for step distance evaluation: a^2 + b^2 = c^2. c = sqrt(a^2 + b^2)
            #step_distance = ((prev_pos['x'] - cur_pos['x'])**2 + (prev_pos['z'] - cur_pos['z'])**2)**0.5
            step_distance = euclidean_dist(prev_pos, cur_pos)
            rtn_distance = euclidean_dist(prev_rtn, cur_rtn)
            if rtn_distance > 180.0:
                rtn_distance -= 360.0 # If greater than 180, then the true rotation will be less than 180. abs(x - 360.0)
            total_distance += (step_distance + abs(rtn_distance/180.0)) # this gives a 0.25 rotation distance for 45 degrees
            #total_distance += step_distance
            #print(step_distance, " + ", rtn_distance, " = ", abs(rtn_distance/180.0))

        prev_pos = cur_pos
        prev_rtn = cur_rtn

    #print("total_distance: ", total_distance)
    return total_distance

def normalize_colors(frame):
    """Correct the color balance of third-party camera frames to match agent camera"""
    # Convert BGR to RGB if necessary (depends on how you're displaying the images)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Apply color correction - this is a simple adjustment
        # You may need to fine-tune these values
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=0)

        # Adjust color channels individually if needed
        # Reduce blue channel intensity
        b, g, r = cv2.split(frame)
        b = cv2.convertScaleAbs(b, alpha=0.85, beta=0)  # Reduce blue channel
        frame = cv2.merge([b, g, r])

    return frame

##
# From the max and min of the habitat coordinates we can generate full grid of the habitat.
# Later we can infer unreachable positions from this and reachable positions.
##
def create_full_grid_from_room_layout(rooms_in_habitat, step = 0.25):
    #print("AE, rooms_in_habitat: ", rooms_in_habitat)
    room_coords = [] # here we will store coordinates of every corner of each room
    [room_coords.extend(r[1]) for r in rooms_in_habitat]
    zf = lambda x: zip(*x) # this will allow to turn the tuples of coordinates into two lists - X and Y coordinates lists.
    [x_coords, y_coords] = zf(room_coords) # get the two lists
    # get the max and min coordinates from each list
    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x_coords)
    max_y = max(y_coords)

    # Create the grid coordinates
    x_coords = np.arange(min_x, max_x + step, step)
    y_coords = np.arange(min_y, max_y + step, step)

    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)

    # Create list of (x, y) tuples
    all_positions = list(zip(X.flatten(), Y.flatten()))
    return all_positions

def add_buffer_to_unreachable(reachable_points, all_grid_points, step=0.25, buffer_size=1):
    """
    Add buffer around unreachable positions using grid-based approach.

    Parameters:
    reachable_positions: list of (x, y) tuples from AI2-THOR
    all_grid_points: full list of all (x, y) tuples including both reachable and unreachable
    step: grid step size
    buffer_size: number of grid cells to buffer (default: 1 cell = 0.25m)
    """

    # Find unreachable positions
    unreachable = all_grid_points - reachable_points

    # Add buffer around unreachable positions
    buffered_unreachable = set(unreachable)  # Start with original unreachable

    # Define neighbor directions (4-connected or 8-connected)
    directions_4 = [(0, step), (0, -step), (step, 0), (-step, 0)]
    directions_8 = directions_4 + [(step, step), (step, -step), (-step, step), (-step, -step)]

    # Add buffer layers
    for _ in range(buffer_size):
        new_buffer = set()
        for point in buffered_unreachable:
            x, z = point
            for dx, dz in directions_8:  # Use 8-connected for better coverage
                neighbor = (round(x + dx, 2), round(z + dz, 2))
                if neighbor in all_grid_points:
                    new_buffer.add(neighbor)
        buffered_unreachable.update(new_buffer)

    # Final safe positions are all grid points minus buffered unreachable
    safe_positions = all_grid_points - buffered_unreachable

    return safe_positions, buffered_unreachable

# Define actions that we have to move around the agent
action_mapping = {
    'RotateLeft': 0,
    'RotateRight': 1,
    'MoveAhead': 2,
    'STOP': 3,
    # Add more actions as needed
}

# An inverted action_mapping dictionary
inverted_action_mapping = {v: k for k, v in action_mapping.items()}

##
# Easily look up action of the given index
##
def index_to_action(index):
    return inverted_action_mapping.get(index, 'NONE')

##
# Easily look up the index of the given action
##
def action_to_index(action):
    """
    Helper function to convert action strings to indices.
    You can customize this based on your specific actions.
    """
    return action_mapping.get(action, -1)  # Return -1 for unknown actions
