from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from room_type import RoomType
import math, cv2
from thortils.utils.math import (euclidean_dist, to_deg)
from thortils.agent import thor_pose_as_tuple

##
# My own utilities functions for AI2-THOR. I couldn't find analogous functions in Thortils,
# so I wrote my own here. Eventually some of them should probably be pushed to Thortils
# project.
##
class AI2THORUtils:
    def __init__(self):
        pass

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

##
# Converts pose from the format of ({'x': 2.25, 'y': 0.9001, 'z': 11.75}, {'x': 0.0, 'y': 225.0, 'z': 0.0})
# into ((2.25, 0.9001, 11.75)(0, 225, 0))
#
# There is a Thortils version of this in agent.thor_pose_as_tuple
##
def convert_pose_set2tuple(pose_as_set):
    return ((pose_as_set[0]['x'], pose_as_set[0]['y'], pose_as_set[0]['z']),
            (pose_as_set[1]['x'], pose_as_set[1]['y'], pose_as_set[1]['z']))

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
