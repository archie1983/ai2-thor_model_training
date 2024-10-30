from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from room_type import RoomType

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
        rooms.append((room["roomType"], room_poly))

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
# Calculates a cost (or length) of a path.
##
def get_path_length(path, current_pose):
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
