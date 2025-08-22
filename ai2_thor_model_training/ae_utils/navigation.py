from thortils.utils import PriorityQueue, normalize_angles, euclidean_dist
from thortils.navigation import _round_pose
from enum import Enum
from . import angle_to_turn_to_face_p2_from_p1, room_this_point_belongs_to, get_all_objects_of_type, is_point_inside_room_ground_truth
from shapely.geometry import Point
import math

class AStarNode:
    '''
    Imagine a 2D grid:
    .---.---.---.---.
    !___!___!___!___!
    !___!___!___!___!
    !___!___!___!___!
    !___!___!___!___!
    Every dot (even those under exclamation marks) is a node. All dashes and underscores are boundaries of cells in the
    grid.
    One of those nodes will be our starting point and another one will be destination. We will need to find a path between
    them. We will model these nodes with the AStarNode class. Each of them will have f, g and h values (A* algorithm
    vocabulary) and they mean:
    h: Heuristic (an estimate) of the path length from here to the destination.
    g: Measured shortest known distance from the start point to here.
    f: f = g + h
    '''
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
        return self.x == other.x and self.y == other.y and self.yaw == other.yaw

    def __hash__(self):
        return hash((self.x, self.y, self.yaw))

    def get_xy(self):
        return (self.x, self.y)

    def get_yaw(self):
        return self.yaw

    def get_xyr(self):
        return (self.x, self.y, self.yaw)

    def get_ai2thor_pose(self):
        return (self.x, 0.9009993672370911, self.y)

    def get_ai2thor_pose_and_rtn(self):
        return ((self.x, 0.9009993672370911, self.y), (0, self.yaw, 0))

class NavigationAction(Enum):
    '''
    Imagine the same grid that is in the comments of AStarNode class:
    .---.---.---.---.
    !___NW__N___NE__!
    !___W___a___E___!
    !___SW__S___SE__!
    !___!___!___!___!
    But this time we are the node labelled "a". We can move in 8 different directions: N, S, E, W, NE, NW, SE, SW.
    Each such movement will update x or y or both coordinates. In the case of grid_size = 0.25 in AI2-Thor, it will
    update our coordinates by 0.25 one way or another.

    Cost of such movements: Obviously if we are facing North and then move North, then we incurred 1 action, so a cost of 1.
    But we might be facing South West and wanting to go in the North East direction. In such case the action that we will
    want to apply will be a rotation action (with cost still 1), but we will do it several times because we will have to
    turn from SW->W, then W->NW, then NW->N and finally N->NE and then we will need to move. That's 5 actions in total,
    but that's handled outside of here. The moral of the story is that each action costs 1.

    This is how it would work under normal circumstances (where point (0, 0) is the top left corner), but AI2-Thor has
    a peculiarity: point (0, 0) is the left lower corner. Therefore we need to change the coordinate changes for each
    direction. SOUTH becomes (0, -0.25), NORTH becomes (0, 0.25), SW becomes (-0.25, -0.25), NW becomes (-0.25, 0.25),
    SE becomes (0.25, -0.25) and NE becomes (0.25, 0.25).

    Finally, when we move diagonally, we don't really advance by 0.25 in two directions at once, e.g. when we move NW,
    we do not advance towards N by 0.25 and towards W by 0.25. We only move in a straight line by 0.25. So if we got N,
    W, S or E, then we move 0.25 in that direction, but if we move diagonally, then we only move 0.176776885986328. However,
    if we do that, then we will end up somewhere that is between grid nodes and it will be hard to validate whether it is
    a reachable position or not. Therefore we still need to advance by 0.25 in both directions, but reflect the cost appropriately.
    '''
    MOVE_NORTH = (0, 0.25, 0, 1) #(0, -0.25, 0)
    MOVE_SOUTH = (0, -0.25, 180, 1) #(0, 0.25, 180)
    MOVE_EAST = (0.25, 0, 90, 1)
    MOVE_WEST = (-0.25, 0, 270, 1)
    MOVE_NORTHEAST = (0.25, 0.25, 45, 1.414213562) #(0.25, -0.25, 45)
    MOVE_NORTHWEST = (-0.25, 0.25, 315, 1.414213562) #(-0.25, -0.25, 315)
    MOVE_SOUTHEAST = (0.25, -0.25, 135, 1.414213562) #(0.25, 0.25, 135)
    MOVE_SOUTHWEST = (-0.25, -0.25, 225, 1.414213562) #(-0.25, 0.25, 225)
    TURN_LEFT = (0, 0, -45, 1)
    TURN_RIGHT = (0, 0, 45, 1)

    def __init__(self, dx, dy, yaw, cost_of_move):
        self._dx = dx
        self._dy = dy
        # The yaw is either delta_yaw (change of yaw) if it's a turning action or an absolute yaw if it is a moving action.
        self._yaw = yaw
        self._cost_of_move = cost_of_move

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def yaw(self):
        return self._yaw

    @classmethod
    def valid_actions(cls, current_pose_yaw):
        ret_set = {action for action in cls if action.yaw == current_pose_yaw}
        ret_set.add(NavigationAction.TURN_RIGHT)
        ret_set.add(NavigationAction.TURN_LEFT)
        return ret_set

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
        #print("AE: full_pose: ", full_pose)
        # now that new cost has been calculated, we can assign the new yaw to the pose fields.
        # If we're turning, then yaw will change by the specified value,
        # if not, then it should be the same as before and also equal to the yaw of the specified value.
        if self in [NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT]:
            new_yaw = NavigationUtils.normalize_yaw(full_pose[1][1] + self._yaw)
        else:
            new_yaw = self._yaw
            #print("AE: new_yaw == full_pose[1][1]: ", new_yaw, full_pose[1][1], self.name)
            assert(new_yaw == full_pose[1][1])
        new_full_pose = ((new_x, full_pose[0][1], new_y), (0.0, new_yaw, 0.0))
        # Cost of this move will always be as defined in the constructor, 1 - for the movement where only one coordinate
        # changes or when turning. Or 1.414 for a diagonal move.
        # Therefore the g value for the new node will be old g value + self._cost_of_move
        new_node = AStarNode(new_full_pose, node.g + self._cost_of_move,
                                        euclidean_dist(new_full_pose[0], destination[0]), node)

        return new_node

##
# I want to implement a "scoring circle". Imagine circular zones (1 cell wide each) around the target point.
# The optimal A* path would usually lead through all the zones right to the target starting at the outermost
# and ending with the innermost and then the target. Usually the path would not go back to a zone that has
# already been visited. I say *usually* because sometimes the obstacles in the scene would require to re-visit
# a zone that has already been visited, but for our purposes (reaching the middle of the room) such cases
# would be relatively rare. Now, the inferred path may not be as efficient and could potentially snake back
# to where we have already been. We want to prevent that, so we should give a reward point for reaching a new
# zone, but a penalty for reaching a visited one.
#
##
class ScoringCircle:
    def __init__(self, center_node):
        self.center_node = center_node


class NavigationUtils:
    '''
    Here we put it all together- We use the A* algorithm to do something useful. Initially just getting the path cost
    from start point to destination.
    '''

    def __init__(self):
        # Last path generated
        self.last_path_gen = None

    def get_last_path_and_params(self):
        return (self.last_path_gen,
                self.reachable_positions,
                self.start_point,
                self.destination)

    ##
    # This will help find the angle required to turn to face the target.
    # target_pos: position that we want to face (just x and y)
    # our_pos: our current position (x, y and rotation)
    # returns the angle required to turn from current position
    ##
    def angle_to_face_target(self, target_pos, our_pos):
        # Define the coordinates of the two points
        (Tx, Ty) = target_pos # x, y
        (Ox, Oy, Or) = our_pos # x, y and yaw
        # Calculate the vector components from point 1 to point 2
        dx = Tx - Ox
        dy = Ty - Oy
        # Calculate the angle using atan2
        angle_to_face_point2 = math.atan2(dx, dy)
        # Convert the angle from radians to degrees
        deg_to_turn_from_N = math.degrees(angle_to_face_point2)

        # At this point we have a value that we need to be facing. If we are now looking at N, then
        # we can turn by deg_to_turn_from_N and be where we want to be. But what about our current rotation?
        # How much we need to turn by from the current yaw?
        deg_to_turn_from_current = deg_to_turn_from_N - Or

        # E.g., if we're going to turn -181, then it's easier to turn +179, so we do (x % 360)
        # If we're +361, then (x % 360) will return +1.
        while deg_to_turn_from_current >= 180: deg_to_turn_from_current = deg_to_turn_from_current - 360
        while deg_to_turn_from_current <= -180: deg_to_turn_from_current += 360
        while deg_to_turn_from_N >= 180: deg_to_turn_from_N = deg_to_turn_from_N - 360
        while deg_to_turn_from_N <= -180: deg_to_turn_from_N += 360

        print("Tx, Ty:", Tx, Ty, " Ox, Oy, Or: ", Ox, Oy, Or, " deg_to_turn_from_current: ", deg_to_turn_from_current, " deg_to_turn_from_N: ", deg_to_turn_from_N)

        return deg_to_turn_from_current, deg_to_turn_from_N

    ##
    # start_point:  where we start (start_position, start_rotation)
    # target_point: where we want to get to (target_position, target_rotation)
    # reachable_positions: Positions that are possible to reach (no objects are sitting in those places)
    # close_enough: how close is enough to consider target achieved
    ##
    def get_path_cost_to_target_point(self, start_point, target_point, reachable_positions_in, close_enough = 0.5):
        destination = ((target_point.x, 0.9009993672370911, target_point.y),
                       start_point[1])
        #print("AE: start_point: ", start_point, " target_point: ", target_point)
        # the defined 2D coordinates and same rotation as start position
        # Normalize angles in start and goal to be within 0 to 360 (see top comments)
        # Also, round the poses so that we don't have irrational numbers in them that would be hard to look up
        # e.g. (10.75, 8.25) instead of (10.86666666, 8.3333333333).
        # Also angles need to be discrete values in [0, 45, 90, 135, 180, 225, 270, 315]
        start_point = _round_pose((start_point[0], normalize_angles(start_point[1])))
        destination = _round_pose((destination[0], normalize_angles(destination[1])))
        start_point = self.normalize_to_grid(start_point)
        destination = self.normalize_to_grid(destination)
        # positions that we can reach
        reachable_positions = set(reachable_positions_in)
        # Save the search parameters in case we want to visualize the path later
        self.reachable_positions = reachable_positions
        self.start_point = start_point
        self.destination = destination
        # The priority queue. We will keep poses in it with the estimates of their distances to the goal stored as priorities.
        worklist = PriorityQueue()
        # First pose will be the start and the estimate to the goal is its priority.
        # Make sure that all poses in the worklist and elsewhere are rounded though,
        # else they may not match later when we look them up and lead to no path found.
        start_node = AStarNode(start_point, h=euclidean_dist(start_point[0], destination[0]))
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
            if euclidean_dist(destination[0], current_node.get_ai2thor_pose()) <= close_enough:
                best_cost = cost[current_node.get_ai2thor_pose_and_rtn()]
                (angle_req, deg_to_turn) = self.angle_to_face_target((target_point.x, target_point.y), current_node.get_xyr())
                print("C_yaw: ", current_node.get_ai2thor_pose_and_rtn()[1][1], " angle_req: ", angle_req)
                angle_req = self.normalize_yaw(angle_req)
                #print("C_yaw: ", current_node.get_ai2thor_pose_and_rtn()[1][1], " angle_req: ", angle_req)
                # here we will store our path
                self.last_path_gen = []
                # Now work it back
                while current_node.parent:
                    self.last_path_gen.append(current_node.get_xyr())
                    last_pose_in_path = current_node.get_xyr()
                    current_node = current_node.parent

                # Reverse it because we started at the destination when working our way back
                self.last_path_gen.reverse()
                return best_cost

            # AE: Look at all defined actions and try each of them from the current pose and see what happens
            #print("ae: current_node.get_yaw() : ", current_node.get_yaw(), " self.normalize_yaw(current_node.get_yaw()): ", self.normalize_yaw(current_node.get_yaw()))
            for action in NavigationAction.valid_actions(self.normalize_yaw(current_node.get_yaw())):
                nx, ny = action.apply(current_node.x, current_node.y)
                # If we end up in a legal place, then generate a new node and add it to the priority queue
                if (nx, ny) in reachable_positions:
                    # generate a new node from this action. There may be different nodes for the same location
                    # on the grid because they may have different yaw rotations and even different parents.
                    # So in the worst case there can be a node for <each grid location> * <all possible yaw rotations> * <each grid location as a parent>
                    #print("AE: current_node: ", current_node.get_yaw(), " destination: ", destination)
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

    ##
    # Finds the next door to navigate to given the current position
    ##
    def find_door_target(self, current_point_and_rtn, rooms_in_habitat, reachable_positions, controller):
        point_for_room_search = (current_point_and_rtn[0], "", current_point_and_rtn[1])
        cur_pos = ((current_point_and_rtn[0], 0.9009993672370911, current_point_and_rtn[1]),
                   (0.0, float(current_point_and_rtn[2]), 0.0))
        # cur_pos = self.rnc.get_agent_pos_and_rotation()
        # print("cur_pos: ", cur_pos, "cur_pos2: ", cur_pos2)
        # This is the room where we are
        room_of_placement = room_this_point_belongs_to(rooms_in_habitat, point_for_room_search)
        print(room_of_placement)

        doors = get_all_objects_of_type(controller, "Doorway")
        # the target is not really the door, but a point in front of the door, so we will need to extract those.
        # we will also want to know if the door is visible and if it is in the same room as we are
        all_door_targets = []
        selected_target = None

        # print(doors[0])
        for door in doors:
            # find the centre of the door because we will want to arrive at the centre of the door, not the edge of
            # the frame
            door_center_pos = door["axisAlignedBoundingBox"]["center"]
            #print("corners: ", door["axisAlignedBoundingBox"]["cornerPoints"])
            #print("size: ", door["axisAlignedBoundingBox"]["size"])
            #print("rotation: ", door["rotation"])
            #print("isOpen: ", door["isOpen"])

            # TODO: Use angle_to_turn_to_face_p2_from_p1 from ai2_thor_utils.py and incorporate it into the
            # path planning so that at the end it turns to face the door.

            target_position_tuple = (door_center_pos['x'], door_center_pos['y'], door_center_pos['z'])
            target_position = {"x": door_center_pos['x'], "y": door_center_pos['y'], "z": door_center_pos['z']}
            target_position_point = Point(door_center_pos['x'], door_center_pos['z'])
            is_in_same_room = is_point_inside_room_ground_truth(target_position_tuple, room_of_placement[1])

            # the distance to that point as A* goes
            try:
                door_path_length = self.get_path_cost_to_target_point(cur_pos,
                                                                         target_position_point,
                                                                         reachable_positions)
            except ValueError as e:
                door_path_length = 1000

            # Ignore doors to which path could not be planned
            if door_path_length < 1000:
                all_door_targets.append({"pos": target_position,
                                         "in_same_room": is_in_same_room,
                                         "visible": door['visible'],
                                         "distance": door_path_length})
        #print("all_door_targets: ", all_door_targets)

        # Ideally we want to get a door that is in the same room, but not yet visible, because we want to elicit
        # the behaviour of seeking the door in our model.
        # If that's not possible then at least a visible door in the same room where we are.
        # If that's also not possible, then the nearest door.
        same_room_invisible = [door_target for door_target in all_door_targets if
                               door_target["in_same_room"] and not door_target["visible"]]

        same_room_visible = [door_target for door_target in all_door_targets if
                             door_target["in_same_room"] and door_target["visible"]]

        same_room_invisible = sorted(same_room_invisible, key=lambda room_tuple: room_tuple["distance"])
        same_room_visible = sorted(same_room_visible, key=lambda room_tuple: room_tuple["distance"])
        all_rooms_sorted_by_distance = sorted(all_door_targets, key=lambda room_tuple: room_tuple["distance"])

        # print("same_room_invisible: ", same_room_invisible)
        # print("same_room_visible: ", same_room_visible)
        # print("all_rooms_sorted_by_distance: ", all_rooms_sorted_by_distance)

        if len(same_room_invisible) > 0:
            selected_target = same_room_invisible[0]
        elif len(same_room_visible) > 0:
            selected_target = same_room_visible[0]
        elif len(all_rooms_sorted_by_distance) > 0:
            selected_target = all_rooms_sorted_by_distance[0]

        if selected_target is not None:
            return Point(selected_target["pos"]["x"], selected_target["pos"]["z"])
        else:
            raise ValueError("No door found that can be navigated to")

    ##
    # Normalize any input coordinate (e.g. (10.86666, 8.3333)) to the nearest valid grid point (e.g. (10.75, 8.25)).
    ##
    def normalize_to_grid(self, pose, step=0.25):
        location = pose[0]
        rotation = pose[1]
        return ((self.round_to_step(location[0], step), self.round_to_step(location[1], step), self.round_to_step(location[2], step)),
                (0.0, self.normalize_yaw(rotation[1]), 0.0))

    def round_to_step(self, val, step=0.25):
        return round(val / step) * step

    @staticmethod
    def normalize_yaw(yaw):
        """
        Normalize yaw angle to the nearest multiple of 45 in [0, 315].
        Input yaw can be any real number (positive or negative).
        """
        yaw = yaw % 360  # Bring into [0, 360)
        return round(yaw / 45) * 45 % 360

