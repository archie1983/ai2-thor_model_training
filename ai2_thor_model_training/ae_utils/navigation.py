from thortils.utils import PriorityQueue, normalize_angles, euclidean_dist
from thortils.navigation import _round_pose
from enum import Enum

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
    But we might be facing South West and wanting to go in the North East direction. In such case the cost incurred will be 5,
    because we will have to turn from SW->W, then W->NW, then NW->N and finally N->NE and then we will need to move. That's
    5 actions in total.
    '''
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
        new_node = AStarNode(new_full_pose, node.g + new_cost,
                                        euclidean_dist(new_full_pose[0], destination[0]), node)

        return new_node

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
    # start_point:  where we start (start_position, start_rotation)
    # target_point: where we want to get to (target_position, target_rotation)
    # reachable_positions: Positions that are possible to reach (no objects are sitting in those places)
    ##
    def get_path_cost_to_target_point(self, start_point, target_point, reachable_positions_in):
        destination = ((target_point.x, 0.9009993672370911, target_point.y),
                       start_point[1])
        print("AE: start_point: ", start_point, " target_point: ", target_point)
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
            if euclidean_dist(destination[0], current_node.get_ai2thor_pose()) <= 0.5:
                best_cost = cost[current_node.get_ai2thor_pose_and_rtn()]
                # here we will store our path
                self.last_path_gen = []
                # last pose in path - we need this to detect turns. When we start, it will be None, but for all other
                # nodes it will be the predecessor - the one we were at before we followed the "parent" link.
                last_pose_in_path = None
                # Now work it back
                while current_node.parent:
                    if last_pose_in_path:
                        (_, _, dest) = last_pose_in_path # This is closer to the destination
                        (_, _, start) = current_node.get_xyr() # This is further from the destination
                        # Normally we would use here: turning_deg_required = (dest - start + 180) % 360 - 180
                        # But because we are working our way backwards, this needs to be a reversed list of turning
                        # steps. Therefore, swap dest and start.
                        turning_deg_required = (start - dest + 180) % 360 - 180
                        turn_direction = -1 if turning_deg_required < 0 else 1
                        print("turning_deg_required: ", turning_deg_required)
                        # We turn in 45 degree increments, so this many turns we will need
                        turns_required = abs(turning_deg_required) // 45
                        old_yaw = dest
                        print("turns_required: ", turns_required, " dest: ", dest, " start: ", start)
                        for i in range(turns_required):
                            new_yaw = old_yaw + turn_direction * 45
                            self.last_path_gen.append((*current_node.get_xy(), new_yaw))
                            old_yaw = new_yaw

                    self.last_path_gen.append(current_node.get_xyr())
                    last_pose_in_path = current_node.get_xyr()
                    current_node = current_node.parent

                # Reverse it because we started at the destination when working our way back
                self.last_path_gen.reverse()
                return best_cost

            # AE: Look at all defined actions and try each of them from the current pose and see what happens
            for action in NavigationAction:
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

    def normalize_yaw(self, yaw):
        """
        Normalize yaw angle to the nearest multiple of 45 in [0, 315].
        Input yaw can be any real number (positive or negative).
        """
        yaw = yaw % 360  # Bring into [0, 360)
        return round(yaw / 45) * 45 % 360

