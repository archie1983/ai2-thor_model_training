import math

from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.prepared import prep
from collections import deque
import matplotlib.pyplot as plt
from . import NavigationActions

class BoundaryCalculations:
    def __init__(self, grid_size=0.125):
        self.na = NavigationActions(step = grid_size)

    def get_room_perimeter_points_1st_pass(self, reachable_points, unreachable_points, room_of_placement, na):
        '''
        First pass for room perimeter point gathering. Here we get all points that form a boundary- either the room
        boundary or some object within the room. The idea is that we look at all reachable points and attempt to walk
        from them to any direction. If we reach a point that is unreachable or not in reachable point set, then that
        means we started walking from a boundary point.

        :param reachable_points:
        :param unreachable_points:
        :param room_of_placement:
        :param na:
        :return:
        '''
        room_polygon = prep(Polygon(room_of_placement[1]))

        reachable_room_points = {
            (x, y) for (x, y) in reachable_points if room_polygon.contains(Point(x, y))
        }

        unreachable_room_points = {
            (x, y) for (x, y) in unreachable_points if room_polygon.contains(Point(x, y))
        }

        # Test each reachable point whether it has neighbours that are not reachable
        boundary_points = set()
        for rpos in reachable_room_points:
            for move in na.MOVE_MOVES:
                new_x, new_y = na.apply(rpos[0], rpos[1], move)
                if (new_x, new_y) in unreachable_room_points or (new_x, new_y) not in reachable_room_points:
                #if (new_x, new_y) not in reachable_room_points:
                    boundary_points.add((rpos[0], rpos[1]))
                    break
        return boundary_points

    def get_room_perimeter_points_2nd_pass(self, boundary_points, na):
        '''
        Second pass for room boundary points. The idea is that we take any point in the boundary points from
        get_room_perimeter_points_1st_pass(...) and start walking in all directions. If we reach another point
        from the original boundary points set, then it's the point that belongs to the same boundary, so take it
        out from the starting set and keep going. Once there are no points left to reach, we have established one
        boundary. Now take any point from the remaining set and repeat the same procedure. In the end we should
        have a set of boundaries within the room. Now we form a polygon from each of those sets and check which
        one is within another one. The outer polygon is what we want.

        :param boundary_points:
        :param na:
        :return:
        '''

        boundary_points = boundary_points.copy()
        all_compacted_sub_boundaries = list()
        all_explored_sub_boundaries = list()
        current_sub_boundary = list()
        neighbours_found = 0
        crossroads = deque() # stack for crossroad points
        cbp = None
        cbp_prev = None
        cbp_prev_prev = None
        cbp_next = None
        seen_4_cross = False
        print_debug = False

        # when we visit vertices and build our paths, we may discover disconnected graphs, where a path can be formed,
        # but it only visits a subset of vertices. If that is the case, then we want to re-run the algorithm and choose
        # the starting point from the un-visited vertices. For that we will need to track the visisted vertices.
        visited_points = set()

        #cbp = boundary_points.pop() # take any point as a starter
        # cbp is "current boundary point"
        while boundary_points != visited_points:
            points_to_choose_from = boundary_points - visited_points
            for cbp in points_to_choose_from:
                break

            while cbp:
                #breakpoint()
                # add current point to the current boundary
                current_sub_boundary.append(cbp)
                visited_points.add(cbp)
                neighbours_found = 0
                next_points = []
                # now move forward until we see either a visited point or a crossroads (more than 2 valid paths from here)
                for move in na.MOVE_MOVES: # walk in all directions from current point until we find another point from the boundary or exhaust all moves
                    new_x, new_y = na.apply(cbp[0], cbp[1], move)
                    # count how many other boundary points we can see from this one

                    # if current_sub_boundary + [(new_x, new_y)] in all_explored_sub_boundaries:
                    #     print("FILTERD SUB Boundary")

                    if ((new_x, new_y) in boundary_points and # only proceed if the point is whithin accessible points
                        cbp_prev != (new_x, new_y) and # and we're not going backwards
                        cbp_prev_prev != (new_x, new_y) and # to avoid triangular paths around every 90 degree corner
                        not current_sub_boundary + [(new_x, new_y)] in all_explored_sub_boundaries): # and we haven't seen this kind of path before (to avoid cyclic travelling)
                        neighbours_found += 1
                        next_points.append((new_x, new_y))
                        if neighbours_found == 1:
                            # The first neighbour that we find will be the regular one to explore
                            cbp_next = (new_x, new_y)
                            #print("reg w: ", cbp, " -> ", (new_x, new_y))
                        else:
                            # if there are more, then store them as directions in crossroads
                            crossroads.append(((new_x, new_y), cbp, cbp_prev, current_sub_boundary.copy()))
                            #print("cross: ", cbp, " -> ", (new_x, new_y))

                            #print("len(crossroads): ", len(crossroads), "cbp: ", cbp, "cbp_prev: ", cbp_prev, "(new_x, new_y): ", (new_x, new_y))
                if print_debug:
                    print("next_points: ", next_points)

                # If we have 1 neighbour, then cbp is an end part of an unconnected boundary. We're not interested int this kind of path,
                # purge it.
                if neighbours_found < 1:
                    #print("TEST current_sub_boundary: ", current_sub_boundary, "cbp: ", cbp, "cbp_prev: ", cbp_prev, "cbp_next: ", cbp_next)
                    #breakpoint()
                    cbp = None
                    if len(crossroads) > 0:
                        cbp, cbp_prev, cbp_prev_prev, current_sub_boundary = crossroads.pop()
                        #print("exploring cross1: ", cbp_prev, " -> ", cbp, " CSB: ", current_sub_boundary)
                else:
                    cbp_prev_prev = cbp_prev
                    cbp_prev = cbp
                    cbp = cbp_next

                #print(len(current_decision_chain), " @ ", current_decision_chain)
                # see if we've found a closure for the current boundary
                while cbp is not None and cbp in current_sub_boundary:
                    #breakpoint()
                    # if we see cbp already in the current path, then we have completed a loop and current_sub_boundary is a complete sub-boundary
                    cbp_ndx = current_sub_boundary.index(cbp)
                    compacted_sub_boundary = current_sub_boundary[cbp_ndx:]
                    # only add it to the all_sub_boundaries if it is a unique boundary. We are not interested
                    # in boundaries that have different start end end points, but contain the same points
                    s_compacted_sub_boundary = set(compacted_sub_boundary)
                    if all(s_compacted_sub_boundary != set(sb) for sb in all_compacted_sub_boundaries):
                        all_compacted_sub_boundaries.append(compacted_sub_boundary)

                    #breakpoint()
                    # remember a decision chain that we have already explored
                    all_explored_sub_boundaries.append(current_sub_boundary.copy())

                    # if there are more crossroads left, then explore those
                    cbp = None
                    if len(crossroads) > 0:
                        cbp, cbp_prev, cbp_prev_prev, current_sub_boundary = crossroads.pop()
                        #print("exploring cross2: ", cbp_prev, " -> ", cbp, " CSB: ", current_sub_boundary)
                        #crossroad_decision_chains_explored.append(discovered_vectors)

            all_compacted_sub_boundaries = [sb for sb in all_compacted_sub_boundaries if len(sb) > 2]
            all_compacted_sub_boundaries = sorted(all_compacted_sub_boundaries, key=lambda boundary: Polygon(boundary).area)
            # for sb in all_sub_boundaries:
            #     print("SB: ", sb)
            #print("len(crossroads) at END: ", len(crossroads))
        return all_compacted_sub_boundaries

    def visualize(self, collection_to_visualize, collection_to_visualize2 = None):
        # Plot the room boundary and internal obstacles
        plt.figure(figsize=(8, 6))

        # Plot all perimeter points (including obstacles)
        all_perimeter = list(collection_to_visualize)
        plt.scatter([p[0] for p in all_perimeter], [p[1] for p in all_perimeter],
                    c='blue', s=5, alpha=0.5, label='All perimeters')

        plt.plot([p[0] for p in all_perimeter], [p[1] for p in all_perimeter],
                    'b-', c='gray', alpha=0.5, label='All perimeters', linewidth=0.5, zorder=1)

        if collection_to_visualize2 is not None:
            collection_to_visualize2 = list(collection_to_visualize2)
            plt.scatter([p[0] for p in collection_to_visualize2], [p[1] for p in collection_to_visualize2],
                        c='red', s=5, alpha=0.5, label='All perimeters')

        plt.legend()
        plt.axis('equal')
        plt.title('Room Boundary vs Internal Obstacle Perimeters')
        plt.show()

    def remove_redundant_points(self, boundary, na, step=0.125):
        '''
        Redundant points are undesirable, because they create redundant paths. For example
        consider the following 90 degree corner:

             C
             |
        B----A

        If we're at B, then we can get to C either directly, or through point A, following path BAC. That creates
        2 paths that the algorithm needs to consider. If we have another corner like this, then we get 4 paths.
        The third one will give us 8 paths and so on. And that's just corners with 2 possible paths. If we have
        structures with more possible paths (e.g. two sequences of points side by side), then the multiplicatory
        effect becomes even worse. Before you know it, what should have been like 11 boundaries to explore, has
        turned into 100s of thousands or millions and our algorithm hangs because it is processing all the redundant
        paths. That's why we want to remove the redundant points.

        The way we do it, is this:
        1) Take any point and look at its immediate neighbours
        2) For each neighbour, see where we can get by performing just 1 step
        3) See where we can get by performing exactly 2 steps from each original neighbour
        4) The union of vertices acquired in steps (1) and (2) form the points that we want to be still reachable
        within 2 steps or less even after removing a redundant point.
        5) Now remove the point selected at step (1) and repeat steps (2) and (3).
        6) If union of vertices acquired is the same as before, then we can safely leave the point removed.
        7) If there is a difference for 1 or more neighbours, then we have to put that point from step (1) back because it is not redundant.

        We can, of course, optimize further, by performing 3 steps instead of 2 after removal. That way we can also remove
        the point if all vertices are still reachable, but might take an extra 1 step. This leads to better results for
        our purposes here. We could of course carry on and do 4 or more steps, but then the boundary might start
        degenerating beyond what is desired

        :param boundary:
        :param na:
        :param step:
        :return:
        '''
        boundary_copy = boundary.copy()
        to_discard = set()
        # look at each point
        for (x, y) in boundary:
            # get its neighbours
            all_neighbors = {na.apply(x, y, move) for move in na.MOVE_MOVES if na.apply(x, y, move) in boundary_copy}
            this_point_discard_decisions = []
            for n in all_neighbors:
                # see where we can get in 1 step
                step1_points = {na.apply(n[0], n[1], move)
                                        for move in na.MOVE_MOVES
                                        if na.apply(n[0], n[1], move) in boundary_copy}
                step2_points = set()

                # see where we get in 2 steps
                for point in step1_points:
                    # if x == y:
                    #     print("Step2 update at ", point, " : ", {na.apply(point[0], point[1], move)
                    #                     for move in na.MOVE_MOVES
                    #                     if na.apply(point[0], point[1], move) in boundary})
                    step2_points.update({na.apply(point[0], point[1], move)
                                    for move in na.MOVE_MOVES
                                    if na.apply(point[0], point[1], move) in boundary_copy})

                # discard the point
                boundary_copy.discard((x, y))
                # and repeat the 1 step and 2 step experiments
                step1_points_after_discard = {na.apply(n[0], n[1], move)
                                for move in na.MOVE_MOVES
                                if na.apply(n[0], n[1], move) in boundary_copy}

                step2_points_after_discard = set()

                for point in step1_points_after_discard:
                    step2_points_after_discard.update({na.apply(point[0], point[1], move)
                                    for move in na.MOVE_MOVES
                                    if na.apply(point[0], point[1], move) in boundary_copy})

                step3_points_after_discard = set()
                # and do 3 steps too
                for point in step2_points_after_discard:
                    step3_points_after_discard.update({na.apply(point[0], point[1], move)
                                    for move in na.MOVE_MOVES
                                    if na.apply(point[0], point[1], move) in boundary_copy})

                # if x==y:
                #     print("(x, y)", (x, y))
                #     print("step1_points: ", step1_points)
                #     print("step2_points: ", step2_points)
                #
                #     print("step1_points_after_discard: ", step1_points_after_discard)
                #     print("step2_points_after_discard: ", step2_points_after_discard)

                # vertices reachable before and after discarding the selected point
                before_discard = step1_points.union(step2_points) - {(x, y)}
                after_discard = step1_points_after_discard.union(step2_points_after_discard).union(step3_points_after_discard)

                # if after the removal we can reach all the same vertices in 3 steps
                # or less that we could previously reach in 2 or less, then this neighbour is not affected
                # significantly by removing the point from step (1)
                if after_discard.intersection(before_discard) == before_discard:
                    #to_discard.add((x, y))
                    #print("discarded: ", (x, y))
                    this_point_discard_decisions.append(True)
                else:
                    this_point_discard_decisions.append(False)
                    #boundary_copy.add((x, y))
                    break

            # If all neighbours are not significantly affected, then remember that the point in question
            # can be removed and leave it removed from the experimental boundary that we will carry on
            # working with, otherwise reinstate it.
            if all(this_point_discard_decisions):
                to_discard.add((x, y))
            else:
                boundary_copy.add((x, y))
        #print("to_discard: ", to_discard)

        return boundary - to_discard

    def find_room_perimeter_path(self, reachable_room_points, unreachable_room_points, room_of_placement, house):
        # First let's establish the boundaries between walkable and non-walkable locations in the room
        expanded_poly = self.expand_room_poly_if_needed(room_of_placement, house)
        room_of_placement = (room_of_placement[0], expanded_poly, room_of_placement[2])

        boundary_points = self.get_room_perimeter_points_1st_pass(reachable_room_points, unreachable_room_points,
                                                                room_of_placement, self.na)

        # Remove redundant points
        boundary_points = self.remove_redundant_points(boundary_points, self.na)

        # Finally, take the optimized bounadries and find the biggest polygon that can be detected between them-
        # that's the floor perimeter that we want.
        separated_boundaries = self.get_room_perimeter_points_2nd_pass(boundary_points, self.na)

        return separated_boundaries[-1]

    def get_room_polys(self, house):
        '''
        Returns polygons of all rooms' floors in a house
        :param house:
        :return:
        '''
        room_polys = []
        for room in house["rooms"]:
            room_poly = [(corner["x"], corner["z"]) for corner in room["floorPolygon"]]
            print(room["roomType"] + ": " + str(room_poly))
            room_polys.append(room_poly)
        return room_polys

    def is_wallless_neighbours(self, room_poly1, room_poly2, all_walls):
        neighbours, common_vertices = self.is_neighbours(room_poly1, room_poly2)
        if neighbours:
            # test if there is no wall between rooms. Sometimes though there are rooms which are
            # separated by a short opening, which I would prefer to treat as open door rather than
            # a missing wall. Therefore if the opening is less than 2m, then that will be treated
            # as a door rather than a missing wall.
            if (not self.is_wall_between_rooms(common_vertices, all_walls) and
                self.euclidean_dist(common_vertices[0], common_vertices[1]) > 2.0):
                #print("NO WALL", self.euclidean_dist(common_vertices[0], common_vertices[1]))
                #print("Neighbours: ", room_poly1, " and ", room_poly2, " common vertices: ", common_vertices)
                return True
        return False

    def expand_room_poly_if_needed(self, room_of_placement, house):
        all_room_polys = self.get_room_polys(house)
        all_walls = self.get_house_walls(house)
        this_room_poly = room_of_placement[1]
        resulting_poly = this_room_poly

        for rp in all_room_polys:
            if self.is_wallless_neighbours(this_room_poly, rp, all_walls):
                print("walless neighbour room: ", rp)
                #resulting_poly.extend(rp)
                resulting_poly = unary_union([Polygon(this_room_poly), Polygon(rp)])
                #resulting_poly = rp
                break

        return resulting_poly

    def is_neighbours(self, room_poly1, room_poly2):
        '''
        Tests if two room polygons share a border
        :param room_poly1:
        :param room_poly2:
        :return:
        '''
        common_vertices = []
        # if the same polygons, then return False
        if room_poly2 == room_poly1:
            return False, []

        # otherwise test corners
        for corner in room_poly1:
            if corner in room_poly2:
                common_vertices.append(corner)
        return len(common_vertices) > 1, common_vertices

    def get_house_walls(self, house):
        '''
        Returns all wall lines in a house
        :param house:
        :return:
        '''
        all_walls = []
        for wall in house["walls"]:
            wall_line = {(corner["x"], corner["z"]) for corner in wall["polygon"]}
            empty = False
            if "empty" in wall and wall["empty"]:
                empty = True
            #print(wall["roomId"] + " : " + wall["id"] + " : " + str(wall_line), " : EMPTY = ", empty)
            all_walls.append((wall_line, empty))
        return all_walls

    def is_wall_between_rooms(self, common_vertices, all_walls):
        '''
        Tests if a wall exists between two rooms that share a border.
        :param common_vertices:
        :param all_walls:
        :return:
        '''
        for w, is_empty in all_walls:
            #print("w: ", w, " common_vertices: ", common_vertices)
            if w == set(common_vertices) and not is_empty:
                return True
        return False

    def euclidean_dist(self, p1, p2):
        if len(p1) > 2:
            p1 = (p1[0], p1[2])
        if len(p2) > 2:
            p2 = (p2[0], p2[2])
        return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

    def find_furthest_boundary_point(self, agent_pos, heading_angle, boundary_polygon, max_distance=10.0):
        """
        Find the furthest boundary point in the direction the agent is looking.
        This is useful for perimeter navigation where we want to go to the outer wall,
        not obstacles in the way.
        """
        dx = math.cos(heading_angle)
        dy = math.sin(heading_angle)

        # Sample points along the ray at small intervals
        step = 0.05  # Small step size for accuracy
        num_steps = int(max_distance / step)

        boundary_points = list(boundary_polygon.exterior.coords)[:-1]
        boundary_set = set(boundary_points)

        furthest_point = None
        furthest_distance = -1

        for i in range(num_steps):
            distance = i * step
            x = agent_pos[0] + dx * distance
            y = agent_pos[1] + dy * distance

            # Round to grid resolution to match boundary points
            x_rounded = round(x, 2)
            y_rounded = round(y, 2)

            # check if we are sufficiently close to a boundary point
            if min([self.euclidean_dist(p, (x_rounded, y_rounded)) for p in boundary_set]) < step:
            #if (x_rounded, y_rounded) in boundary_set:
                # This is a boundary point. Keep the furthest one.
                if distance > furthest_distance:
                    furthest_distance = distance
                    furthest_point = (x_rounded, y_rounded)

        return furthest_point

    def find_furthest_boundary_point_raycast(self, agent_pos, heading_angle, boundary_polygon, max_distance=10.0):
        """
        Find the furthest boundary point in the direction the agent is looking.
        Uses ray casting and handles GeometryCollection and MultiPoint correctly.
        """
        dx = math.cos(heading_angle)
        dy = math.sin(heading_angle)

        ray_start = Point(agent_pos[0], agent_pos[1])
        ray_end = Point(agent_pos[0] + dx * max_distance,
                        agent_pos[1] + dy * max_distance)

        ray = LineString([ray_start, ray_end])
        boundary_ring = boundary_polygon.exterior

        intersection = ray.intersection(boundary_ring)

        if intersection.is_empty:
            # Try extending the ray further
            ray_end = Point(agent_pos[0] + dx * 100.0,
                            agent_pos[1] + dy * 100.0)
            ray = LineString([ray_start, ray_end])
            intersection = ray.intersection(boundary_ring)

            if intersection.is_empty:
                return None

        # Helper function to extract points from a geometry
        def extract_points(geom):
            """Recursively extract all points from a geometry."""
            points = []

            if geom.geom_type == 'Point':
                points.append(geom)
            elif geom.geom_type == 'MultiPoint':
                points.extend(list(geom.geoms))
            elif geom.geom_type in ['LineString', 'LinearRing']:
                # Check if boundary is a MultiPoint and extract its points
                boundary = geom.boundary
                if boundary.geom_type == 'MultiPoint':
                    points.extend(list(boundary.geoms))
                elif boundary.geom_type == 'Point':
                    points.append(boundary)
                # If boundary is empty or GeometryCollection, skip
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    points.extend(extract_points(line))
            elif geom.geom_type == 'GeometryCollection':
                for sub_geom in geom.geoms:
                    points.extend(extract_points(sub_geom))

            return points

        # Extract all points from the intersection
        all_points = extract_points(intersection)

        if not all_points:
            return None

        # Find the furthest point from the agent
        furthest = max(all_points, key=lambda p: p.distance(ray_start))
        return (furthest.x, furthest.y)

    def get_target_boundary_point(self, room_boundary, agent_pos_with_rtn):
        """
        Get the target boundary point for perimeter navigation.
        This returns the furthest boundary point in the agent's line of sight.
        """
        room_boundary_poly = Polygon(room_boundary)
        agent_x = agent_pos_with_rtn[0]
        agent_y = agent_pos_with_rtn[1]

        #print("WITHIN BOUNDS: ", room_boundary_poly.contains(Point(agent_x, agent_y)))

        agent_rtn = agent_pos_with_rtn[2]
        agent_pos = (agent_x, agent_y)

        # Convert heading to math angle (0=right, counterclockwise)
        heading_rad = math.radians(agent_rtn)
        math_angle = -heading_rad + math.pi / 2

        # Use the ray casting approach
        target = self.find_furthest_boundary_point_raycast(
            agent_pos, math_angle, room_boundary_poly, max_distance=10.0
        )

        if target is None:
            # Fallback: use sampling
            target = self.find_furthest_boundary_point(
                agent_pos, math_angle, room_boundary_poly, max_distance=10.0
            )

        return target

#if __name__ == "__main__":
    #bc = BoundaryCalculations()