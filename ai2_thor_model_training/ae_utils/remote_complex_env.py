# ai2thor_server.py (Run on X86 Laptop)

import socket, json, cv2, logging, threading, elements, random, traceback, pbd, pickle, os, time, itertools
import numpy as np
from connection import (recv_data, send_data)
from ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable, RoomType)
from ai2_thor_model_training.training_data_extraction import RobotNavigationControl
import thortils as tt
from thortils import launch_controller
from thortils.agent import thor_reachable_positions
from thortils.utils import roundany, getch
from thortils.utils.math import sep_spatial_sample
from enum import Enum

class RemoteEnv:
	hab_exploration_stats_collection = []
	LOCK = threading.Lock()

	# Define an enum to help specify how to handle habitat position data after we load a new habitat,
	# because sometimes we want to send the latest data, but sometimes we want to only store it to send
	# out later.
	class WhatToDoWithHabPosData(Enum):
		SEND_FRESH = "send fresh"
		SEND_STORED = "send stored"
		STORE = "store"

	def __init__(self,
				 encoding = 'utf-8',
				 conn = None,
				 hab_space=(100, 600),
				 hab_set="test",
				 behaviour_type = "simple"): ## behaviour_type == simple or behaviour_type == complex. If complex, then we have to carefully coordinate two or more agents working with this env
		self.encoding = encoding
		self.conn = conn
		self.hab_set = hab_set
		self.atu = AI2THORUtils()
		self.rnc = RobotNavigationControl()
		self.grid_size = 0.125 # how fine do we want the 2D grid to be.
		self.plan_close_enough = 0.25 # how close to the target is close enough for the purposes of path planning. We may end up planning path to a point anywhere near the actual target by this much
		self.reward_close_enough = 0.25  # how close to the target is close enough for the purposes of reward. If we're this close or closer in simulation to the target, then consider it done

		self.choose_habitats_randomly_or_sequentially = False
		self.controller = None
		self.nu = NavigationUtils(step=self.grid_size)

		# When we store the statistics of each test run, we will want to capture these variables
		self.astar_path = []
		self.path_start = None
		self.path_dest = None
		self.travelled_path = []
		self.chosen_actions = []

		self._bad_spot = False
		self._bad_spot_cnt = 0
		self._total_reward_for_this_run = 0
		self.step_count_in_current_episode = 0
		self.step_count_since_start = 0
		self.distance_left = np.float32(0.0)
		self.room_type = -1  # current room type
		self.starting_room = None  # which room we end up in when we spawn
		self.target_room = None  # which room we want to end up in
		self.current_room = None  # which room are we in now
		self.steps_in_new_room = 0  # how many steps have we made inside the new room since we first stepped into the target room (resets if we leave target room)
		self.env_retired = False  # in some cases we want to be able to signal to driver.py that this env does not need driving anymore. This will help with that.
		self.prev_obs = None

		# upon beginning we don't have any habitat loaded yet, but we will check this variable to determine if we have
		self.habitat_id = None
		self.explored_placements_in_current_habitat = []
		(self.hab_min, self.hab_max) = hab_space

		# Dreamer stuff
		self.isFirst = False
		self._step = 0

		self.need_to_run = True
		self.stored_hab_and_pos = None
		self.places_per_hab = 10
		self.behaviour_type = behaviour_type
		self.is_complex_behaviour = False if self.behaviour_type == "simple" else True
		self.episode_stats = None

	def load_random_habitat(self, how_to_handle_hab_pos_data: 'RemoteEnv.WhatToDoWithHabPosData' = WhatToDoWithHabPosData.SEND_FRESH):
		print("LRH1")
		# choose a random habitat from a space of given habitats by self.hab_max and self.hab_min
		loaded = False

		# we are going to choose a completely new habitat now. Before we do that, we want to register somewhere
		# what habitat was being explored up until now and what placements were looked at in there.
		if len(self.explored_placements_in_current_habitat) > 0:
			print("LRH2")
			hab_exploration_stats = {
				"local_step": self.step_count_since_start,
				"habitat_id": self.habitat_id,
				"explored_placements_in_current_habitat": self.explored_placements_in_current_habitat
			}
			# print(hab_exploration_stats)
			RemoteEnv.hab_exploration_stats_collection.append(hab_exploration_stats)
			with open("stat_store", "wb") as stat_store:
				pickle.dump(RemoteEnv.hab_exploration_stats_collection, stat_store)
			print("LRH2.1")
		# now that we've saved previous habitat exploration stats, we can carry on with a new habitat

		while not loaded:
			try:
				if (self.choose_habitats_randomly_or_sequentially):  # if we want a random habitat (e.g. we're training)
					print("LRH3")
					sp = elements.Space(np.int32, (), self.hab_min, self.hab_max)
					self.habitat_id = sp.sample()
				else:
					# if we want a sequential habitat (e.g. we're evaluating or testing)
					print("LRH4")
					if (self.habitat_id == None):
						self.habitat_id = self.hab_min
					elif (self.habitat_id < self.hab_max):
						self.habitat_id += 1
					else:
						# we're done, we need to terminate the evaluation process now
						# exit()
						# but instead of just exiting the whole program, let's set up a flag that will tell driver.py
						# that this env does not need driving anymore.
						self.env_retired = True
						break

				# load_habitat will also call self.choose_random_placement_in_habitat(), which will in turn calculate
				# current distance cost to the target
				print("LRH5")
				self.load_habitat(self.habitat_id, how_to_handle_hab_pos_data)
				# enfore at least 2 rooms in a habitat
				if len(self.rooms_in_habitat) >= 2:
					loaded = True
			except ValueError as e:
				continue

	# print("LRH2")
	##
	# This kind of combines 2 functions: load_random_habitat and choose_random_placement_in_habitat.
	# The idea is that usually we only want to load a different starting point within the same habitat,
	# but sometimes we will want to load a new habitat entirely. Also, if we have exhausted all usable random
	# places in the given habitat, then we want to load a new habitat. This is all best handled in one place-
	# this function.
	##
	def load_next_start_point(self, how_to_handle_hab_pos_data: 'RemoteEnv.WhatToDoWithHabPosData' = WhatToDoWithHabPosData.SEND_FRESH):
		print("L1")
		# if nothing has been loaded, then we just load a brand new habitat - Simple
		if self.habitat_id is None:
			print("L1.1")
			self.load_random_habitat(how_to_handle_hab_pos_data)
		else:
			# otherwise, we want to look at what have we explored and what is available
			# if we have already explored 20 random locations in this habitat, then it's time to move on
			print("L1.2")
			if len(self.explored_placements_in_current_habitat) > self.places_per_hab:
				print("L1.3")
				self.load_random_habitat(how_to_handle_hab_pos_data)
			else:
				# otherwise try to load the next random placement (it will attempt a few times, currently 10).
				# If that fails, then we load new habitat.
				try:
					print("L1.4")
					self.choose_random_placement_in_habitat(how_to_handle_hab_pos_data)
				except ValueError as e:
					print("L1.5")
					self.load_random_habitat(how_to_handle_hab_pos_data)

		self.isFirst = True  # we just loaded a new scene or habitat. The next observation will be first

	print("L2")

	##
	# Load the given habitat- load it, and put agent in a random place
	##
	def load_habitat(self, habitat_id, how_to_handle_hab_pos_data: 'RemoteEnv.WhatToDoWithHabPosData' = WhatToDoWithHabPosData.SEND_FRESH):
		print("LH1")
		# load required habitat
		# print("AE: haba: ", habitat_id)
		self.habitat = self.atu.load_proctor_habitat(int(habitat_id), self.hab_set)

		self.explored_placements_in_current_habitat = []
		#breakpoint()
		# Launch a controller for the loaded habitat. If we already have a controller,
		# then reset it instead of loading a new one.
		if (self.controller == None):
			print("LH2")
			self.controller = launch_controller({"scene": self.habitat,
												 "VISIBILITY_DISTANCE": 3.0,
												 "headless": False,
												 "IMAGE_WIDTH": 64,
												 "IMAGE_HEIGHT": 64,
												 "GRID_SIZE": self.grid_size,
												 "GPU_DEVICE": 1,
												 # "RENDER_DEPTH": False,
												 # "RENDER_INSTANCE_SEGMENTATION": False,
												 # "RENDER_IMAGE": True
												 # "IMAGE_WIDTH": 64,
												 # "IMAGE_HEIGHT": 64
												 })
			self.rnc.set_controller(
				self.controller)  # This allows our control scripts to interact with AI2-THOR environment
			self.atu.set_controller(self.controller)
			print("LH3")
		else:
			print("LH4")
			self.controller.reset(self.habitat)
			print("LH4.1")
			# self.reset_state()
			self.rnc.reset_state()
			print("LH5")
		# self.rnc.set_controller(self.controller)

		# Take a snapshot of all available positions- these won't change while we're in this habitat,
		# so no need to re-do them everytime we plan a path.
		# self.grid_size = self.controller.initialization_parameters["gridSize"]
		print("LH6")
		self.reachable_positions, self.unreachable_postions, self.full_grid, self.rooms_in_habitat = self.update_navigation_artifacts(
			self.habitat)

		## Check the suitability of the habitat
		rooms_in_habitat = get_rooms_ground_truth(self.habitat)
		# we want to check that each room centre can be reached from all other room centres
		if len(rooms_in_habitat) < 2 or not self.verify_habitat_connectivity(rooms_in_habitat):
			print("SKIPPING habitat ", habitat_id, " as not suitable")
			raise ValueError("Not all rooms are connected.")

		# Now place the robot in a random position and figure out the target from there.
		print("LH7")
		self.choose_random_placement_in_habitat(how_to_handle_hab_pos_data)

	# self.choose_specific_placement_in_habitat()
	# print("LH2")

	##
	# Checking if all rooms are connected. We will want that for complex tasks.
	##
	def verify_habitat_connectivity(self, rooms_in_habitat):
		"""Verify all rooms in habitat are connected"""
		centers = [room[2] for room in rooms_in_habitat]
		# Simple combination check
		for center_a, center_b in itertools.combinations(centers, 2):
			start_point = ((center_a.x, 0.9009993672370911, center_a.y), (0, 180, 0))
			try:
				path_length = self.nu.get_path_cost_to_target_point(start_point,
																				 center_b,
																				 self.reachable_positions,
																				 close_enough=self.plan_close_enough,
																				 step=self.grid_size)
			except ValueError as e:
				return False

		return True

	# Get all reachable positions and store them in a variable.
	def update_navigation_artifacts(self, house):
		# print("U1")
		reachable_positions = [
			tuple(map(lambda x: round(roundany(x, self.grid_size), 2), pos))
			for pos in thor_reachable_positions(self.controller)]
		# print(reachable_positions, self.grid_size)

		# In this habitat we have these rooms
		rooms_in_habitat = get_rooms_ground_truth(house)
		# print(house["rooms"])
		# print("reachable_positions: ", reachable_positions)
		# AE: Path length infra set up
		# pos_ba = thor_reachable_positions(controller, by_axes = True)
		# print("AE, by axes: ", pos_ba)
		full_grid = create_full_grid_from_room_layout(rooms_in_habitat, step=self.grid_size)
		full_grid = [tuple(map(lambda x: round(x, 2), pos)) for pos in full_grid]
		unreachable_postions = set(full_grid) - set(reachable_positions)
		# (safe_pos, buf_unreachable_pos) = add_buffer_to_unreachable(set(reachable_positions), set(full_grid), step=self.grid_size)

		# print("U2")
		return reachable_positions, unreachable_postions, full_grid, rooms_in_habitat

	# This function will calculate path length to the desired point from the current position.
	# Also- what room type we're in
	def get_current_path_and_pose_state(self):
		# print("G1")
		try:
			cur_pos = self.rnc.get_agent_pos_and_rotation()
			self.current_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
																			 self.current_target_point,
																			 self.reachable_positions,
																			 close_enough=self.plan_close_enough,
																			 step=self.grid_size)
			self.current_path_length = np.float32(self.current_path_length)
		except ValueError as e:
			# print(f"ERROR: {e}")
			# print("Using previous current_path_length: ", self.current_path_length)
			print('!', end='', sep='')
			self._bad_spot = True
			self._bad_spot_cnt += 1
			raise e  # pass it on because reward calculation also needs to know

		# if we've been successful so far, then we can now look up room type
		cur_pos_xy = (cur_pos[0][0], "", cur_pos[0][2])
		room_type = self.find_room_type_of_this_point(cur_pos_xy)
		if room_type == None:
			print('i', end='', sep='')
			raise ValueError("Bad room picked, type can't be determined.")
		# and the actual room
		self.current_room = room_this_point_belongs_to(self.rooms_in_habitat, cur_pos_xy)
		if (self.current_room == None):
			print('y', end='', sep='')
			raise ValueError("Current room not identifiable")

		# print("G2")
		return self.current_path_length, room_type, cur_pos_xy

	def create_rnd_object(self):
		seed = 1983
		if not hasattr(self, "rnd"):
			self.rnd = random.Random(seed)
		return self.rnd

	##
	# Here we will select a number of random placements and then choose one to navigate from it
	# to some goal.
	##
	def choose_random_placement_in_habitat(self,
										   how_to_handle_hab_pos_data: 'RemoteEnv.WhatToDoWithHabPosData' = WhatToDoWithHabPosData.SEND_FRESH):
		print("CH1")
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
		## If we are training (i.e., loading habitats randomly), then don't use a seed
		if (self.choose_habitats_randomly_or_sequentially):
			rnd = random.Random()
		else:  # if, on the other hand, we are evaluating or testing (loading habitats sequentially), then test everything the same way- use a seed
			rnd = self.create_rnd_object()

		initial_agent_pose = tt.thor_agent_pose(self.controller)
		initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

		# reachable_positions = tt.thor_reachable_positions(self.controller)
		# self.reachable_positions
		placements = sep_spatial_sample(self.reachable_positions, sep, num_stops, rnd=rnd)

		# print(placements)

		# Choose one placement in the set of placements and then plan path from that placement to
		# the middle of the room. If planning path is not possible, then choose another one.
		path_planned = False
		placement_attempts = 0
		while not path_planned:
			print("CH1.2")
			placement_attempts += 1
			# els = elements.Space(np.int32, (), 0, len(placements))
			# p = list(placements)[int(els.sample())]
			el_ndx = rnd.randrange(0, len(placements))
			p = list(placements)[el_ndx]
			# p = placements.pop()

			# append a rotation to the place.
			yaw = rnd.sample(h_angles, 1)[0]
			place_with_rtn = p + (yaw,)
			# print("Placement: ", place_with_rtn, " el_ndx: ", el_ndx)
			self.explored_placements_in_current_habitat.append(place_with_rtn)
			## Teleport, then start new exploration. Achieve goal. Then repeat.
			self.rnc.teleport_to(place_with_rtn)
			print("CH1.3")

			# We've just been put in a random place in a habitat. We want to move now to where we want to go,
			# e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
			try:
				point_for_room_search = (p[0], "", p[1])
				print("CH1.3.1 p1: ", place_with_rtn, " p2: ", point_for_room_search)
				self.current_target_point = self.choose_target_point(place_with_rtn,
																	 point_for_room_search)  # self.target_room will be set in this function
				print("CH1.3.2")
				cur_pos = self.rnc.get_agent_pos_and_rotation()
				print("CH1.3.3", self.current_target_point)
				# print("Placement: ", place_with_rtn, " cur_pos: ", cur_pos, " el_ndx: ", el_ndx)
				self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
																				 self.current_target_point,
																				 self.reachable_positions,
																				 close_enough=self.plan_close_enough,
																				 step=self.grid_size)

				print("CH1.4")
				# Now let's remember the A* path- we will want it for results.
				(self.astar_path, _, self.path_start, self.path_dest) = self.nu.get_last_path_and_params()
				# print("AE: Path: ", self.astar_path)
				print("CH2")
				if isinstance(self, DoorFinder):
					# what is the room we start in
					print("CH3")
					self.starting_room = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

					if (self.starting_room == None): raise ValueError("Starting room not identifiable")

					# We must ensure that we navigate from one room to another
					if self.target_room == self.starting_room: raise ValueError("start and end points in same room")
					print("CH4")
			except (ValueError, AttributeError) as e:
				# If the path could not be planned, then drop it and carry on with the next one
				# print(f"ERROR: {e}")
				if placement_attempts <= 10:
					print('.', sep='', end='')
					continue
				else:
					# If we have tried for 10 times already, then give up with this habitat
					print("next_hab", sep="", end="")
					raise e

			# print("PATH & PLAN: ", path_and_plan)
			#            path = path_and_plan[0]
			#            plan = path_and_plan[1]
			# self.prev_pose = thor_agent_pose(self.controller)  # This is where we are before the plan started
			# place_with_rtn
			# thor_pose_as_tuple(self.prev_pose)
			# print("AE poses: place_with_rtn: ", place_with_rtn, " p: ", p, " self.rnc.get_agent_pos_and_rotation(): ",
			#      self.rnc.get_agent_pos_and_rotation())
			#            cur_pos = self.rnc.get_agent_pos_and_rotation()
			#            self.initial_path_length = get_path_length(path, cur_pos)

			# at this point current path length is the initial path length. We will re-calculate current path length
			# many times and reward will be calculated using it.
			self.current_path_length = self.initial_path_length
			self.best_path_length = self.initial_path_length

			# print("CH2")
			path_planned = True
			print("CH5")
			self.send_habitat_and_pos_data(cur_pos, how_to_handle_hab_pos_data)

	def send_habitat_and_pos_data(self, cur_pos, what_to_do: 'RemoteEnv.WhatToDoWithHabPosData', conn_to_use = None) -> None:
		# Send initial READY response
		#print("AE: self.current_target_point: ", self.current_target_point)
		response = {"msg": "HAB_AND_POS",
					"hab_set": self.hab_set,
					"hab_id": self.habitat_id,
					"cur_pos": cur_pos,
					"current_target_point": (self.current_target_point.x, self.current_target_point.y),
					"initial_path_length": self.initial_path_length,
					"astar_path": self.astar_path,
					"path_start": self.path_start,
					"path_dest": self.path_dest,
					"starting_room": (self.starting_room[2].x, self.starting_room[2].y) if self.starting_room != None else None,
					"current_path_length": self.current_path_length,
					"best_path_length": self.best_path_length
					}
		#print("response: ", response)
		if what_to_do == self.WhatToDoWithHabPosData.SEND_FRESH:
			if conn_to_use == None:
				send_data(self.conn, json.dumps(response).encode(self.encoding))
			else:
				send_data(conn_to_use, json.dumps(response).encode(self.encoding))
		elif what_to_do == self.WhatToDoWithHabPosData.STORE:
			self.stored_hab_and_pos = response
		elif what_to_do == self.WhatToDoWithHabPosData.SEND_STORED:
			if conn_to_use == None:
				send_data(self.conn, json.dumps(self.stored_hab_and_pos).encode(self.encoding))
			else:
				send_data(conn_to_use, json.dumps(self.stored_hab_and_pos).encode(self.encoding))
			#self.stored_hab_and_pos = None

	# Determines if we have little enough left to call it an achieved goal
	def have_we_arrived(self, epsilon=0.0):
		pass

	def close(self):
		if (self.controller != None):
			self.controller.stop()
			print("AI2-THOR controller stopped.")

	##
	# Chooses the target point that we want to navigate to, given current position.
	# This will have to be implemented in derived classes.
	##
	def choose_target_point(self, place_with_rtn=None, place_with_no_rtn=None):
		pass

	# if (self.doors_or_centre):
	#     self.current_target_point = self.choose_door_target(place_with_rtn)
	#     # print("self.current_target_point: ", self.current_target_point)
	# else:
	#     point_for_room_search = (p[0], "", p[1])
	#     # print("point_for_room_search: ", point_for_room_search)
	#     self.current_target_point = self.find_room_centre_target(point_for_room_search)

	def find_room_type_of_this_point(self, point_for_room_search):
		'''
		We want to find out an ID for the room type that this point belongs to
		:param point_for_room_search:
		:return:
		'''
		# print("FR1")
		room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
		if room_of_placement == None:
			return None
		room_type = room_of_placement[0]
		room_type = np.uint8(RoomType.interpret_label(room_type).value)
		# print("FR2")
		return room_type

	def initialize_connection(self, conn, command = None):
		if command == None:
			hab_id = "83"
			self.hab_set = "test"
		else:
			hab_id = command.get("hab_id", "1")
			self.hab_set = command.get("hab_set", "train")

		# Initialize AI2-THOR controller on the server
		#print(f"2: Initializing AI2-THOR for scene: {self.hab_set}[{hab_id}]...")
		#self.load_habitat(hab_id)
		#print("2: AI2-THOR initialized. Ready for actions.")

		# Send initial READY response
		response = {"status": "READY",
					"scene": self.hab_set + "[" + str(hab_id) + "]",
					#"reachable_positions": self.reachable_positions,
					#"unreachable_postions": list(self.unreachable_postions),
					#"full_grid": self.full_grid
					}
		#print("OUT: ", response)
		send_data(conn, json.dumps(response).encode(self.encoding))

	def execute_action(self, conn, command):
		#breakpoint()
		action_from_dreamer = command.get('action_bits', {"action": -1, "reset": True})
		print(f"-> Received action: {action_from_dreamer}")

		# Execute action
		cur_obs, episode_stats = self.step(action_from_dreamer)
		#print("AE: Env stepped")

		# Prepare Frame (Convert numpy array to JPEG bytes)
		# Use CV2 to encode the numpy array as JPEG for efficient transfer
		is_success, buffer = cv2.imencode(".jpg", cur_obs["pov"])
		#print("AE: IMG encoded")
		if not is_success:
			raise Exception("Failed to encode frame to JPEG.")
		frame_bytes = buffer.tobytes()
		#print("AE: img buffered")
		# now nullify the current ndarray of picture data, because we don't want to send it with json data
		cur_obs["pov"] = []
		#print(cur_obs)
		episode_stats = {}
		return_structure = {"obs": cur_obs, "eps": episode_stats}
		#print("AE: cmd ready to send: ", return_structure)
		send_data(conn, json.dumps(return_structure).encode(self.encoding))
		print("AE: cmd sent: ", action_from_dreamer['reset'])
		# now send jpeg data
		send_data(conn, frame_bytes)
		print("AE: img sent: ")
		return action_from_dreamer['reset']

	def step(self, action):
		# If this env has been retired (in evaluation mode we have evaluated everything already), then
		# don't actually do any stepping, but just return the previous obs
		if self.env_retired:
			return self.prev_obs, self.episode_stats

		if action['reset']:
			print('R', end='', sep='')
			# STORE EPISODE STATS:
			# A* path length, A* path, travelled path length, travelled path, habitat id, actions taken.
			if self.hab_set != "train":
				self.episode_stats = {
					"local_step": self.step_count_since_start,
					"steps_used": self.step_count_in_current_episode,
					"habitat_id": self.habitat_id,
					"bad_spot": self._bad_spot,
					"have_arrived": str(bool(self.have_we_arrived(self.reward_close_enough))),
					"path_start": self.path_start,
					"path_dest": self.path_dest,
					"astar_path": self.astar_path,
					"travelled_path": self.travelled_path,
					"chosen_actions": self.chosen_actions,
				}
				# print(hab_exploration_stats)

				with open(self.logdir + "/episode_data.jsonl", "a") as f:
					f.write(json.dumps(self.episode_stats) + "\n")

			obs = self._reset()
		elif index_to_action(int(action['action'])) == "STOP":
			self.chosen_actions.append(int(action['action']))
			self._done = True

			try:
				self.distance_left, self.room_type, self.cur_pos_xy = self.get_current_path_and_pose_state()
				self.travelled_path.append(self.cur_pos_xy)
			except ValueError as e:
				self.distance_left = np.float32(0.0)
				self._bad_spot = True
				print('O', end='', sep='')

			print('S', end='', sep='')
			obs = self.current_ai2thor_observation()
		else:
			raw_action = index_to_action(int(action['action']))
			self.rnc.execute_action(raw_action, moveMagnitude=self.grid_size, grid_size=self.grid_size,
									adhere_to_grid=True)
			self.chosen_actions.append(int(action['action']))
			# This is slightly ugly, but we need to calculate distance_left variable right after rnc.execute_action
			# to allow observation to be up to date. In time this should be moved to some function instead of relying
			# on global variables.
			try:
				self.distance_left, self.room_type, cur_pos_xy = self.get_current_path_and_pose_state()
				self.travelled_path.append(cur_pos_xy)
				#self._done = bool(self.have_we_arrived(self.reward_close_enough))
			except ValueError as e:
				self.distance_left = np.float32(0.0)
				self._bad_spot = True

			if self._bad_spot:
				# print("FORCED SCENE CHANGE!!!", self.step_count_in_current_episode)
				# STORE EPISODE STATS
				print('O', end='', sep='')
				##
				# This must be self._done = True instead of direct reset. We will reset in the next loop
				##
				# obs = self._reset()
				self._done = True
			# else:
			obs = self.current_ai2thor_observation()

		# Now we turn the obs that was returned by the environment into obs that we use for training,
		# and to not confuse the two, make sure that 'pov' field is not there, because it should be 'image'.
		self._step += 1
		self.step_count_in_current_episode += 1
		self.step_count_since_start += 1
		self.prev_obs = obs
		return obs, self.episode_stats

	##
	# Returns current observation of the state (image mostly)
	##
	def current_ai2thor_observation(self):
		# print("O1")
		event = self.controller.last_event
		self._current_image = event.cv2img

		# print("self.current_room == self.target_room", self.current_room, self.target_room)
		# if we're in the target room, then count how many steps we've done in the target room
		self.steps_in_new_room = self.steps_in_new_room + 1 if self.current_room == self.target_room else 0

		obs = dict(
			reward=0.0,
			pov=self._current_image,
			is_first=self.isFirst,
			is_last=self._done,
			is_terminal=self._done,
			# distance_left = np.float32(self.distance_left),
			# steps_after_room_change = np.float32(self.steps_in_new_room),
			# room_type = np.float32(self.room_type),
			distanceleft=float(self.distance_left),
			stepsafterroomchange=int(self.steps_in_new_room),
			roomtype=int(self.room_type),
		)
		if self._done:
			print('D', sep='', end='')

		self.isFirst = False  # this will have to be set to True when we reset the env
		# print("O2")
		return obs

	def _reset(self):
		# print("R1")
		self.astar_path = []
		self.path_start = None
		self.path_dest = None
		self.travelled_path = []
		self.chosen_actions = []
		# Load new point or even a habitat, set reward to 0 and is_first = True and is_last = False and self._done = False
		#with self.LOCK:
		#	self.load_next_start_point(how_to_handle_hab_pos_data = self.WhatToDoWithHabPosData.STORE)
		# obs = self._env.step({'reset': True})

		self.step_count_in_current_episode = 0
		self._step = 0
		self._done = False
		self._bad_spot = False

		self.distance_left = 0
		self.steps_in_new_room = 0
		self.room_type = -1
		self.starting_room = None

		obs = self.current_ai2thor_observation()
		# print("R2")
		return obs

	def handle_client(self, conn, addr, task_completion_callback = None, conn_obj = None):
		print(f"✅ Connection established with {addr} ", self.is_complex_behaviour, conn_obj["unhandled_act_cmd"])
		self.need_to_run = True
		# we have to make sure that we call the callback function at the very end of this thread, otherwise we may
		# start a new one before this one finishes and mess up internal variables.
		self.callback_needs_calling_in_the_end = False
		if self.is_complex_behaviour:
			#self.initialize_connection(conn)
			# Normally, when we see that Dreamer asked us to reset, we would stop the task and hand back
			# the control over the socket to the pre_work_comms_handler routine where we handle it
			# "in a bubble", but we don't want to do that on the first reset, because that's also how
			# a task starts.
			first_reset_dont_end_episode = True
		else:
			first_reset_dont_end_episode = False
		# keep reading commands from client and do what it wants
		#breakpoint()
		try:
			while self.need_to_run:
				# If we have no command to process immediately, then wait until we receive one
				if conn_obj["unhandled_act_cmd"] == None:
					cmd_data_bytes = recv_data(conn)
					if not cmd_data_bytes:
						raise Exception("Client closed connection.")
					command = json.loads(cmd_data_bytes.decode(self.encoding))
				else: # otherwise proceed with what needs to be processed
					command = conn_obj["unhandled_act_cmd"]
					conn_obj["unhandled_act_cmd"] = None
				print("incoming cmd: ", command)
				# here we handle what the client wants exactly
				if command.get("command") == "INIT":
					self.initialize_connection(conn, command)
				elif command.get("command") == "ACT":
					action_from_dreamer = command.get('action_bits', {"action": -1, "reset": True})
					isStopAction = (action_from_dreamer["action"] == 3)
					isReset = action_from_dreamer["reset"]
					# if dreamer requested a reset, then let's keep that ACT command and stop the sequence here.
					# we may want to give back control to this agent later, at which point we may need this resetting ACT command
					if isReset and task_completion_callback != None:
						# if episode was reset (e.g. completed, then we may want to notify the composite task process, if any)
						# we may want to ignore first reset, because that may well be the start of the navigation task
						if first_reset_dont_end_episode:
							first_reset_dont_end_episode = False
							reset = self.execute_action(conn, command)
						else:
							conn_obj["unhandled_act_cmd"] = command
							self.callback_needs_calling_in_the_end = True
							self.need_to_run = False
							# Prepare episode stats for when we'll call the task completion
							self.episode_stats = {
								"local_step": self.step_count_since_start,
								"steps_used": self.step_count_in_current_episode,
								"habitat_id": self.habitat_id,
								"bad_spot": self._bad_spot,
								"have_arrived": str(bool(self.have_we_arrived(self.reward_close_enough))),
								"path_start": self.path_start,
								"path_dest": self.path_dest,
								"astar_path": self.astar_path,
								"travelled_path": self.travelled_path,
								"chosen_actions": self.chosen_actions,
							}
					# actually execute the action
					else:
						reset = self.execute_action(conn, command)
				elif command.get("command") == "NEXT_POINT":
					if self.is_complex_behaviour: # only allow the first room centre find part to select a new location
						#if addr == "rc1":
						#	self.load_next_start_point(how_to_handle_hab_pos_data=self.WhatToDoWithHabPosData.SEND_FRESH)
						cur_pos = self.rnc.get_agent_pos_and_rotation()
						self.send_habitat_and_pos_data(cur_pos, self.WhatToDoWithHabPosData.SEND_STORED)
					else:
						self.load_next_start_point(how_to_handle_hab_pos_data = self.WhatToDoWithHabPosData.SEND_FRESH)
				elif command.get("command") == "GET_SAVED_HAB_AND_POS":
					'''
					When we reset the env, we will be returning an observation as normal, but as part of the reset, will
					be loading the next start point, which in itself triggers sending back data to the client. So to avoid
					a situation where client is expecting an observation and episode stats, but we are sending habitat stats
					for the newly loaded location, we will need to store the habitat stats and send it later, when requested.
					This will handle that type of request.
					'''
					self.send_habitat_and_pos_data(None, self.WhatToDoWithHabPosData.SEND_STORED)
				elif command.get("command") == "KEEP":
					response = {"status": "OK"}
					send_data(conn, json.dumps(response).encode(self.encoding))
					continue
			print("AE: Handler thread EXITING")
			if self.callback_needs_calling_in_the_end:
				task_completion_callback()
		except Exception as e:
			print(f"Error handling client {addr}: {e}")
			self.close()
			conn.close()
			print(f"Connection with {addr} closed.")
			self.need_to_run = False
			print("AE: Handler thread EXITING DUE TO ERROR")

	def set_agent_conn(self, agent_conn):
		self.conn = agent_conn
		#self.initialize_connection(self.conn)

class ServerSocketMaster():
	def __init__(self, host = '0.0.0.0', port = 9999, encoding = 'utf-8'):
		self.host = host
		self.port = port
		self.encoding = encoding
		self.running_envs = [] # this can be used for testing or training individual behaviours and we can have multiple
		self.explore_env = None # this one is for a complex behavior, e.g. exploring a habitat and we only want one of it

	def start_server(self):
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			server_socket.bind((self.host, self.port))
			server_socket.listen(5)
			print(f"AI2-THOR Server listening on {self.host}:{self.port}")

			while True:
				conn, addr = server_socket.accept()
				cmd_data_bytes = recv_data(conn)
				if not cmd_data_bytes:
					raise Exception("Client closed connection.")
				command = json.loads(cmd_data_bytes.decode(self.encoding))
				print("IN: ", command)
				# here we handle what the client wants exactly
				if command.get("command") == "INIT":
					hab_id = command.get("hab_id", "1")
					hab_set = command.get("hab_set", "train")
					hab_min = command.get("hab_min", 0)
					hab_max = command.get("hab_max", 9)
					hab_set = "test"
					hab_min = 873
					hab_max = 999
					env_type = command.get("env_type", "RoomCentreFinder")
					agent_type = command.get("agent_type", "")
				elif command.get("command") == "KEEP":
					response = {"status": "OK"}
					send_data(conn, json.dumps(response).encode(self.encoding))
					continue
				else:
					raise Exception("Client's first command was not INIT")

				# Initialize AI2-THOR controller on the server
				print(f"Initializing AI2-THOR for scene: {hab_set}[{hab_id}], at: {agent_type}, et: {env_type}...")
				print("AI2-THOR initialized. Ready for actions.")

				# Handle client connection in a new thread
				if agent_type == "" and env_type == "RoomCentreFinder":
					env = RoomCentreFinder(encoding = self.encoding, conn = conn, hab_space=(hab_min, hab_max), hab_set = hab_set)
				elif agent_type == "" and env_type == "DoorFinder":
					env = DoorFinder(encoding = self.encoding, conn = conn, hab_space=(hab_min, hab_max), hab_set = hab_set)
				elif agent_type == "rc" or agent_type == "dr":
					# SO we want to do the exploration task. For that we will need to have two agents - rc for room centre
					# and dr for door finding. We can create env if it doesn't exist, but we must make sure that we have
					# both agents before we do anything
					if self.explore_env == None:
						self.explore_env = ExploreTask(encoding = self.encoding, conn = conn, hab_space=(hab_min, hab_max), hab_set = hab_set)
						env = self.explore_env # so that it can be added to the running_envs list
						self.explore_env.prepare_and_setup()
					self.explore_env.accept_agent(agent_conn = conn, agent_type = agent_type, first_cmd = command)
				else:
					raise Exception("Unknown Env type specified by client")

				if env != None: self.running_envs.append(env) # env can be none, if we have ExploreTask and 2nd agent connects

				# The activity thread will be started in the task manager for ExploreTask
				if agent_type == "" and env_type in ["RoomCentreFinder", "DoorFinder"]:
					client_thread = threading.Thread(target=env.handle_client, args=(conn, addr))
					client_thread.start()
				env = None

		except socket.error as e:
			print(f"Failed to start server: {e}")
			print("Ensure the port is not in use and firewall is open (as discussed previously).")
		finally:
			server_socket.close()
			for env in self.running_envs:
				env.need_to_run = False

##
# Room centre finding task
##
class RoomCentreFinder(RemoteEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logdir = "rc_log"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        return self.find_room_centre_target(place_with_no_rtn)

    ##
    # Finds the centre of the current room given the current position and the rooms in habitat.
    ##
    def find_room_centre_target(self, point_for_room_search):
        #print("FRC1")
        # We've just been put in a random place in a habitat. We want to move now to where we want to go,
        # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

        if (room_of_placement == None): raise ValueError("Room of placement not identifiable")

        room_centre = room_of_placement[2]
        #print("FRC2")
        return room_centre

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0):
        return (self.current_path_length <= epsilon)

##
# Door Finding Task
##
class DoorFinder(RemoteEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logdir = "door_log"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        return self.choose_door_target(place_with_rtn)

    ##
    # Go through all the doors and find the most appropriate as a target, then add a little bit extra so that
    # we end up going through the door.
    # place_with_rtn: Place with rotation, e.g.: (6.62, 6.25, 180)
    ##
    def choose_door_target(self, place_with_rtn):
        #print("CD1")
        current_target_point = None
        try:
            current_target_point = self.nu.find_door_target(place_with_rtn,
                                                            self.rooms_in_habitat,
                                                            self.reachable_positions,
                                                            self.habitat,
                                                            self.controller, close_enough=self.plan_close_enough,
                                                            step=self.grid_size, extend_path=True)

            # t1 = time.time()
            pose = ((place_with_rtn[0], 0.0, place_with_rtn[1]),
                    (0.0, place_with_rtn[2], 0.0))  # place_with_rtn in AI2-Thor format
            path_length = self.nu.get_path_cost_to_target_point(pose,
                                                                current_target_point,
                                                                self.reachable_positions,
                                                                close_enough=self.plan_close_enough,
                                                                step=self.grid_size)

            # if we've been successful so far, then we can now look up room type
            print("AE1: current_target_point: ", current_target_point)
            trg_pos_xy = (current_target_point.x, "", current_target_point.y)
            self.target_room = room_this_point_belongs_to(self.rooms_in_habitat, trg_pos_xy)
            # print("AE: path plan time: ", (time.time() - t1))
        except ValueError as e:
            path_length = 0
            # print("AE: No Path Found", e)
            print('£', end='', sep='')

        if (self.target_room == None):
            raise ValueError("Target room not identifiable")

        if current_target_point == None:
            raise ValueError("No suitable doors were found")

        # print("AE: Path Length: ", path_length)
        #(cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        # print("AE: Path: ", cur_path)
        # atu.visualise_path2(cur_path, reachable_positions, unreachable_postions, rooms_in_habitat, start, dest,
        #                    show_unreachable_pos=False,
        #                    show_reachable_pos=False)
        # atu.visualise_path2(cur_path, reachable_positions, buf_unreachable_pos, rooms_in_habitat, start, dest, show_unreachable_pos=True)
        #print("CD2")
        return current_target_point

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0):
        return (self.current_path_length <= epsilon or self.steps_in_new_room >= 3)


class ExplorerTaskState(Enum):
	IDLE = "doing nothing"
	WAIT_FOR_AGENT1 = "waiting for agent 1 (rc or dr)"
	WAIT_FOR_AGENT2 = "waiting for agent 2 (rc or dr)"
	LAUNCH_ENV = "Launch AI2-Thor env"
	FIND_RC1 = "Finding Room centre in the initial room"
	FIND_DOOR = "Finding a door and navigating through it"
	FIND_RC2 = "Finding the room centre of the new room"
	WAIT_FOR_RC1 = "Waiting for Room centre finding to complete"
	WAIT_FOR_RC2 = "Waiting for 2nd room centre finding to complete"
	WAIT_FOR_DOOR = "Waiting for door finding to complete"

class ExploreTask():
	# This will be a combined behaviour class for tracking the progress of our first task: to go to the centre of a different room
	# This will entail:
	# 1) Finding the room centre of the initial room
	# 2) Finding a door and navigating through it
	# 3) Finding the room centre of the new room

	def __init__(self, *args, **kwargs):
		#self.env_actions = actions
		self.env_args = args
		self.env_kw_args = kwargs

		self.transitions = {
			ExplorerTaskState.WAIT_FOR_AGENT1: ExplorerTaskState.LAUNCH_ENV,
			#ExplorerTaskState.WAIT_FOR_AGENT2: ExplorerTaskState.LAUNCH_ENV,
			ExplorerTaskState.LAUNCH_ENV: ExplorerTaskState.FIND_RC1,
			ExplorerTaskState.FIND_RC1: ExplorerTaskState.WAIT_FOR_RC1,
			ExplorerTaskState.WAIT_FOR_RC1: ExplorerTaskState.FIND_DOOR,
			ExplorerTaskState.FIND_DOOR: ExplorerTaskState.WAIT_FOR_DOOR,
			ExplorerTaskState.WAIT_FOR_DOOR: ExplorerTaskState.FIND_RC2,
			ExplorerTaskState.FIND_RC2: ExplorerTaskState.WAIT_FOR_RC2,
			ExplorerTaskState.WAIT_FOR_RC2: ExplorerTaskState.IDLE,
			ExplorerTaskState.IDLE: ExplorerTaskState.WAIT_FOR_AGENT1,
		}
		self.current_state = ExplorerTaskState.IDLE
		self.task_stats = {
			"rc1": None,
			"dr": None,
			"rc2": None
		}
		self.task_cnt = 0

		self.logdir = "task_log"
		if not os.path.exists(self.logdir):
			os.makedirs(self.logdir)

		self.launch_remote_env()

		self.all_conn = {
			"rc": {
				"conn": None,
				"conn_cnt": 0,
				"early_monitor": False,
				"id": "rc",
				"thread_launched": False,
				"first_cmd": None,
				"unhandled_act_cmd": None
			},
			"dr": {
				"conn": None,
				"conn_cnt": 0,
				"early_monitor": False,
				"id": "dr",
				"thread_launched": False,
				"first_cmd": None,
				"unhandled_act_cmd": None
			}
		}

	def task_complete(self):
		print("STATE MACHINE: ", self.current_state, " => ", self.transitions[self.current_state])
		if self.current_state == ExplorerTaskState.FIND_RC1:
			# Do something else, because we don't want to start blocking receive. The agent already is waiting.
			#self.all_conn["rc"]["early_monitor"] = True

			# we have just finished finding room centre for the first time, store the episode stats for that
			self.task_stats["rc1"] = self.re.episode_stats
			self.re.episode_stats = {}

			self.find_door()
		elif self.current_state == ExplorerTaskState.FIND_DOOR:
			# we have just finished finding the door, store the episode stats for that
			self.task_stats["dr"] = self.re.episode_stats
			self.re.episode_stats = {}

			self.find_rc2()
		elif self.current_state == ExplorerTaskState.FIND_RC2:
			self.task_cnt += 1
			# we have just finished finding room centre for the 2nd time, store the episode stats for that
			self.task_stats["rc2"] = self.re.episode_stats
			self.re.episode_stats = {}

			self.current_state = ExplorerTaskState.IDLE

			with open(self.logdir + "/episode_data.jsonl", "a") as f:
				f.write(json.dumps(self.task_stats) + "\n")

			print("DONE, #", self.task_cnt)
			self.task_stats = {
				"rc1": None,
				"dr": None,
				"rc2": None
			}
			# load next starting point and repeat
			self.re.load_next_start_point(how_to_handle_hab_pos_data=self.re.WhatToDoWithHabPosData.STORE)
			self.find_rc1()

	def find_rc1(self):
		# first switch off early comms monitor before passing the control to the navigation class
		self.all_conn["rc"]["early_monitor"] = False

		self.re.set_agent_conn(self.all_conn["rc"]["conn"])
		client_thread = threading.Thread(target=self.re.handle_client, args=(self.all_conn["rc"]["conn"], "rc1", self.task_complete, self.all_conn["rc"]))
		client_thread.start()
		#self.task_complete()
		self.current_state = ExplorerTaskState.FIND_RC1

	def find_rc2(self):
		self.all_conn["rc"]["early_monitor"] = False

		self.re.set_agent_conn(self.all_conn["rc"]["conn"])
		client_thread = threading.Thread(target=self.re.handle_client, args=(self.all_conn["rc"]["conn"], "rc2", self.task_complete, self.all_conn["rc"]))
		client_thread.start()
		#self.task_complete()
		self.current_state = ExplorerTaskState.FIND_RC2

	def find_door(self):
		self.all_conn["dr"]["early_monitor"] = False

		self.re.set_agent_conn(self.all_conn["dr"]["conn"])
		client_thread = threading.Thread(target=self.re.handle_client, args=(self.all_conn["dr"]["conn"], "dr", self.task_complete, self.all_conn["dr"]))
		client_thread.start()
		#self.task_complete()
		self.current_state = ExplorerTaskState.FIND_DOOR

	def accept_agent(self, agent_conn, agent_type, first_cmd = None):
		if agent_type != "rc" and agent_type != "dr":
			raise Exception("Unexpected agent type")
		else:
			self.all_conn[agent_type]["conn"] = agent_conn
			self.all_conn[agent_type]["conn_cnt"] += 1
			self.all_conn[agent_type]["first_cmd"] = first_cmd

			if not self.all_conn[agent_type]["thread_launched"]:
				self.all_conn[agent_type]["thread_launched"] = True
				self.all_conn[agent_type]["early_monitor"] = True
				comms_handler_thread = threading.Thread(target=self.pre_work_comms_handler, args=(agent_type,))
				comms_handler_thread.start()
			elif not self.all_conn[agent_type]["early_monitor"]:
				self.all_conn[agent_type]["early_monitor"] = True

		print("AA: ", self.current_state, " rcc: ", self.all_conn["rc"]["conn_cnt"], " dfc: ", self.all_conn["dr"]["conn_cnt"])

	def launch_remote_env(self):
		#self.re = RemoteEnv(*self.env_args, **self.env_kw_args, behaviour_type = "complex")
		self.re = RoomCentreFinder(*self.env_args, **self.env_kw_args, behaviour_type="complex")
		self.re.load_next_start_point(how_to_handle_hab_pos_data=self.re.WhatToDoWithHabPosData.STORE)
		#self.task_complete()

	def prepare_and_setup(self):
		#assert (self.current_state == ExplorerTaskState.IDLE)
		pass
		#self.task_complete()

	# This will handle the comms before we pass them to the Environment for sending observations and stuff.
	# Before that happens, we have stuff like NEXT_POINT, INIT, HAB_AND_POS which needs to be heard and handled.
	def pre_work_comms_handler(self, agent_type):
		while True:
			if self.all_conn[agent_type]["early_monitor"]: # if it wants monitoring, then doing that. We may not want it while doing navigation.
				self.handle_single_comms_obj(self.all_conn[agent_type])
				first_cmd = None
			else:
				time.sleep(0.01)

	def handle_single_comms_obj(self, agent_conn):
		try:
			if agent_conn["first_cmd"] == None:
				cmd_data_bytes = recv_data(agent_conn["conn"])
				if not cmd_data_bytes:
					raise Exception(f"Client closed connection [{agent_conn['id']}].")
				command = json.loads(cmd_data_bytes.decode(self.re.encoding))
			else:
				command = agent_conn["first_cmd"]
				agent_conn["first_cmd"] = None

			print("IN2: ", command, " id: ", agent_conn['id'])
			# here we handle what the client wants exactly
			if command.get("command") == "INIT":
				self.re.initialize_connection(agent_conn["conn"])
			elif command.get("command") == "ACT":
				#raise Exception(f"ACT command handled in wrong place {agent_conn['id']}")
				print(f"{agent_conn['id']} Received ACT cmd, it can wait now until navigation starts")
				agent_conn["unhandled_act_cmd"] = command
				agent_conn["early_monitor"] = False
				if agent_conn['id'] == "rc":
					self.find_rc1()
			elif command.get("command") == "NEXT_POINT":
				self.re.send_habitat_and_pos_data(None, self.re.WhatToDoWithHabPosData.SEND_STORED, conn_to_use=agent_conn["conn"])
			elif command.get("command") == "GET_SAVED_HAB_AND_POS":
				self.re.send_habitat_and_pos_data(None, self.re.WhatToDoWithHabPosData.SEND_STORED, conn_to_use=agent_conn["conn"])
		except Exception as e:
			print(f"Error handling client3: {e}")
			agent_conn["conn"].close()
			agent_conn["early_monitor"] = False
			print(f"Connection closed 4.")

if __name__ == "__main__":
	ssm = ServerSocketMaster()
	ssm.start_server()
