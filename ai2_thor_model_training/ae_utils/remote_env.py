# ai2thor_server.py (Run on X86 Laptop)

import socket, json, cv2, logging, threading, elements, random, traceback, pbd, pickle
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

class RemoteEnv:
	hab_exploration_stats_collection = []
	LOCK = threading.Lock()
	def __init__(self, host = '0.0.0.0', port = 9999, encoding = 'utf-8'): # Listen on all available network interfaces by default
		self.host = host
		self.port = port
		self.encoding = encoding
		self.atu = AI2THORUtils()
		self.rnc = RobotNavigationControl()
		self.grid_size = 0.125
		self.plan_close_enough = 0.25
		self.choose_habitats_randomly_or_sequentially = False
		self.controller = None
		self.nu = NavigationUtils(step=self.grid_size)

	def load_random_habitat(self):
		# print("LRH1")
		# choose a random habitat from a space of given habitats by self.hab_max and self.hab_min
		loaded = False

		# we are going to choose a completely new habitat now. Before we do that, we want to register somewhere
		# what habitat was being explored up until now and what placements were looked at in there.
		if len(self.explored_placements_in_current_habitat) > 0:
			hab_exploration_stats = {
				"local_step": self.step_count_since_start,
				"habitat_id": self.habitat_id,
				"explored_placements_in_current_habitat": self.explored_placements_in_current_habitat
			}
			# print(hab_exploration_stats)
			RemoteEnv.hab_exploration_stats_collection.append(hab_exploration_stats)
			with open("stat_store", "wb") as stat_store:
				pickle.dump(RemoteEnv.hab_exploration_stats_collection, stat_store)
		# now that we've saved previous habitat exploration stats, we can carry on with a new habitat

		while not loaded:
			try:
				if (self.choose_habitats_randomly_or_sequentially):  # if we want a random habitat (e.g. we're training)
					sp = elements.Space(np.int32, (), self.hab_min, self.hab_max)
					self.habitat_id = sp.sample()
				else:
					# if we want a sequential habitat (e.g. we're evaluating or testing)
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
				self.load_habitat(self.habitat_id)
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
	def load_next_start_point(self):
		# print("L1")
		# if nothing has been loaded, then we just load a brand new habitat - Simple
		if self.habitat_id is None:
			self.load_random_habitat()
		else:
			# otherwise, we want to look at what have we explored and what is available
			# if we have already explored 20 random locations in this habitat, then it's time to move on
			if len(self.explored_placements_in_current_habitat) > self.places_per_hab:
				self.load_random_habitat()
			else:
				# otherwise try to load the next random placement (it will attempt a few times, currently 10).
				# If that fails, then we load new habitat.
				try:
					self.choose_random_placement_in_habitat()
				except ValueError as e:
					self.load_random_habitat()

		self.isFirst = True  # we just loaded a new scene or habitat. The next observation will be first

	# print("L2")

	##
	# Load the given habitat- load it, and put agent in a random place
	##
	def load_habitat(self, habitat_id):
		# print("LH1")
		# load required habitat
		# print("AE: haba: ", habitat_id)
		self.habitat = self.atu.load_proctor_habitat(int(habitat_id), self.hab_set)
		self.explored_placements_in_current_habitat = []
		#breakpoint()
		# Launch a controller for the loaded habitat. If we already have a controller,
		# then reset it instead of loading a new one.
		if (self.controller == None):
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
		else:
			self.controller.reset(self.habitat)
			# self.reset_state()
			self.rnc.reset_state()
		# self.rnc.set_controller(self.controller)

		# Take a snapshot of all available positions- these won't change while we're in this habitat,
		# so no need to re-do them everytime we plan a path.
		# self.grid_size = self.controller.initialization_parameters["gridSize"]
		self.reachable_positions, self.unreachable_postions, self.full_grid, self.rooms_in_habitat = self.update_navigation_artifacts(
			self.habitat)

		# Now place the robot in a random position and figure out the target from there.
		self.choose_random_placement_in_habitat()

	# self.choose_specific_placement_in_habitat()
	# print("LH2")

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
	def choose_random_placement_in_habitat(self):
		# print("CH1")
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

			# We've just been put in a random place in a habitat. We want to move now to where we want to go,
			# e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
			try:
				point_for_room_search = (p[0], "", p[1])
				self.current_target_point = self.choose_target_point(place_with_rtn,
																	 point_for_room_search)  # self.target_room will be set in this function

				cur_pos = self.rnc.get_agent_pos_and_rotation()
				# print("Placement: ", place_with_rtn, " cur_pos: ", cur_pos, " el_ndx: ", el_ndx)
				self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
																				 self.current_target_point,
																				 self.reachable_positions,
																				 close_enough=self.plan_close_enough,
																				 step=self.grid_size)

				# Now let's remember the A* path- we will want it for results.
				(self.astar_path, _, self.path_start, self.path_dest) = self.nu.get_last_path_and_params()
				# print("AE: Path: ", self.astar_path)

				if isinstance(self, DoorFinder):
					# what is the room we start in
					self.starting_room = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

					if (self.starting_room == None): raise ValueError("Starting room not identifiable")

					# We must ensure that we navigate from one room to another
					if self.target_room == self.starting_room: raise ValueError("start and end points in same room")
			except ValueError as e:
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

	def initialize_connection(self, conn, command):
		hab_id = command.get("hab_id", "1")
		self.hab_set = command.get("hab_set", "train")

		# Initialize AI2-THOR controller on the server
		print(f"Initializing AI2-THOR for scene: {self.hab_set}[{hab_id}]...")
		self.load_habitat(hab_id)
		print("AI2-THOR initialized. Ready for actions.")

		# Send initial READY response
		response = {"status": "READY",
					"scene": self.hab_set + "[" + str(hab_id) + "]",
					"reachable_positions": self.reachable_positions,
					"unreachable_postions": list(self.unreachable_postions),
					"full_grid": self.full_grid}
		send_data(conn, json.dumps(response).encode(self.encoding))

	def execute_action(self, conn, command):
		action_from_dreamer = command.get('action', 'NO_ACTION')
		print(f"-> Received action: {action_from_dreamer}")

		# Execute action
		cur_obs = self.step(action_from_dreamer)
		send_data(conn, json.dumps(cur_obs).encode(self.encoding))

		## --- PROCESS AND SEND BACK ---
		## TODO: Make sure that the converted image to JPEG here is the same that we get when we unpack
		## before feeding it to Dreamer
		## a) Prepare Metadata
		#metadata = event.metadata
		#metadata['success'] = event.metadata['lastActionSuccess']
		## b) Prepare Frame (Convert numpy array to JPEG bytes)
		## Use CV2 to encode the numpy array as JPEG for efficient transfer
		#is_success, buffer = cv2.imencode(".jpg", event.frame)
		#if not is_success:
		#	raise Exception("Failed to encode frame to JPEG.")
		#frame_bytes = buffer.tobytes()
		## c) Combine and send (JSON metadata first, then image)
		## Send metadata
		#send_data(conn, json.dumps(metadata).encode(self.encoding))
		## Send image data
		#send_data(conn, frame_bytes)

	def step(self, action):
		# If this env has been retired (in evaluation mode we have evaluated everything already), then
		# don't actually do any stepping, but just return the previous obs
		if self.env_retired:
			return self.prev_obs

		if action['reset']:
			print('R', end='', sep='')
			# STORE EPISODE STATS:
			# A* path length, A* path, travelled path length, travelled path, habitat id, actions taken.
			if self.hab_set != "train":
				episode_stats = {
					"local_step": self.step_count_since_start,
					"steps_used": self.step_count_in_current_episode,
					"habitat_id": self.habitat_id,
					"bad_spot": self._bad_spot,
					"have_arrived": str(self.have_we_arrived(self.reward_close_enough)),
					"path_start": self.path_start,
					"path_dest": self.path_dest,
					"astar_path": self.astar_path,
					"travelled_path": self.travelled_path,
					"chosen_actions": self.chosen_actions,
				}
				# print(hab_exploration_stats)

				with open(self.logdir + "/episode_data.jsonl", "a") as f:
					f.write(json.dumps(episode_stats) + "\n")

			obs = self._reset()
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
				self._done = self.have_we_arrived(self.reward_close_enough)
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
		return obs

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
			is_first=np.bool(self.isFirst),
			is_last=np.bool(self._done),
			is_terminal=np.bool(self._done),
			# distance_left = np.float32(self.distance_left),
			# steps_after_room_change = np.float32(self.steps_in_new_room),
			# room_type = np.float32(self.room_type),
			distanceleft=np.float32(self.distance_left),
			stepsafterroomchange=np.float32(self.steps_in_new_room),
			roomtype=np.float32(self.room_type),
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
		with self.LOCK:
			self.load_next_start_point()
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

	def handle_client(self, conn, addr):
		print(f"✅ Connection established with {addr}")

		# keep reading commands from client and do what it wants
		while True:
			try:
				cmd_data_bytes = recv_data(conn)
				if not cmd_data_bytes:
					raise Exception("Client closed connection.")

				command = json.loads(cmd_data_bytes.decode(self.encoding))

				# here we handle what the client wants exactly
				if command.get("command") == "INIT":
					self.initialize_connection(conn, command)
				elif command.get("command") == "ACT":
					self.execute_action(conn, command)

			except Exception as e:
				print(f"Error handling client {addr}: {e}")
			finally:
				self.close()
				conn.close()
				print(f"Connection with {addr} closed.")

	def start_server(self):
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			server_socket.bind((self.host, self.port))
			server_socket.listen(5)
			print(f"AI2-THOR Server listening on {self.host}:{self.port}")

			while True:
				conn, addr = server_socket.accept()
				# Handle client connection in a new thread
				client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
				client_thread.start()

		except socket.error as e:
			print(f"Failed to start server: {e}")
			print("Ensure the port is not in use and firewall is open (as discussed previously).")
		finally:
			server_socket.close()

##
# Room centre finding task
##
class RoomCentreFinder(RemoteEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, actions, *args, **kwargs):
        super().__init__(actions, *args, **kwargs)

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

if __name__ == "__main__":
	rcf = RoomCentreFinder()
	rcf.start_server()