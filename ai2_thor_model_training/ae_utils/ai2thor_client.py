# ai2thor_client.py (Run on Jetson Orin)

import socket
import json
import numpy as np
import cv2
from connection import recv_data, send_data

class AI2ThorClient:
	def __init__(self, server_ip = '0.0.0.0', port = 9999, encoding = 'utf-8'): # Listen on all available network interfaces by default
		self.server_ip = server_ip
		self.port = port
		self.encoding = encoding

		self.reachable_positions = None
		self.unreachable_postions = None
		self.full_grid = None

	def run_client(self):
		client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		try:
			print(f"Attempting to connect to server at {self.server_ip}:{self.port}...")
			client_socket.connect((self.server_ip, self.port))
			print("✅ Connected to server.")

			# 1. SEND INITIAL COMMAND
			initial_command = {"command": "INIT", "hab_id": "83", "hab_set": "test"}
			send_data(client_socket, json.dumps(initial_command).encode(self.encoding))

			# Await READY response
			init_response_bytes = recv_data(client_socket)
			if not init_response_bytes:
				raise Exception("Server failed to send initialization response.")

			init_response = json.loads(init_response_bytes.decode(self.encoding))
			if init_response.get("status") == "READY":
				print(f"Server initialized scene: {init_response.get('scene')}")
				self.reachable_positions = init_response.get("reachable_positions")
				self.unreachable_postions = init_response.get("unreachable_postions")
				self.full_grid = init_response.get("full_grid")

				#print("self.reachable_positions: ", self.reachable_positions)
			else:
				raise Exception("Server reported initialization failure.")

			# 2. MAIN ACTION LOOP
			action_sequence = [
				{"command": "ACT", "action_bits": {"action": 0, "reset": False}},
				{"command": "ACT", "action_bits": {"action": 1, "reset": False}},
				{"command": "ACT", "action_bits": {"action": 0, "reset": False}},
				{"command": "ACT", "action_bits": {"action": 2, "reset": False}}
			]

			for i, action_dict in enumerate(action_sequence):
				print(f"\nSending action {i + 1}: {action_dict}")

				# Send action
				send_data(client_socket, json.dumps(action_dict).encode(self.encoding))

				# Receive Metadata
				metadata_bytes = recv_data(client_socket)
				if not metadata_bytes: break
				metadata = json.loads(metadata_bytes.decode(self.encoding))

				# Receive Frame (JPEG bytes)
				frame_bytes = recv_data(client_socket)
				if not frame_bytes: break

				# Convert JPEG bytes back to numpy array (frame)
				np_array = np.frombuffer(frame_bytes, np.uint8)
				frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

				# Display results on the Jetson
				print(f"-> Action metadata: {metadata}")
				print(len(np_array))

				# Use OpenCV to display the received frame
				#cv2.imshow("AI2-THOR Remote View (Jetson)", frame)
				# Wait for 1 millisecond and check for 'q' to quit
				#if cv2.waitKey(1) & 0xFF == ord('q'):
				#	break

			# Example of how to access other data:
			# print(f"Agent Position: {metadata['agent']['position']}")
		except ConnectionRefusedError:
			print(f"❌ Connection Refused. Ensure server is running at {self.server_ip}:{self.port} and firewall is open.")
		except Exception as e:
			print(f"An error occurred: {e}")
		finally:
			client_socket.close()
			#cv2.destroyAllWindows()
			print("Client disconnected.")


if __name__ == "__main__":
	# Ensure you have the OpenCV window for displaying the frame
	atc = AI2ThorClient(server_ip = '192.168.0.32')
	print("Press 'q' in the display window to quit.")
	atc.run_client()
