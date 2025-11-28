# ai2thor_server.py (Run on X86 Laptop)

import socket
import threading
import json
import numpy as np
import cv2
import sys
from ai2thor.controller import Controller
from connection import (recv_data, send_data)

class RemoteEnv:
	def __init__(self, host = '0.0.0.0', port = 9999, encoding = 'utf-8'): # Listen on all available network interfaces by default
		self.host = host
		self.port = port
		self.encoding = encoding

	def handle_client(self, conn, addr):
		print(f"✅ Connection established with {addr}")
		thor_controller = None

		try:
			# 1. RECEIVE INITIAL COMMAND (e.g., {"command": "INIT", "scene": "FloorPlan1"})
			init_data_bytes = recv_data(conn)
			if not init_data_bytes:
				raise Exception("Client closed connection during initialization.")

			init_command = json.loads(init_data_bytes.decode(self.encoding))

			if init_command.get("command") == "INIT":
				scene_name = init_command.get("scene", "FloorPlan1")

				# Initialize AI2-THOR controller on the server
				print(f"Initializing AI2-THOR for scene: {scene_name}...")
				thor_controller = Controller(
					start_unity=True, # is the default and correct setting for the server
					scene=scene_name,
					width=300,  # Smaller resolution for faster transfer
					height=300,
					rotateBy=45,
					gridSize=0.25,
				)
				print("AI2-THOR initialized. Ready for actions.")

				# Send initial READY response
				response = {"status": "READY", "scene": scene_name}
				send_data(conn, json.dumps(response).encode(self.encoding))

			# 2. MAIN ACTION LOOP
			while True:
				# Receive action dictionary from client
				action_bytes = recv_data(conn)
				if not action_bytes:
					print("Client disconnected.")
					break

				action_dict = json.loads(action_bytes.decode(self.encoding))
				action_name = action_dict.get('action', 'NO_ACTION')
				print(f"-> Received action: {action_name}")

				# Execute action
				event = thor_controller.step(action_dict)

				# --- PROCESS AND SEND BACK ---

				# a) Prepare Metadata
				metadata = event.metadata
				metadata['success'] = event.metadata['lastActionSuccess']

				# b) Prepare Frame (Convert numpy array to JPEG bytes)
				# Use CV2 to encode the numpy array as JPEG for efficient transfer
				is_success, buffer = cv2.imencode(".jpg", event.frame)
				if not is_success:
					raise Exception("Failed to encode frame to JPEG.")

				frame_bytes = buffer.tobytes()

				# c) Combine and send (JSON metadata first, then image)
				# Send metadata
				send_data(conn, json.dumps(metadata).encode(self.encoding))

				# Send image data
				send_data(conn, frame_bytes)

		except Exception as e:
			print(f"Error handling client {addr}: {e}")
		finally:
			if thor_controller:
				thor_controller.stop()
				print("AI2-THOR controller stopped.")
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


if __name__ == "__main__":
	re = RemoteEnv()
	re.start_server()