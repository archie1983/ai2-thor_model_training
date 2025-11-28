# ai2thor_client.py (Run on Jetson Orin)

import socket
import json
import numpy as np
import cv2
import struct
import sys
import io

# --- Configuration ---
SERVER_IP = "192.168.0.32"  # !! REPLACE THIS !!
PORT = 9999
ENCODING = 'utf-8'


# ---------------------

# Utility function to ensure all data is sent
def send_data(sock, data):
	"""Sends a complete message (action command) to the server."""
	size_prefix = struct.pack("!I", len(data))
	sock.sendall(size_prefix)
	sock.sendall(data)


# Utility function to receive all data
def recv_data(sock):
	"""Receives a complete message from the server (prefixed by its size)."""
	# 1. Receive the 4-byte size prefix
	size_prefix = sock.recv(4)
	if not size_prefix:
		return None

	# Unpack the size (the '!' means network byte order, 'I' is unsigned integer)
	message_size = struct.unpack("!I", size_prefix)[0]

	# 2. Receive the actual data payload based on the size
	data = b''
	while len(data) < message_size:
		chunk = sock.recv(message_size - len(data))
		if not chunk:
			return None
		data += chunk
	return data


def run_client():
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	try:
		print(f"Attempting to connect to server at {SERVER_IP}:{PORT}...")
		client_socket.connect((SERVER_IP, PORT))
		print("✅ Connected to server.")

		# 1. SEND INITIAL COMMAND
		initial_command = {"command": "INIT", "scene": "FloorPlan1"}
		send_data(client_socket, json.dumps(initial_command).encode(ENCODING))

		# Await READY response
		init_response_bytes = recv_data(client_socket)
		if not init_response_bytes:
			raise Exception("Server failed to send initialization response.")

		init_response = json.loads(init_response_bytes.decode(ENCODING))
		if init_response.get("status") == "READY":
			print(f"Server initialized scene: {init_response.get('scene')}")
		else:
			raise Exception("Server reported initialization failure.")

		# 2. MAIN ACTION LOOP
		action_sequence = [
			{"action": "MoveAhead"},
			{"action": "RotateRight"},
			{"action": "MoveAhead"},
			{"action": "MoveRight", "moveMagnitude": 0.5}
		]

		for i, action_dict in enumerate(action_sequence):
			print(f"\nSending action {i + 1}: {action_dict}")

			# Send action
			send_data(client_socket, json.dumps(action_dict).encode(ENCODING))

			# Receive Metadata
			metadata_bytes = recv_data(client_socket)
			if not metadata_bytes: break
			metadata = json.loads(metadata_bytes.decode(ENCODING))

			# Receive Frame (JPEG bytes)
			frame_bytes = recv_data(client_socket)
			if not frame_bytes: break

			# Convert JPEG bytes back to numpy array (frame)
			np_array = np.frombuffer(frame_bytes, np.uint8)
			frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

			# Display results on the Jetson
			print(f"-> Action Success: {metadata.get('success')}")

			# Use OpenCV to display the received frame
			cv2.imshow("AI2-THOR Remote View (Jetson)", frame)
			# Wait for 1 millisecond and check for 'q' to quit
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# Example of how to access other data:
		# print(f"Agent Position: {metadata['agent']['position']}")

	except ConnectionRefusedError:
		print(f"❌ Connection Refused. Ensure server is running at {SERVER_IP}:{PORT} and firewall is open.")
	except Exception as e:
		print(f"An error occurred: {e}")
	finally:
		client_socket.close()
		cv2.destroyAllWindows()
		print("Client disconnected.")


if __name__ == "__main__":
	# Ensure you have the OpenCV window for displaying the frame
	print("Press 'q' in the display window to quit.")
	run_client()