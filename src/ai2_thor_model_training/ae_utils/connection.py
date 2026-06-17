import struct
# Utility function to ensure all data is sent
def send_data(sock, data):
	"""Sends a complete message (action command) to the server."""
	size_prefix = struct.pack("!I", len(data))
	sock.sendall(size_prefix)
	if len(data) < 1000:
		print("OUT: ", data)
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