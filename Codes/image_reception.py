from multiprocessing import Process, Queue
import sys, time, socket
import subprocess, cv2
#import OptimizedServer
#import receive_image as RI

import numpy as np

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def receive(sock, buff): # learning core
	
	#Inference Image List	
	#images = ["data/dog.jpg", "data/person.jpg"]

	for i in range(10):
		# Image Receive
		start = time.time()
		length = sock.recv(buff)
		file_size = int(length)
		sock.send('OK'.encode())

		stringData = recvall(sock, file_size)
		
		data = np.frombuffer(stringData, dtype='uint8')
		decimg = cv2.imdecode(data, 1)
		# Image Inference
		with open(f'data/{i+1}.jpg', 'wb') as f:
			f.write(stringData)

		stop = time.time()
		
		print(f"DONE -- RECEIVE -- {file_size/(stop-start)}")
	return


def main():
	SEPARATOR = "<SEPARATOR>"
	BUFFER_SIZE = 4096 # send 4096 bytes each time step
	# the ip address or hostname of the server, the receiver
	#host = "192.168.0.5"
	#host = "192.168.0.10" # LAN
	host = "192.168.0.18" # wireless
	#host = "127.0.0.1" # Local Host
	port = 5000


	try:
		# connect
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind((host, port))
		s.listen(5)
		print(f"[*] Listening as {host}:{port}")

		client_socket, address = s.accept()
		print(f"[+] {address} is connected.")
		result = Queue()
	
		# TASK ALLOCATION SECTION #
		receive(client_socket, BUFFER_SIZE)
		# TASK ALLOCATION SECTION #

	finally:
		s.close()

	return 0


if __name__ == '__main__':
	main()
