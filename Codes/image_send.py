from multiprocessing import Process, Queue
import sys, time, socket
import subprocess, cv2
#import OptimizedClient
#import send_image as SI
#import receive_image as RI

import numpy as np
import time

def sendImage(sock, buff): # Send (Communication) Core
	# Inference and Sending Image List
	images = ["data/dog.jpg", "data/person.jpg"]  # for test

	index = 0
	image_path = images[index]
	image = cv2.imread(image_path)
	
	
	for i in range(10):
		#print(f"START: Offloading --- {i+1}")
		# Basic Preprocessing
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
		result, imgencode = cv2.imencode('.jpg', image, encode_param)
		data = np.array(imgencode)
		stringData = data.tostring()
	
		# Send Image
		sock.send(str(len(stringData)).encode()) # send image size
		sock.recv(buff)
		sock.send(stringData) # send image file

		# Receive Result
		# data = sock.recv(buff)
		#if data:
		#	result = SI.pickle.loads(data)

		#SI.cv2.imshow('color', image)
		#SI.destroyWindow('color')
		
		print(f"DONE -- RECEPTION : {i+1}")
	return


def main():
	SEPARATOR = "<SEPARATOR>"
	BUFFER_SIZE = 4096 # send 4096 bytes each time step
	# the ip address or hostname of the server, the receiver
	#host = "192.168.0.5"
	#host = "192.168.0.10" # LAN
	#host = "192.168.0.17" # wireless
	host = "127.0.0.1" # Local Host
	port = 5001


	try:
                # connect
		s = socket.socket()
		print(f"[+] Connecting to {host}:{port}")
		s.connect((host, port))
		print("[+] Connected.")

		result = Queue()
		# TASK ALLOCATION SECTION #
		sendImage(s, BUFFER_SIZE)
		# TASK ALLOCATION SECTION #

	finally:
		#client_socket.close()
		pass
	

	return 0
	


if __name__ == '__main__':
	main()
