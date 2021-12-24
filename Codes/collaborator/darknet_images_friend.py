import argparse
import multiprocessing
import threading
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from multiprocessing import Process, Queue,Value,Array
import sys, time, socket
import subprocess

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="lab/img_src/img",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="cfg/yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def image_detection2(image_path, network, class_names, class_colors, thresh,val,i_max,time_limit):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    init_time = time.time()
    init_val = int(val.value)
    while(int(val.value) < i_max+1):

        image = cv2.imread(image_path+str(int(val.value))+'.jpg',cv2.IMREAD_COLOR)
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                        interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        val.value = val.value + 1
        rate = (time.time()-init_time)/int(val.value-init_val)
        if (time.time()-init_time)>(time_limit-rate):
            break

    return 0 
       

def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def sendImage(sock, buff,img_index,img_number,time_limit): # Send (Communication) Core
    # Inference and Sending Image List
    init_time = time.time()
    images = ["data/dog.jpg", "data/person.jpg"]  # for test

    index = int(img_index.value)
    image_path = images[index]
    image = cv2.imread(image_path)

   
    for i in range(img_number):
        #print(f"START: Offloading --- {i+1}")
        # Basic Preprocessing
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode(str(i+index)+'.jpg', image, encode_param)
        data = np.array(imgencode)
        stringData = data.tostring()

        # Send Image
        sock.send(str(len(stringData)).encode()) # send image size
        sock.recv(buff)
        sock.send(stringData) # send image file
    return


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def receive(sock, buff,image_num): # learning core
    ff = open('server_bps.txt','w')
    for i in range(image_num):
        # Image Receive
        start = time.time()
        length = sock.recv(buff)
        file_size = int(length)
            
        sock.sendall('ACK'.encode('utf-8'))
        stringData = recvall(sock, file_size)
        data = np.frombuffer(stringData, dtype='uint8')
        decimg = cv2.imdecode(data, 1)
        with open(f'temp2/{i+1}.jpg', 'wb') as f:
            f.write(stringData)
        stop = time.time()
        ff.write(str(file_size/(stop-start))+" ")
    ff.close()	
    ins_cpu_tem=(int(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000)
    ins_gpu_tem=(int(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000)
    f = open('cpu_tem.txt','a')
    f.write(str(ins_cpu_tem)+" ")
    f.close()
    f = open('gpu_tem.txt','a')
    f.write(str(ins_gpu_tem)+" ")
    f.close()

    return


def main():
    #image_num=int(input("Input image number: "))
    #time_interval=float(input("Input time slot size:(float) "))
    #increase_t=float(input("Input time slot increasing size:(float) "))
    #test_num = int(input("Input how many experimnet you perform with increasing time interval:( at least 1) "))
    #sleep_time2=int(input("How much time let nano be free between each experiment: (int)"))
    #cpulim = int(input("init cpu temperature:"))   
    #gpulim = int(input("init gpu temperature:"))
    #packet_size = float(input("Percentage that you expect from your partner's capablity for task: (least 100%) "))/100

    #with open("lab/condition.txt",'w') as f:
    #    f.write("Input image number: "+ str(image_num)+"\n"+
    #    "Input time slot size:(float) "+ str(time_interval)+"\n"+
    #    "Input time slot increasing size:(float)   "+ str(increase_t)+"\n"+
    #    "Input how many experimnet you perform with increasing time interval:( at least 1)  "+ str(test_num)+"\n"+
    #    "How much time let nano be free between each experiment: (int) "+ str(sleep_time2)+"\n"+
    #    "init cpu temperature: "+ str(cpulim)+"\n"+
    #    "init gpu temperature: "+ str(gpulim)+"\n"
    #    )
    
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096 # send 4096 bytes each time step
    # the ip address or hostname of the server, the receiver
    #host = "192.168.0.5"
    #host = "192.168.0.10" # LAN
    host = "192.168.0.20" # wireless
    #host = "127.0.1.1" # Local Host
    port = 5001

    args = parser()
    check_arguments_errors(args)
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    while(True):
        try:
            # bind section is moved into Ln 315 ~ 316 
            # connect
            s.listen()
            #print(f"[*] Listening as {host}:{port}")
            client_socket, address = s.accept()

            #print(f"[+] {address} is connected.")
            result = Queue()
            image_num = client_socket.recv(BUFFER_SIZE)
            client_socket.send('ACK'.encode('utf-8'))
            print(image_num)
            image_num = int(image_num.decode('utf-8'))
            receive(client_socket, BUFFER_SIZE,image_num)

        finally:
            pass
        #detection_list=[0]*image_num
    
        #bool1=jetson.temperature['CPU'] > cpulim
        #bool2=jetson.temperature['GPU'] > gpulim
        #while(bool1 or bool2):
        #    time.sleep(sleep_time2)
        #    bool1=jetson.temperature['CPU'] > cpulim
        #    bool2=jetson.temperature['GPU'] > gpulim

        image_path='temp2/'
        processed_image_num=0

        while(processed_image_num<image_num):
            image, detections = image_detection(image_path+str(processed_image_num+1)+".jpg", network, class_names, class_colors, args.thresh)
            processed_image_num+=1

        client_socket.send('done'.encode('utf-8'))
        
        for i in range(image_num):
            os.remove(image_path+str(i+1)+'.jpg')
    s.close()
    return 1    



if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
