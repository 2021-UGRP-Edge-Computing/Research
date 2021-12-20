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
from jtop import jtop

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

def sendImage(sock, buff,img_number): # Send (Communication) Core
    # Inference and Sending Image List
    init_time = time.time()

    #images = ['dog.jpg', 'horses.jpg', 'eagle.jpg', 'scream.jpg']
    images = ['big1.jpg', 'big2.jpg', 'big3.jpg', 'big4.jpg']
    image_path = 'lab/img_src/img/'

    sock.send(str(img_number).encode('utf-8')) #1
    print(f"{img_number}  | {str(img_number).encode('utf-8')}")
    sock.recv(buff) #1

    f = open('client_bps.txt','w')
    for i in range(img_number):
        img_index = random.randint(0, 3)
        #print(f"START: Offloading --- {i+1}")
        # Basic Preprocessing
        image = cv2.imread(image_path+str(images[img_index]))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', image, encode_param)
        data = np.array(imgencode)
        stringData = data.tostring()

        # Send Image
        
        p = time.time()
        sock.send(str(len(stringData)).encode()) # send image size #2
        sock.recv(buff) #2
        sock.sendall(stringData) # send image file #3
        s = time.time()
        f.write(str(len(stringData)/(s-p))+" ")
    f.close()

    return

def main():
    image_num = int(input("Input image number: 3000")) #3000
    time_interval = float(input("Input init time interval size:(float) 35")) # 35
    increase_delta = float(input("Input time slot increasing size:(float) 10")) # 10
    ti_test_num = int(input("Input how many experimnet you perform with increasing time interval:( at least 1) 5")) # 5
    sleep_time2 = int(input("How much time let nano take a rest to match temperature restriction: (int)30")) # 30
    heating_time = int(input("input Heating time to match the cpu,gpu temperature limit:30")) # 30
    heat_num = int(input("input how many experimnet you perform with increasing cpu, gpu heating:3")) # 3
    increase_delta = float(input("Input inital cpu,gpt temperature increasing size:(float) 3")) # 3
    cpulim = int(input("initial minimal cpu temperature:51"))   # 51
    gpulim = int(input("initial minimal gpu temperature:48"))   # 48
    packet_size = float(input("Percentage that you expect from your partner's capablity for task: (100%) 100"))/100 # 100
    ref_gpu= float(input("Cooperation Reference gpu temperature:53"))   # 53
    gpu_test_num = int(input("Test number of Cooperation Reference gpu temperature:( at least 1) 4")) # 4
    ref_gpu_delta = float(input("Increment of Cooperation Reference gpu temperature: 5")) # 5

    with open("lab/condition.txt",'w') as f:
        f.write("Input image number: "+ str(image_num)+"\n"+
        "Input time slot size:(float) "+ str(time_interval)+"\n"+
        "Input time slot increasing size:(float)   "+ str(increase_delta)+"\n"+
        "Input how many experimnet you perform with increasing time interval:( at least 1)  "+ str(ti_test_num)+"\n"+
        "How much time let nano take a rest to match temperature restriction: (int) "+ str(sleep_time2)+"\n"+
        "input Heating time to match the cpu,gpu temperature limit: (int) "+ str(heating_time)+"\n"+
        "input how many experimnet you perform with increasing cpu, gpu heating: "+ str(heat_num)+"\n"+
        "Input inital cpu,gpt temperature increasing size:(float)  "+ str(increase_delta)+"\n"+
        "init cpu temperature: "+ str(cpulim)+"\n"+
        "init gpu temperature: "+ str(gpulim)+"\n"+
        "Percentage that you expect from your partner's capablity for task: (100%) "+ str(packet_size)+"\n"+
        "Cooperation Reference gpu temperature:"+ str(ref_gpu)+"\n"+
        "Test number of Cooperation Reference gpu temperature:( at least 1) "+ str(gpu_test_num)+"\n"+
        "Increment of Cooperation Reference gpu temperature: "+ str(ref_gpu_delta)+"\n"
        )
    
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096 # send 4096 bytes each time step
	# the ip address or hostname of the server, the receiver
	#host = "192.168.0.5"
	#host = "192.168.0.10" # LAN
	#host = "192.168.0.17" # wireless
    #host = "127.0.1.1"
    host = "192.168.0.20"
    port = 5001

    # Image Data
    images = ['big1.jpg', 'big2.jpg', 'big3.jpg', 'big4.jpg']
    #images = ['eagle.jpg', 'dog.jpg', 'horses.jpg', 'scream.jpg']
    img_index = 0

    args = parser()
    check_arguments_errors(args)
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    
    time_interval_list = []
    for i in range(ti_test_num):
        time_interval_list.append(time_interval+i*increase_delta)

    tslist_file = open("lab/result9/Time_interval_list.txt",'w')
    for m in time_interval_list:
        tslist_file.write(str(m)+" ")
    tslist_file.close()

    #jetson = jtop()
    #jetson.start()
    for j in range(heat_num):
        for x in range(gpu_test_num):
            experiment_x_avg_fps=[]
            img_index = 0
            tol=2.0
            for i in range(ti_test_num):

                #ct=jetson.temperature
                ctc=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000
                ctg=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000
                
                #bool1=ct['CPU'] > cpulim + j*increase_h + tol
                #bool2=ct['GPU'] > gpulim + j*increase_h + tol
                #bool3=ct['CPU'] < cpulim + j*increase_h - tol
                #bool4=ct['GPU'] < gpulim + j*increase_h - tol
                bool1=ctc > cpulim + j*increase_delta + tol
                bool2=ctg > gpulim + j*increase_delta + tol
                bool3=ctc < cpulim + j*increase_delta - tol
                bool4=ctg < gpulim + j*increase_delta - tol
                while((bool1 or bool2)or(bool3 or bool4)):

                    if(bool1 or bool2):
                        print("===============================")
                        print('nano need to rest........')
                        #print('cpu temerature :',ct['CPU'])
                        #print('gpu temerature :',ct['GPU'])
                        print('cpu temerature :',ctc)
                        print('gpu temerature :',ctg)
                        print("===============================")
                        time.sleep(sleep_time2)
                        #ct=jetson.temperature
                        ctc=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000
                        ctg=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000
                
                        #bool1=ct['CPU'] > cpulim + j*increase_h + tol
                        #bool2=ct['GPU'] > gpulim + j*increase_h + tol
                        #bool3=ct['CPU'] < cpulim + j*increase_h - tol
                        #bool4=ct['GPU'] < gpulim + j*increase_h - tol
                        bool1=ctc > cpulim + j*increase_delta + tol
                        bool2=ctg > gpulim + j*increase_delta + tol
                        bool3=ctc < cpulim + j*increase_delta - tol
                        bool4=ctg < gpulim + j*increase_delta - tol
                    else:
                        print("===============================")
                        print('nano need to be hot........')
                        #print('cpu temerature :',ct['CPU'])
                        #print('gpu temerature :',ct['GPU'])
                        print('cpu temerature :',ctc)
                        print('gpu temerature :',ctg)
                        print("===============================")
                        t=0
                        zt=time.time()
                        while(t<heating_time):
                            image, detections = image_detection(
                                    "lab/img_src/big.jpg", network, class_names, class_colors, args.thresh
                                    )
                            t = time.time()-zt
                        #ct=jetson.temperature
                        ctc=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000
                        ctg=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000
                        bool1=ctc > cpulim + j*increase_delta + tol
                        bool2=ctg > gpulim + j*increase_delta + tol
                        bool3=ctc < cpulim + j*increase_delta - tol
                        bool4=ctg < gpulim + j*increase_delta - tol
                        
                print("=========== start ============")
                instant_fps = []
                instant_time = [0]
                cor_timing=[]
                processed_imagenum=0
                image_path='lab/img_src/img/'
                context = True
                task=0
                history=0
                fri_perf=1.0
                my_perf=0.0
                with open("temp/start.txt",'w') as f:
                        pass
                zero_time=time.time()
                while(processed_imagenum < image_num):
                    if context:
                        tem_cpu=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000
                        tem_gpu=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000
                        condition1=tem_gpu < ref_gpu+x*ref_gpu_delta
                        condition2=my_perf > fri_perf
                        #condition1= processed_imagenum < image_num
                        #condition2= processed_imagenum < image_num
                        
                        prev_time = time.time()
                        while(condition1 or condition2):
                            t1 = time.time()
                            inn=100       
                            if processed_imagenum>image_num-1:
                                break
                            if processed_imagenum>image_num-inn:
                                inn=image_num-processed_imagenum
                            for jnn in range(inn):
                                img_index = random.randint(0, 3)
                                image, detections = image_detection(
                                        image_path+str(images[img_index]), network, class_names, class_colors, args.thresh
                                        )
                            processed_imagenum+=inn
                            tem_gpu=float(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000
                            t2 = time.time()
                            my_perf = inn/(t2-t1)
                            condition1=tem_gpu < ref_gpu+x*ref_gpu_delta
                            condition2=my_perf > fri_perf
                            #condition1= processed_imagenum < image_num
                            #condition2= processed_imagenum < image_num
                            #time4=time.time()
                            #totalt=time4-t1
                            #pret1=time2-time1
                            #inft=time3-time2
                            #temacct=t2-time3
                            #cont=time4-t2
                            #print("pre1 time:"+str((pret1/totalt)*100)+"%\n"+ 
                            #      "inference time:"+str((inft/totalt)*100)+"%\n"+
                            #      "temerature access time:"+str((temacct/totalt)*100)+"%\n"+
                            #      "condition check time:"+str((cont/totalt)*100)+"%\n"
                            #    )

                        end_time = time.time()
                        #instant_fps.append(((processed_imagenum - history)/(end_time - prev_time)))
                        #instant_time.append(instant_time[-1]+end_time - prev_time)
                        context = False
                        history=processed_imagenum

                    else:
                        cor_timing.append(time.time()-zero_time)
                        task = int(my_perf*time_interval_list[i])
                        task_size=1
                        if int(task *packet_size)+history <image_num:
                            task_size = int(task *packet_size)
                        else:
                            task_size = image_num-history

                            
                        prev_time = time.time()
                        """ ###################### """
                        s = socket.socket()
                        print(f"[+] Connecting to {host}:{port}")
                        s.connect((host, port))
                        print("[+] Connected.")
                        sendImage(s, BUFFER_SIZE,task_size)
                        # Receive Result
                        #for i in range(task_size):
                        #    data = s.recv(BUFFER_SIZE)
                        done = s.recv(BUFFER_SIZE)
                        end_time = time.time()
                        s.sendall("CLOSE".encode())
                        s.recv(BUFFER_SIZE)
                        s.close()
                        end_time = time.time()
                        processed_imagenum+=task_size
                        context = True
                        """ ###################### """
                        fri_perf=task_size/(end_time - prev_time)
                    
                        #instant_fps.append(task_size/(end_time - prev_time))
                        #instant_time.append(instant_time[-1]+end_time - prev_time)
                        history=processed_imagenum

                experiment_x_avg_fps.append(processed_imagenum/(time.time() - zero_time))

                #instant_ff = open("lab/result9/instant_fps_list"+ str(j) + "_" + str(i)+"_"+str(x)+".txt",'w')
                #for k in instant_fps:
                #    instant_ff.write(str(k)+" ")
                #instant_ff.close()

                instant_tf = open("lab/result9/instant_time_list" + str(j) + "_" + str(i)+"_"+str(x)+".txt",'w')
                for k in instant_time:
                    instant_tf.write(str(k)+" ")
                instant_tf.close()

                instant_ctf = open("lab/result9/cor_time_list" + str(j) + "_" + str(i)+"_"+str(x)+".txt",'w')
                for k in cor_timing:
                    instant_ctf.write(str(k)+" ")
                instant_tf.close()

                with open("temp/end.txt","w") as f:
                    pass

            fpslist_file = open("lab/result9/Fps_list"+str(j)+"_"+str(x)+".txt",'w')
            for m in experiment_x_avg_fps:
                fpslist_file.write(str(m)+" ")
            fpslist_file.close()
        
    #jetson.close()    

    return 1    

if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
