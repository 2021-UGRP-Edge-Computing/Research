import subprocess
import time
import os
from jtop import jtop

eta = 0
i=0
test_num=10																																																																																																																																		
mode_num=3    # 1 = corwork experiment, 2= no communication experiment
time_period=1
jetson = jtop()
jetson.start()

#image num = 1000
#init time interval = 1
#increase t = 5
#test num = 10
#sleep time for cgpu temperature = 300  
#initial cpu tem < 43
#initial gpu tem < 41
cpu_power_list = []
gpu_power_list = []
device_power_list = []

while i<test_num:
    while True:
        try:
            with open('temp/start.txt', 'r') as f:     
                pass
            os.remove('temp/start.txt')
            break
        except FileNotFoundError:
            pass
    cpu_power=0
    gpu_power=0
    device_power=0
    total_time=0
    time_list=[]
    ins_cpu_pow_list=[]
    ins_gpu_pow_list=[]
    ins_dev_pow_list=[]
    ins_cpu_tem_list=[]
    ins_gpu_tem_list=[]

    while True:
        try:
            with open('temp/end.txt', 'r') as f:     
                pass
            os.remove('temp/end.txt')
            with open('lab/result'+str(mode_num)+'/instant_cpu_pow_list'+str(i)+'.txt', 'w') as f:
                for x in ins_cpu_pow_list:
                    f.write(str(x)+' ')
            with open('lab/result'+str(mode_num)+'/instant_gpu_pow_list'+str(i)+'.txt', 'w') as f:
                for x in ins_gpu_pow_list:
                    f.write(str(x)+' ')
            with open('lab/result'+str(mode_num)+'/instant_dev_pow_list'+str(i)+'.txt', 'w') as f:
                for x in ins_dev_pow_list:
                    f.write(str(x)+' ')
            with open('lab/result'+str(mode_num)+'/instant_cpu_tem_list'+str(i)+'.txt', 'w') as f:
                for x in ins_cpu_tem_list:
                    f.write(str(x)+' ')
            with open('lab/result'+str(mode_num)+'/instant_gpu_tem_list'+str(i)+'.txt', 'w') as f:
                for x in ins_gpu_tem_list:
                    f.write(str(x)+' ')
            with open('lab/result'+str(mode_num)+'/pow_time_list'+str(i)+'.txt', 'w') as f:
                for x in time_list:
                    f.write(str(x)+' ')
            cpu_power_list.append(cpu_power*time_period)
            gpu_power_list.append(gpu_power*time_period)
            device_power_list.append(device_power*time_period)
            i+=1
            break
        except FileNotFoundError:
            pt=time.time()
            time.sleep(time_period)
            x=jetson.power
            instant_cpu_power = x[1]['5V CPU']['cur']
            instant_gpu_power =  x[1]['5V GPU']['cur']
            instant_device_power =  x[0]['cur']
            cpu_power += instant_cpu_power
            gpu_power += instant_gpu_power
            device_power += instant_device_power
            total_time+=time.time()-pt
            time_list.append(total_time)
            ins_cpu_pow_list.append(instant_cpu_power)
            ins_gpu_pow_list.append(instant_gpu_power)
            ins_dev_pow_list.append(instant_device_power)
            ins_cpu_tem_list.append(jetson.temperature['CPU'])
            ins_gpu_tem_list.append(jetson.temperature['GPU'])
            #print(f"eta={eta}s \t cpu={cpu_power/eta:.3f}mW \t gpu={gpu_power/eta:.3f}mW \t device={device_power/eta:.3f}mW")
																																																																																																																																																																																																								
with open('lab/result'+str(mode_num)+'/total_cpu_pow_list.txt', 'w') as f:
    for x in cpu_power_list:
        f.write(str(x)+ ' ')
with open('lab/result'+str(mode_num)+'/total_gpu_pow_list.txt', 'w') as f:
    for x in gpu_power_list:
        f.write(str(x)+ ' ')
with open('lab/result'+str(mode_num)+'/total_dev_pow_list.txt', 'w') as f:
    for x in device_power_list:
        f.write(str(x)+ ' ')

