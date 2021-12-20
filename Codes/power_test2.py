import subprocess
import time
import os
from jtop import jtop

eta = 0
i=0
test_num=5
heat_num=3
gpu_test_num = 4

mode_num=9    # 1 = corwork experiment, 2= no communication experiment, 3= inital tem increasing 4= same with 3 but more data , 7= dynamic model v1
time_period=1
jetson = jtop()
jetson.start()

# 1,2,3
#image num = 1000
#init time interval = 1
#increase t = 5
#test num = 10
#sleep time for cgpu temperature = 30  
#initial cpu tem < 47
#initial gpu tem < 45

# 4
#image num = 1000
#init time interval = 1
#increase time = 2
#increase tem =3
#test num = 50
#sleep time for cgpu temperature = 30  
#initial cpu tem < 45
#initial gpu tem < 43

for j in range(heat_num):
    cpu_power_list = []
    gpu_power_list = []
    device_power_list = []

    for mn in range(gpu_test_num):
        for i in range(test_num):
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
                    with open('lab/result'+str(mode_num)+'/instant_cpu_pow_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in ins_cpu_pow_list:
                            f.write(str(x)+' ')
                    with open('lab/result'+str(mode_num)+'/instant_gpu_pow_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in ins_gpu_pow_list:
                            f.write(str(x)+' ')
                    with open('lab/result'+str(mode_num)+'/instant_dev_pow_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in ins_dev_pow_list:
                            f.write(str(x)+' ')
                    with open('lab/result'+str(mode_num)+'/instant_cpu_tem_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in ins_cpu_tem_list:
                            f.write(str(x)+' ')
                    with open('lab/result'+str(mode_num)+'/instant_gpu_tem_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in ins_gpu_tem_list:
                            f.write(str(x)+' ')
                    with open('lab/result'+str(mode_num)+'/pow_time_list'+str(j)+"_"+str(mn)+"_"+str(i)+'.txt', 'w') as f:
                        for x in time_list:
                            f.write(str(x)+' ')
                    cpu_power_list.append(cpu_power*time_period)
                    gpu_power_list.append(gpu_power*time_period)
                    device_power_list.append(device_power*time_period)
                    break
                except FileNotFoundError:
                    pt=time.time()
                    time.sleep(time_period)
                    x=jetson.power
                    #instant_cpu_power = x[1]['5V CPU']['cur']
                    #instant_gpu_power =  x[1]['5V GPU']['cur']
                    #instant_device_power =  x[0]['cur']
                    instant_cpu_power = int(subprocess.check_output(["cat","/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input"],universal_newlines=True)[:-1])
                    instant_gpu_power = int(subprocess.check_output(["cat","/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input"],universal_newlines=True)[:-1])
                    instant_device_power =  int(subprocess.check_output(["cat","/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"],universal_newlines=True)[:-1])
                    cpu_power += instant_cpu_power
                    gpu_power += instant_gpu_power
                    device_power += instant_device_power
                    total_time+=time.time()-pt
                    time_list.append(total_time)
                    ins_cpu_pow_list.append(instant_cpu_power)
                    ins_gpu_pow_list.append(instant_gpu_power)
                    ins_dev_pow_list.append(instant_device_power)
                    #ins_cpu_tem_list.append(jetson.temperature['CPU'])
                    #ins_gpu_tem_list.append(jetson.temperature['GPU'])
                    ins_cpu_tem_list.append(int(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone1/temp"],universal_newlines=True)[:-1])/1000)
                    ins_gpu_tem_list.append(int(subprocess.check_output(["cat","/sys/devices/virtual/thermal/thermal_zone2/temp"],universal_newlines=True)[:-1])/1000)
                    #print(f"eta={eta}s \t cpu={cpu_power/eta:.3f}mW \t gpu={gpu_power/eta:.3f}mW \t device={device_power/eta:.3f}mW")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        with open('lab/result'+str(mode_num)+'/total_cpu_pow_list'+str(j)+str(mn)+"_"'.txt', 'w') as f:
            for x in cpu_power_list:
                f.write(str(x)+ ' ')
        with open('lab/result'+str(mode_num)+'/total_gpu_pow_list'+str(j)+str(mn)+"_"+'.txt', 'w') as f:
            for x in gpu_power_list:
                f.write(str(x)+ ' ')
        with open('lab/result'+str(mode_num)+'/total_dev_pow_list'+str(j)+str(mn)+"_"+'.txt', 'w') as f:
            for x in device_power_list:
                f.write(str(x)+ ' ')

