# OpenCV in C++ with Jetson Nano
## File Structure
```
vision ─ build ─ run.sh
       ├ CMakeLists.txt
       └ main.cpp
```

## Build & Execution
```
# Check the installation of OpenCV
$ python3
>>> import cv2
>>> exit()

# Make a copy of the attached folder named 'vision'
# and open terminal
$ cd the/path/to/vision
$ cd build

# Build, Compile & Execution 
$ sh run.sh
```



# OpenCV in Python with Jetson Nano
## Import
```
import cv2
```


# Object Detection in Python with Jetson Nano

## Setup

First, check the nvcc version.
```shell
nvcc --version
```

If your nano cannot find the path of nvcc, you should set the environment variable for the nvcc.
```shell
# open terminal
$ vim ~/.bashrc

# add the following two lines after the last lines
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# Log off and log on / or restart nano
```

## File Structure
```
darknet─ build ─ run.sh
       ├ CMakeLists.txt
       └ main.cpp

```

# Object Detection in Python with Jetson Nano

# Yolo compile on linux 

## (using cmake)
```
$git clone https://github.com/AlexeyAB/darknet
$cd darknet
$mkdir build_release
$cd build_release
$cmake ..

-may have an error:
    if it is about the cmake version, use this.
        cmake download website: https://cmake.org/download/

        $wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0.tar.gz
        $tar -xvf cmake-3.21.0.tar.gz
        $cd cmake-3.21.0
        $./bootstrap
   
        - may have an error:
            if it is about the OpenSSL development package installation, use this.
            $sudo apt install libssl-dev
            or
            $sudo apt-get install libssl-dev
        
        $make
        $sudo make install 
        

$cmake --build . --target install --parallel 8

```
# Yolo compile on linux (using make)

## using make
```
$git clone https://github.com/AlexeyAB/darknet
$cd darknet
$make
