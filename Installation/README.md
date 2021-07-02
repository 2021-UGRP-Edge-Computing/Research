# INSTALLATION

## OpenCV Installation

``` bash
check out from https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html

# Install minimal prerequisites (Ubuntu 18.04 as reference)
$sudo apt-get update
$sudo apt update && sudo apt install -y cmake g++ wget unzip


# Download and unpack sources
$wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
$unzip opencv.zip

# Create build directory
$mkdir -p build && cd build

# Configure
$cmake  ../opencv-master

# Build
$cmake --build .

```

## Pandas Installation

``` bash
check out from https://varhowto.com/install-pandas-ubuntu-20-04/#pandas_Ubuntu_20_04

# Install python3-pandas system package
$sudo apt install python3-pandas

# Install panda’s documentation package
$sudo apt install python-pandas-doc

```

## Pandas Installation

``` bash
check out from https://phoenixnap.com/kb/how-to-install-keras-on-linux

# Install and Update Python3 and Pip
$sudo apt install python3 python3-pip
$sudo pip3 install ––upgrade pip

# Upgrade Setuptools
$pip3 install ––upgrade setuptools

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. launchpadlib 1.10.6 requires testresources, which is not installed.

$sudo apt install python3-testresources
$pip3 install ––upgrade setuptools
```

## Install TensorFlow
``` bash
check out from https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

#Install JetPack on your Jetson device.
https://developer.nvidia.com/embedded/jetpack

#Install system packages required by TensorFlow:
$sudo apt-get update
$sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

#Install and upgrade pip3.
$sudo apt-get install python3-pip
$sudo pip3 install -U pip testresources setuptools==49.6.0 

#Install the Python package dependencies.
$sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

#Install TensorFlow using the pip3 command. This command will install the latest version of TensorFlow compatible with JetPack 4.5.
$sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow

#Upgrading TensorFlow
$sudo pip3 install --upgrade --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$45 tensorflow

#Test if there are any execution error
$python3
$import tensorflow





```
