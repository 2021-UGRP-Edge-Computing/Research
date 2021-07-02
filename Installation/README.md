# INSTALLATION

## OpenCV Installation

``` bash
check out from https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html

# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt-get update
sudo apt update && sudo apt install -y cmake g++ wget unzip


# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip

# Create build directory
mkdir -p build && cd build

# Configure
cmake  ../opencv-master

# Build
cmake --build .

```

## Pandas Installation

``` bash
check out from https://varhowto.com/install-pandas-ubuntu-20-04/#pandas_Ubuntu_20_04

# Install python3-pandas system package
sudo apt install python3-pandas

# Install panda’s documentation package
sudo apt install python-pandas-doc

```

## Pandas Installation

``` bash
check out from https://phoenixnap.com/kb/how-to-install-keras-on-linux

# Install and Update Python3 and Pip
sudo apt install python3 python3-pip
sudo pip3 install ––upgrade pip

# Upgrade Setuptools
pip3 install ––upgrade setuptools

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. launchpadlib 1.10.6 requires testresources, which is not installed.

sudo apt install python3-testresources
pip3 install ––upgrade setuptools

# Install TensorFlow
pip3 install tensorflow






```
