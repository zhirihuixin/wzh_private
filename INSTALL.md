# Installing PytorchEveryThing

This document covers how to install PytorchEveryThing, its dependencies (including Pytorch), and the COCO dataset.

- For general information about PytorchEveryThing, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python3.5
- Pytorch-0.4, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- PytorchEveryThing has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.


## Python3.5

To install Python3.5 and soft link to `python3`.

1. Add source:

```
sudo apt-get install python-software-properties

sudo apt-get install software-properties-common

sudo add-apt-repository ppa:fkrull/deadsnakes

sudo apt-get update
```

2. Install python3.5:

```
sudo apt-get install python3.5

sudo apt-get install python3.5-dev

sudo apt-get install python3.5-tk
```

3. Soft link to `python3` and check (you can use `python2` for `python2.7`):

```
sudo rm -r /usr/bin/python3

sudo ln -s /usr/bin/python3.5 /usr/bin/python3

python3 -V
```

4. Install `pip3` and upgrade (if you want to user pip to install packages for python2, please use `pip2`):

```
sudo apt-get install python3-pip

sudo pip3 install --upgrade pip
```

5. Install basic packages

```
sudo pip3 install setuptools

sudo pip3 install numpy scipy scikit-image matplotlib six
```

   **Note:** If some packages cannot be installed, you can go into `/usr/local/lib/python3.5/dist-packages/` or `/usr/lib/python3/dist-packages/` or `/usr/lib/python3.5/` to delete the old version by `sudo`.


6. Wrapper `opencv3.4` to `python3.5`:

```
sudo pip3 install opencv-python
```

For install opencv3.5, please refer to [`Compile_opencv3.md`](https://github.com/soeaver/environment_config/blob/master/Compile_opencv3.md)


## Pytorch and torchvision

Install Pytorch with CUDA support.

1. Install Pytorch-0.4.0:

```
sudo pip3 install torch==0.4.0
```

2. Install torchvision-0.2.1:

```
sudo pip3 install torchvision
```


## PytorchEveryThing

1. Clone the PytorchEveryThing repository:

```
# PytorchEveryThing=/path/to/clone/PytorchEveryThing
git clone https://github.com/soeaver/PytorchEveryThing $PytorchEveryThing
```

2. Install Python dependencies:

```
sudo pip3 install -r $PytorchEveryThing/requirements.txt
```

3. Set up `pet`:

```
cd $PytorchEveryThing/pet

sh ./make.sh
```

   **Note:** If you get `CompileError: command 'x86_64-linux-gnu-gcc'`, please:

```
export CXXFLAGS="-std=c++11"

export CFLAGS="-std=c99"

sh ./make.sh
```

   **Note:** If you get `name 'unicode' is not defined` when use cocoapi, please:

```
sudo vim /usr/local/lib/python3.5/dist-packages/pycocotools/coco.py

change 'if type(resFile) == str or type(resFile) == unicode:' to
'if type(resFile) == str:' in line num 308

```
