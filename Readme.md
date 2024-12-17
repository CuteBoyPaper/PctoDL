## PctoDL

`./src` is the source code of our work.

`./profile` is used to collect data and build predictive models

## Installation and Deployment Process

1. ### Installation of Libraries and Software: 

   - Install docker.

   ```
   apt install docker.io
   ```

   - Install NVIDIA Container Toolkit.

   ```
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

   - Pull docker images.

   ```
   docker pull nvcr.io/nvidia/tritonserver:24.10-py3
   docker pull nvcr.io/nvidia/tritonserver:24.10-py3-sdk
   ```

   - Install Python environment

   

   ```
   pip install -r requirements.txt
   ```

   ​	

2. ### Deployment of Code:

   - Pull PctoDL code.

   ```
   git clone https://github.com/CuteBoyPaper/PctoDL.git
   ```


   ### 3.Getting Started:

   run src/main.py for starting 

​			 