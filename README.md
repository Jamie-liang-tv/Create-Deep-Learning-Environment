# Create-Deep-Learning-Environment
Setup Environment for Deep Learning

#First, install graphic driver, (nvidia-smi)
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-450
sudo apt-get update
#Pure the driver
sudo apt-get purge nvidia*
#Reboot
sudo reboot

#Second, step install gcc (to prevent Failed to verify gcc version)
sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
sudo apt install gcc
gcc --version

# Do to https://developer.nvidia.com/cuda-11.0-download-archive to download suitable cuda (11.0l2)
wget https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo chmod +x cuda_11.0.2_450.51.05_linux.run
sudo sh cuda_11.0.2_450.51.05_linux.run # only check on CUDA Toolkit 11.0 it will work

################### Exoport path ################################
sudo nano /etc/profile.d/cuda.sh
export PATH=/usr/local/cuda-11.1/bin:$PATH
export CUDADIR=/usr/local/cuda-11.1
sudo chmod +x /etc/profile.d/cuda.sh
sudo nano /etc/ld.so.conf.d/cuda.conf
/usr/local/cuda-11.1/lib64 
sudo ldconfig 
#################### Another the way to add PATH Alway work
nano ~/.bashrc 
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
###################################################

# Download cuDNN suitable version #Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1#
install cudnn https://developer.nvidia.com/rdp/cudnn-archive 
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux
# extract cuDNN
tar -xvf cudnnXX
# Copy to Path
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Install Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7

# Install Virtual Environment
Cd to Home
mkdir venv_envs
cd venv_envs
sudo apt install python3-venv
python3 -m venv PT-TF24
source ~/PT-TF24/PT-TF24/bin/activate
pip install --upgrade pip
Install Tensorflow-gpu
pip install tensorflow-gpu==2.4.1 or pip install tensorflow-gpu==2.4.0

# Install Pytorch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Install jupyter
pip install jupyter
pip install jupyterlab

# Congratsss!







