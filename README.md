# Install Pytorch 1.8.0 and Tensorflow-gpu 2.4.1 on Ubuntu20.04
 Also works on Ubuntu18.04
# First, install graphic driver (nvidia-smi)
sudo add-apt-repository ppa:graphics-drivers/ppa <br/>
sudo apt install nvidia-driver-470 <br/>
#sudo apt install nvidia-driver-450 <br/>
sudo apt-get update <br/>
## Pure the driver 
sudo apt-get purge nvidia* <br/>
## Reboot  
sudo reboot  <br/>
# Second, step install gcc (to prevent Failed to verify gcc version) 
sudo apt update  <br/>
sudo apt install build-essential  <br/>
sudo apt-get install manpages-dev  <br/>
sudo apt install gcc  <br/>
gcc --version 
### Visit https://developer.nvidia.com/cuda-11.0-download-archive to download suitable cuda (our case 11.0.2) 
wget https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run <br/>
sudo chmod +x cuda_11.0.2_450.51.05_linux.run <br/>
sudo sh cuda_11.0.2_450.51.05_linux.run # only check on CUDA Toolkit 11.0 it will work <br/>
#### Exoport path
sudo nano /etc/profile.d/cuda.sh <br/>
export PATH=/usr/local/cuda-11.1/bin:$PATH <br/>
export CUDADIR=/usr/local/cuda-11.1 <br/> 
sudo chmod +x /etc/profile.d/cuda.sh <br/>
sudo nano /etc/ld.so.conf.d/cuda.conf <br/>
/usr/local/cuda-11.1/lib64  <br/>
sudo ldconfig <br/>
#### Another the way to add PATH Alway work
nano ~/.bashrc  <br/>
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$  <br/>
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} <br/>
source ~/.bashrc <br/>

## Download cuDNN suitable version 
#### Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.
install cudnn https://developer.nvidia.com/rdp/cudnn-archive <br/>
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux <br/>
### Extract cuDNN
tar -xvf cudnnXX <br/>
#### Copy to Path
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include <br/>
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 <br/>
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* <br/>

# Install Python
sudo add-apt-repository ppa:deadsnakes/ppa <br/>
sudo apt-get update <br/>
sudo apt-get install python3.7 # 3.6 3.9  <br/>

# Install Virtual Environment
Cd to Home <br/>
mkdir venv_envs <br/>
cd venv_envs <br/>
sudo apt install python3-venv <br/>
python3 -m venv PT-TF24 <br/>
source ~/PT-TF24/PT-TF24/bin/activate <br/>
pip install --upgrade pip <br/>

# Install Tensorflow-gpu 
pip install tensorflow-gpu==2.4.1 # or pip install tensorflow-gpu==2.4.0 <br/>

# Install Pytorch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Install jupyter
pip install jupyter <br/>
pip install jupyterlab <br/>

# Congratsss!
