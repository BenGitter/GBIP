# Check memory and GPU
df -m       # allocated disc size
free -m     # allocated memory
nvidia-smi  # GPU info

# Download GBIP/KSE
cd /workspace
git clone https://github.com/BenGitter/GBIP.git
git clone -b full_yolo https://github.com/BenGitter/GBIP.git
git clone -b YOLO+COCO https://github.com/BenGitter/KSE.git
cd GBIP


# Test down/upload speed
sudo apt-get install speedtest-cli
speedtest-cli --secure

# Install unzip, needed for COCO download
sudo apt-get install unzip gcc

# Download MS COCO
bash ./get_coco.sh

# Create environment file
conda env export > environment.yml
# Create conda environment from file
conda env create -f data/environment.yaml
sudo apt-get install gcc
pip install pycocotools
pip install opencv-python-headless # or: opencv-python
apt-get install libgl1 # sometimes needed


# enable scrolling in tmux
echo "set -g mouse on" >> ~/.tmux.conf
tmux source-file ~/.tmux.conf

# Copy something from another system to this system
scp username@hostname:/path/to/remote/file /path/to/local/file

# Clone conda base env
conda init bash
conda create --name GBIP --clone base
conda activate GBIP

# List number of files in dir
ls | wc -l # pipe output of ls into word count (-l : count lines)


# Required packages
opencv
seaborn
pycocotools
yaml
tqdm
pathlib
pandas?
json
torchvision


60 GB works
