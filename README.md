# Donwload IsaacSim 4 (Omniverse Launcher) if you do not have it
[Download link](https://www.nvidia.com/en-us/omniverse/download/) 

[Installation guideline](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)

You can open Omniverse Launcher and IsaacSim by clicking omniverse-launcher-linux.AppImage

# Donwload and setup MAD3D IsaacLab 
The master branch of MAD3D IsaacLab should follow the official IsaacLab
## Clone mad3d IsaacLab
[Official installation guideline](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html)
```
# git clone MAD3D IsaacLab
git clone https://github.com/johnnylu305/IsaacLab.git
# enter the cloned repository
cd IsaacLab
# create a symbolic link
# example: ln -s /home/nvidia/.local/share/ov/pkg/isaac-sim-4.0.0 _isaac_sim
ln -s path_to_isaac_sim _isaac_sim
# install
./isaaclab.sh --install
```
## Test IsaacLab
```
# cartpole sb3 example
./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task=Isaac-Cartpole-Direct-v0
```
