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
## For ssh user
```
# headless (no gui)
./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task=Isaac-Cartpole-Direct-v0 --livestream 1

# webrtx (gui)
./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task=Isaac-Cartpole-Direct-v0 --livestream 2
# use this link to access gui
http://138.25.209.105:8211/streaming/webrtc-demo/?server=138.25.209.105
```

## Setup Dataset
Download raw [data](https://drive.google.com/file/d/13ojen4WYDuRf47Hg3vuEcuwso5G5Shmh/view?usp=drive_link)

Make USD data
```

```
Download preprocessed occupancy [data](https://drive.google.com/file/d/1nTxavT1OunN_ZK4X5GR_FajhkHCFlD62/view?usp=sharing)

## Try MAD3D Single Drone for ssh user
```
# headless (no gui)
./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Direct-v1 --num_envs 8 --enable_cameras --livestream 1

# webrtx (gui)
./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Direct-v1 --num_envs 8 --enable_cameras --livestream 2
# use this link to access gui
http://138.25.209.105:8211/streaming/webrtc-demo/?server=138.25.209.105
```
