# Testing code for GENNBV and ScanRL:

python source/standalone/mad3d/sb3_inference_gennbv.py --input /home/Dataset/houes3k_env20/preprocess/test.txt --task Isaac-GENNBV-RL --checkpoint /home/hat/Documents/IsaacLab/logs/sb3/Isaac-GENNBV-RL/final/model_3072000_steps.zip --enable_camera --trans 0 0 0 --headless

# Testing code for GENNBV and GENNBV:
python source/standalone/mad3d/inference_scanrl.py --input /home/Dataset/houes3k_env20/preprocess/test.txt --task Isaac-GENNBV-RL --checkpoint ../scanrl_checkpoint/DDQN_ENV_DepthFusionBGrayMultHouseRand-v0_NB_EP_2500_BS_64_LR_0.00025_ep_1460.h5 --enable_camera --trans -4 -4 0 --headless


# Training code for GENNBV and GENNBV:
python source/standalone/mad3d/sb3_gennbv_rl.py --task Isaac-GENNBV-RL --num_envs 256 --enable_cameras
