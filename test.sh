

./isaaclab.sh -p source/standalone/mad3d/sb3_inference.py --input /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/test.txt --task=MAD3D-v0 --checkpoint logs/sb3/MAD3D-v0/2025-01-18_06-47-13/model_3584000_steps.zip --trans 4 -4 0  --enable_camera

./isaaclab.sh -p source/standalone/mad3d/sb3_inference.py --input /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/test.txt --task=MAD3D-v0 --checkpoint logs/sb3/MAD3D-v0/2025-01-18_06-47-13/model_3584000_steps.zip --trans -4 4 0  --enable_camera

./isaaclab.sh -p source/standalone/mad3d/sb3_inference.py --input /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/test.txt --task=MAD3D-v0 --checkpoint logs/sb3/MAD3D-v0/2025-01-18_06-47-13/model_3584000_steps.zip --trans -4 -4 0  --enable_camera
