#CUDA_VISIBLE_DEVICES=1 MESA_GL_VERSION_OVERRIDE=4.6 python source/standalone/mad3d/sb3_inference_neu_nbv.py --input ../Dataset/house3k/preprocess/test.txt --enable_cameras --headless --output ../Dataset/neu-nbv-house3k/ --trans 4 4 0

#CUDA_VISIBLE_DEVICES=1 MESA_GL_VERSION_OVERRIDE=4.6 python source/standalone/mad3d/sb3_inference_neu_nbv.py --input ../Dataset/house3k/preprocess/test.txt --enable_cameras --headless --output ../Dataset/neu-nbv-house3k/ --trans -4 4 0

#CUDA_VISIBLE_DEVICES=1 MESA_GL_VERSION_OVERRIDE=4.6 python source/standalone/mad3d/sb3_inference_neu_nbv.py --input ../Dataset/house3k/preprocess/test.txt --enable_cameras --headless --output ../Dataset/neu-nbv-house3k/ --trans 4 -4 0

CUDA_VISIBLE_DEVICES=1 MESA_GL_VERSION_OVERRIDE=4.6 python source/standalone/mad3d/sb3_inference_neu_nbv.py --input ../Dataset/house3k/preprocess/test.txt --enable_cameras --headless --output ../Dataset/neu-nbv-house3k/ --trans -4 -4 0

#CUDA_VISIBLE_DEVICES=1 MESA_GL_VERSION_OVERRIDE=4.6 python source/standalone/mad3d/sb3_inference_neu_nbv.py --input ../Dataset/house3k/preprocess/test.txt --enable_cameras --headless --output ../Dataset/neu-nbv-house3k/ --trans 0 0 0
