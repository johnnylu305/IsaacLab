import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

r = np.array(
    [[0.9703, 0.0147, 0.2416], [-0.0075, 0.9995, -0.0310], [-0.2419, 0.0282, 0.9699]]
)

rm = R.from_matrix(r)
print(rm.as_quat())
# a = np.load("cameras.npz")
# print("scale_mat:", a["scale_mat_1"])
# S = a["scale_mat_2"]

# P = a["world_mat_2"]
# print("original P:", P)

# P = P[:3]
# K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
# # print(K)
# # K = K / K[2, 2]
# # K = np.insert(K, 3, 0, axis=1)

# # pose = np.eye(4, dtype=np.float32)
# # pose[:3, :3] = R  # .transpose()
# # pose[:3, 3] = (t[:3] / t[3])[:, 0]

# # P_rec = K @ pose

# # print("reconstruct P:", P_rec)
# r0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# t0 = np.array([300, 0, 0])

# P0 = np.eye(4)
# P0[:3, :3] = K @ r0
# P0[:3, 3] = K @ r0 @ t0

# np.savez("../scan129/cameras.npz", world_mat_0=P0, scale_mat_0=S)
