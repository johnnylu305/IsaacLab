import argparse
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("--input", type=str, required=True, help="Path to the txt list.")
parser.add_argument("--vis", action="store_true", help="If set, visualize occupancy grid.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output", type=str, default='', help="Path to save the output metrics.")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import omni
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
import sys
import csv
import time
import open3d as o3d
from PIL import Image
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.lab.sensors import CameraCfg, Camera
from omni.isaac.lab.utils.math import transform_points, unproject_depth
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from omni.isaac.lab.utils.math import convert_camera_frame_orientation_convention, euler_xyz_from_quat
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth, convert_orientation_convention
from collections import deque
#from stable_baselines3.common.vec_env import VecNormalize
sys.path.append("/home/dsr/Documents/mad3d/isaac-sim-4.2.0/home/IsaacLab")
#from sb3_ppo_cus import PPO_Cus
import sys
sys.path.append('/projects/MAD3D/Zhuoli/IsaacLab/source/standalone/mad3d/neural_rendering')
from neu_nbv_utils import util
from neural_rendering.evaluation.pretrained_model import PretrainedModel
from dotmap import DotMap
import yaml
from scipy.spatial.transform import Rotation as R
import copy

# Env
NUM_STEPS = 15
NUM_ENVS = 1
GRID_SIZE = 20
ENV_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_OFFSET = [0.1, 0, 0]
# camera initial position
INITIAL_POSE = [10, 10, 2]
# Sensor"
CAMERA_HEIGHT = 1000
CAMERA_WIDTH = 1000
TARGET_HEIGHT = 300
TARGET_WIDTH = 300

z_near = 0.01
z_far = 60.0
batch_size = 49 # batch size of the candidate view
budget = 1

class OccupancyGrid:
    def __init__(self, env_size, grid_size, device="cpu"):
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.grid_size = grid_size
        self.resolution = env_size / grid_size[1]
        self.device = device

    def update_log_odds(self, i, indices, occupied=True):
        indices = indices.long()
        if occupied:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] += 0.84
        else:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] -= 0.4
        self.grid.clamp_(min=-3.5, max=3.5)

    def log_odds_to_prob(self):
        return 1 / (1 + torch.exp(-self.grid))


def setup_scene(world, scene_path, index, scene_prim_root="/World/Scene"):
    # floor
    world.scene.add_ground_plane(size=40.0, color=torch.tensor([52.0 / 255.0, 195.0 / 255.0, 235.0 / 255.0]))
    # light
    UsdLux.DomeLight.Define(world.scene.stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(500)
    
    # add usd into the stage
    scene = add_reference_to_stage(usd_path=scene_path, prim_path=scene_prim_root)

    # define the property of the stage
    scene_prim = XFormPrim(prim_path=scene_prim_root, translation=[0, 0, 0])

    # activate the stage
    world.scene.add(scene_prim)

    cameraCfg = CameraCfg(
        prim_path=f"/World/Camera_0",
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), convention="world"),
        update_period=0,  # Update every physical step
        data_types=["distance_to_image_plane","rgb"],
        spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.8, # in cm default 24, dji 1.38
                #focus_distance=1.0, # in m 
                horizontal_aperture=24., # in mm 
                clipping_range=(z_near, z_far) # near and far plane in meter
            ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT
    )
 
    scene_entities = {}
    
    scene_entities[f"camera_0"] = Camera(cameraCfg)

    return scene_entities


def create_blocks_from_occupancy(env_id, env_origin, occupancy_grid, cell_size, base_height, z,env_size, target=0, h_off=60):
    stage = omni.usd.get_context().get_stage()
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == target:
                # Calculate position based on cell coordinates
                cube_pos = Gf.Vec3f((x*cell_size)+env_origin[0]-env_size/2, (y*cell_size)+env_origin[1]-env_size/2, base_height+h_off)

                # Define the cube's USD path
                cube_prim_path = f"/World/OccupancyBlocks/Block_{env_id}_{x}_{y}_{z}_{target}"

                # Create a cube primitive or get the existing one
                cube_prim = UsdGeom.Cube.Define(stage, Sdf.Path(cube_prim_path))
                cube_prim.GetPrim().GetAttribute("size").Set(cell_size)

                # Manage the transformation
                xform = UsdGeom.Xformable(cube_prim.GetPrim())
                xform_ops = xform.GetOrderedXformOps()
                if not xform_ops:
                    # If no transform ops exist, add a new translate op
                    xform_op = xform.AddTranslateOp()
                    xform_op.Set(cube_pos)
                else:
                    # If transform ops exist, modify the existing translate op
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(cube_pos)
                            break
                    else:
                        # If no translate op found, add it
                        xform.AddTranslateOp().Set(cube_pos)

def proximity_metric(acc_points, pcd_gt, thres=0.01):
    # Compute pairwise distances between acc_points and pcd_gt
    distances = torch.cdist(acc_points, pcd_gt, p=2)
    
    # Find the minimum distance for each point in acc_points
    min_distances, _ = torch.min(distances, dim=1)
    
    # Count the number of points within distance thres
    close_points = (min_distances <= thres).sum().item()
    
    # Compute the ratio
    ratio = close_points / acc_points.shape[0]
    return ratio

# Compute Chamfer Distance
def chamfer_distance(pcd1, pcd2):
    # Compute pairwise distances
    dist1 = torch.cdist(pcd1, pcd2, p=2)  # Pairwise Euclidean distances

    # For each point in pcd1, find the nearest point in pcd2
    min_dist1, _ = torch.min(dist1, dim=1)

    # For each point in pcd2, find the nearest point in pcd1
    min_dist2, _ = torch.min(dist1, dim=0)

    # Average the nearest neighbor distances
    cd = torch.mean(min_dist1) + torch.mean(min_dist2)
    return cd

def subsample_point_cloud(pcd, num_samples):
    if pcd.shape[0] <= num_samples:
        return pcd  # If already smaller, return as is
    indices = torch.randperm(pcd.shape[0])[:num_samples]  # Randomly sample indices
    return pcd[indices]

def farthest_point_sampling(pcd, num_samples):
    n, _ = pcd.shape
    if n <= num_samples:
        return pcd  # If already smaller, return as is

    # Initialize an empty set of sampled points
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=pcd.device)
    distances = torch.full((n,), float('inf'), device=pcd.device)  # Initialize distances to infinity

    # Start with a random point
    sampled_indices[0] = torch.randint(0, n, (1,), device=pcd.device)
    for i in range(1, num_samples):
        # Compute distances to the latest sampled point
        dist_to_new_point = torch.sum((pcd - pcd[sampled_indices[i - 1]]) ** 2, dim=1)
        # Update the minimum distances to the set of sampled points
        distances = torch.min(distances, dist_to_new_point)
        # Choose the point farthest away from the current set
        sampled_indices[i] = torch.argmax(distances)

    return pcd[sampled_indices]

def pos_quat_to_transformation_matrix(pos, quat):
    """
    Convert position and quaternion to a transformation matrix.

    Args:
        pos (np.ndarray): Position vector (x, y, z).
        quat (np.ndarray): Quaternion (x, y, z, w).

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = pos
    rotation_matrix = R.from_quat(quat).as_matrix()
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix


def get_nbv_ref_index(
    model, images, poses, focal, c, z_near, z_far, candidate_list, budget, ref_index
):
    _, _, H, W = images.shape

    for i in range(budget):
        remain_candidate_list = list(set(candidate_list) - set(ref_index))
        reward_list = []

        model.network.encode(
            images[ref_index].unsqueeze(0),
            poses[ref_index].unsqueeze(0),
            focal.unsqueeze(0),
            c.unsqueeze(0),
        )

        for target_view in remain_candidate_list:
            novel_pose = poses[target_view]
            target_rays = util.gen_rays(
                novel_pose.unsqueeze(0), W, H, focal, z_near, z_far, c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            predict = DotMap(model.renderer_par(target_rays))
            uncertainty = predict["uncertainty"][0]
            #test_visualize(predict)
            reward = torch.sum(uncertainty**2).cpu().numpy()
            reward_list.append(reward)

        nbv_index = np.argmax(reward_list)
        new_ref_index = remain_candidate_list[nbv_index]
        ref_index.append(new_ref_index)

    return ref_index

def run_neu_nbv_model(images, poses, neu_nbv_model, focal, c, z_near, z_far, candidate_list, bugent, initial_ref_index):
    ref_index = get_nbv_ref_index(
                                neu_nbv_model,
                                images,
                                poses.cuda(),
                                focal,
                                c,
                                z_near,
                                z_far,
                                candidate_list,
                                budget,
                                copy.deepcopy(initial_ref_index),
                            )
    new_pose = poses[ref_index[-1]]
    position, quaternion = pose_to_position_quaternion(torch.tensor(new_pose))
    return position, quaternion, ref_index

def random_view(current_xyz, radius, phi_min, min_view_change, max_view_change):
    """
    random scatter view direction changes by given current position and view change range.
    """

    u = current_xyz / np.linalg.norm(current_xyz)

    # pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(u), np.eye(len(u)))

    # form a vector perpendicular to u:
    uperp = r - r.dot(u) * u
    uperp = uperp / np.linalg.norm(uperp)

    # random view angle change in radian
    random_view_change = np.random.uniform(low=min_view_change, high=max_view_change)
    cosine = np.cos(random_view_change)
    w = cosine * u + np.sqrt(1 - cosine**2 + 1e-8) * uperp
    w = radius * w / np.linalg.norm(w)

    view = xyz_to_view(w, radius)

    if view[0] < phi_min:
        view[0] = phi_min

    return view

def xyz_to_view(xyz, radius):
    """
    Calculate spherical coordinates (theta, phi) from a point in Cartesian space.
    theta: angle in xy-plane (azimuth), phi: elevation angle
    """
    x, y, z = xyz
    r = radius
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r)
    return np.array([phi, theta, r])

def uniform_sampling(radius, phi_min):
    """
    Uniformly generate a unit vector on the hemisphere.
    Calculate corresponding view direction targeting the coordinate origin.
    """
    xyz = np.array([0.0, 0.0, 0.0])
    while np.linalg.norm(xyz) < 0.001:  # Avoid numerical errors
        xyz[0] = np.random.uniform(low=-1.0, high=1.0)
        xyz[1] = np.random.uniform(low=-1.0, high=1.0)
        xyz[2] = np.random.uniform(low=0.0, high=1.0)

    xyz = radius * xyz / np.linalg.norm(xyz)
    view = xyz_to_view(xyz, radius)

    # Clamp phi to a minimum threshold
    if view[0] < phi_min:
        view[0] = phi_min

    return view

def view_to_pose_batch(views, radius):
    num = len(views)
    phi = views[:, 0]
    theta = views[:, 1]

    # phi should be within [min_phi, 0.5*np.pi)
    index = phi >= 0.5 * np.pi
    phi[index] = np.pi - phi[index]

    poses = np.broadcast_to(np.identity(4), (num, 4, 4)).copy()

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translations = np.stack((x, y, z), axis=-1)

    angles = np.stack((theta, -phi, np.pi * np.ones(num)), axis=-1)
    rotations = R.from_euler("ZYZ", angles).as_matrix()

    poses[:, :3, -1] = translations
    poses[:, :3, :3] = rotations

    return poses

def pose_to_position_quaternion(pose_matrix):
    """
    Converts a 4x4 transformation matrix to position and quaternion.
    Args:
        pose_matrix: torch.Tensor, shape (4, 4)
    Returns:
        position: torch.Tensor, shape (3,)
        quaternion: torch.Tensor, shape (4,) in [qx, qy, qz, qw] format
    """
    # Extract position (last column of the pose matrix)
    position = pose_matrix[:3, 3]  # Shape (3,)

    # Extract rotation matrix (top-left 3x3 portion of the pose matrix)
    rotation_matrix = pose_matrix[:3, :3].cpu().numpy()

    # Convert rotation matrix to quaternion using scipy
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Quaternion in [qx, qy, qz, qw] format

    # Convert to torch tensors for compatibility
    position = position.clone()  # Ensure no unwanted references
    quaternion = torch.tensor(quaternion)  # Shape (4,)
    quaternion_wxyz = torch.tensor([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    return position, quaternion_wxyz


def run_simulator(sim, scene_entities, agent, hollow_occ_path, gt_pcd_path, coverage_ratio_rec, cd_rec, acc_rec, scene_id, end, dataset_name):
    # env 
    grid_size = GRID_SIZE
    env_size = ENV_SIZE
    # sim time step
    sim_dt = sim.get_physics_dt()
    # camera
    camera = scene_entities[f"camera_0"]   
     
    # initial pose
    target_position = torch.tensor([INITIAL_POSE], dtype=torch.float32)
    # look at (0, 0, 2)
    camera.set_world_poses_from_view(target_position, torch.tensor([0, 0, 0], dtype=torch.float32).unsqueeze(0))  
    #camera.set_world_poses_from_view(torch.tensor([20,0,0]).unsqueeze(0).float(), torch.tensor([0,0,10]).unsqueeze(0).float())
    #camera.set_world_poses_from_view(torch.tensor([15,0,0]).unsqueeze(0).float(), torch.tensor([0,0,10]).unsqueeze(0).float())
    # obv grid
    grid = OccupancyGrid(env_size, (1, grid_size, grid_size, grid_size))
    # obv face
    obv_face = torch.zeros(NUM_ENVS, GRID_SIZE, GRID_SIZE, GRID_SIZE, 6, device=DEVICE) 

    # ground truth
    occ_gt = torch.tensor(np.load(hollow_occ_path))

    # ground truth pcd
    # TODO: update
    pcd_gt = torch.rand((10000, 3), dtype=torch.float32)  # Replace `m` with the number of points

    # save path
    save_img_folder = os.path.join(os.path.dirname(args_cli.output), f"{dataset_name}_data")
    folder_name = os.path.basename(os.path.dirname(hollow_occ_path))
    os.makedirs(os.path.join(save_img_folder, folder_name), exist_ok=True)

    all_points = []
    all_colors = []
    
    for i in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    image0 = camera.data.output["rgb"][0].transpose(0,2).clone()/255.
    #camera.set_world_poses_from_view(torch.tensor([20,0,0]).unsqueeze(0).float(), torch.tensor([0,0,10]).unsqueeze(0).float())
    camera.set_world_poses_from_view(target_position, torch.tensor([0, 0, 2]).unsqueeze(0).float())
    #camera.set_world_poses_from_view(torch.tensor([20,0,0]).unsqueeze(0).float(), torch.tensor([0,0,10]).unsqueeze(0).float())
    for i in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    image1 = camera.data.output["rgb"][0].transpose(0,2).clone()/255.

    images = torch.zeros((3,3,CAMERA_WIDTH, CAMERA_HEIGHT))
    #images = F.interpolate(images, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
    images[0] = image0
    images[1] = image1
    image_queue = deque()
    image_queue.append(images[0])
    image_queue.append(images[1])
    #poses = sample_hemisphere(20, 2000)
    phi_min = 1.0
    #poses = uniform_sampling(env_size, phi_min)
    #view=[]
    
    angle_radius = env_size
    camera_pos = camera.data.pos_w.clone()
    camera_quat = camera.data.quat_w_ros.clone()

    new_pose = pos_quat_to_transformation_matrix(camera_pos, camera_quat)
    #view = []
    #for i in range(50):
    #    local = random_view(camera_pos[0].cpu().numpy(), env_size, 0.15, 0.2, 1.05)
    #    view.append(local)
    #poses = view_to_pose_batch(np.array(view), env_size)
    #poses = torch.tensor(poses).cuda().float()
    # start scanning
    for index in range(NUM_STEPS):
        print("")
        # simulate
        for _ in range(4):
            sim.step()
            camera.update(dt=sim_dt)
        
        view = []
        for i in range(50):
            local = random_view(camera_pos[0].cpu().numpy(), env_size, 0.15, 0.2, 1.05)
            view.append(local)
        poses = view_to_pose_batch(np.array(view), env_size)
        poses = torch.tensor(poses).cuda().float()
        # depth image
        depth_image = torch.clamp(camera.data.output["distance_to_image_plane"], 0, env_size * 4)
        # depth image to local point cloud
        points_3d_cam = unproject_depth(depth_image, camera.data.intrinsic_matrices)
        # local point cloud to global point cloud
        points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)
        # filter out out of boundary points
        mask_x = (points_3d_world[:, :, 0]).abs() < env_size / 2 - 1e-3
        mask_y = (points_3d_world[:, :, 1]).abs() < env_size / 2 - 1e-3
        mask_z = (points_3d_world[:, :, 2] < env_size - 1e-3) & (points_3d_world[:, :, 2] >= 0)
        mask = mask_x & mask_y & mask_z    
        
        # update grid
        offset = torch.tensor([env_size / 2, env_size / 2, 0]).to(points_3d_world.device)
        if points_3d_world[mask].shape[0] > 0:
            ratio = grid_size / env_size
            grid.update_log_odds(0, torch.floor((points_3d_world[mask] + offset) * ratio), occupied=True)
            obv_face = torch.logical_or(obv_face,
                                        get_seen_face(torch.unique(torch.floor((points_3d_world[mask]+offset)*ratio).int(), dim=0),
                                        torch.floor((camera.data.pos_w+offset)*ratio), GRID_SIZE, DEVICE))
        # probability grid
        probability_grid = grid.log_odds_to_prob()

        # coverage ratio
        hard_occ = torch.where(probability_grid[0, :, :, :] >= 0.6, 1, 0)
        hard_occ = hard_occ[:, :, 1:]
        num_match_occ = torch.sum(torch.logical_and(hard_occ, occ_gt[:, :, 1:]), dim=(0, 1, 2))
        total_occ = torch.sum(occ_gt[:, :, 1:], dim=(0, 1, 2))
        coverage_ratio = (num_match_occ/total_occ).reshape(-1, 1)

        # record coverage ratio
        coverage_ratio_rec[scene_id, index] = coverage_ratio.cpu().numpy()[0][0]
        print("Cov:", coverage_ratio[0, 0].item())

        # get rgb image
        #rgb_image = camera.data.output["rgb"].clone()     
        
        rgb_image = camera.data.output["rgb"][0].transpose(0,2).clone()/255.
        #rgb_image = F.interpolate(rgb_image.unsqueeze(0), size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
        image_queue.append(rgb_image)
        image_queue.popleft()
        images = np.array(image_queue)
        
        rgb_image = camera.data.output["rgb"].clone()
        # pcd
        valid_points = points_3d_world[mask].reshape(-1, 3).cpu().numpy()
        valid_colors = torch.transpose(rgb_image[0], 0, 1)[mask.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)].reshape(-1, 3).cpu().numpy()
        # Append valid points and colors to lists
        all_points.append(valid_points)
        all_colors.append(valid_colors)
        acc_points = np.vstack(all_points)
        acc_points = subsample_point_cloud(torch.tensor(acc_points), 100000)
        cd = chamfer_distance(acc_points, pcd_gt).item()
        cd_rec[scene_id, index] = cd
        print("CD:", cd)
        accuracy = proximity_metric(acc_points, pcd_gt, 3.)
        acc_rec[scene_id, index] = accuracy
        print("Acc:", accuracy)

        # get observation 
        obs = _get_observations(probability_grid, rgb_image, torch.tensor(new_pose).cuda(), obv_face)
        # torch tensor to numpy
        for key, value in obs.items():
            obs[key] = value.cpu().numpy()
        
        intrinsic_matrix = camera.data.intrinsic_matrices.clone()[0]
        # Extract focal lengths (fx, fy)
        focal = torch.tensor([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]]).cuda()

        # Extract camera center (cx, cy)
        c = torch.tensor([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]]).cuda()

        # Candidate view list and initial reference index
        candidate_list = list(range(batch_size))
        initial_ref_index = [0,1]  # Start with view 0 as the reference

        # Budget for iterations
        budget = 1
        images = torch.tensor(images).cuda().float()
        input_images = F.interpolate(images, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
        scale_factor = TARGET_HEIGHT/CAMERA_HEIGHT
        new_position, new_quaternion, ref_index = run_neu_nbv_model(input_images, poses, agent, focal*scale_factor, c*scale_factor, z_near, z_far, candidate_list, agent, initial_ref_index)
        new_pose = pos_quat_to_transformation_matrix(new_position.cpu(), new_quaternion.cpu())
        print(ref_index)
        # save data
        x, y, z = new_position
        #_yaw, _pitch = pose[0, 3:]
        coverage_ratio = coverage_ratio.cpu().numpy()[0][0]
        suffix = f"_{index}_{x:.2f}_{y:.2f}_{z:.2f}_{coverage_ratio:.2f}.png"
        plt.imsave(os.path.join(save_img_folder, folder_name, 'depth'+suffix),
                   np.clip(depth_image[0,:,:,0].detach().cpu().numpy(), 0, ENV_SIZE*2).astype(np.uint8),
                           cmap='gray',
                           vmin=0,
                           vmax=ENV_SIZE)
        plt.imsave(os.path.join(save_img_folder, folder_name, 'rgb'+suffix), 
                   rgb_image[0].detach().cpu().numpy().astype(np.uint8))

        # save occ grid
        np.save(os.path.join(save_img_folder, folder_name, f'prob_occ_{index}.npy'), probability_grid[0, :, :, :].cpu().numpy())

        # save pcd
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(valid_points)
        point_cloud.colors = o3d.utility.Vector3dVector(valid_colors/255.)  # Normalize colors to [0, 1]
        # Save to a PLY file
        o3d.io.write_point_cloud(os.path.join(save_img_folder, folder_name, f'pcd_{index}.ply'), point_cloud)

        # this may slow down the process
        org_x, org_y, org_z = env_size/2., env_size/2., 0
        cell_size = env_size/grid_size  # meters per cell
        slice_height = env_size / grid_size  # height of each slice in meters
        if args_cli.vis:
            for i in range(grid_size):
                occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
                create_blocks_from_occupancy(0, [0, 0, 0], occupancy_grid.cpu().numpy(), cell_size, i * cell_size, i, env_size, 1, 30)
        
        # apply action
        #camera.set_world_poses(new_positions, orientation_camera)
        camera.set_world_poses(new_position.unsqueeze(0), convert_orientation_convention(new_quaternion.unsqueeze(0).float(), origin="world", target="ros"))
        #for _ in range(4):
        #    sim.step()
        #    camera.update(dt=sim_dt)
        #view = []
        #for i in range(50):
        #    local = random_view(camera_pos[0].cpu().numpy(), env_size, 0.15, 0.2, 1.05)
        #    view.append(local)
        #poses = view_to_pose_batch(np.array(view), env_size)
        #poses = torch.tensor(poses).float()

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Calculate the average values up to the current scene_id
    # Plot the averaged values
    coverage_values = coverage_ratio_rec[scene_id]
    cd_values =cd_rec[scene_id]
    accuracy_values = acc_rec[scene_id]
    plots = [
        {"values": coverage_values, "label": "Coverage Ratio", "color": "b", "marker": "o", "filename": f'coverage_ratio_plot_scene_{scene_id + 1}.png'},
        {"values": cd_values, "label": "Chamfer Distance", "color": "g", "marker": "s", "filename": f'chamfer_distance_plot_scene_{scene_id + 1}.png'},
        {"values": accuracy_values, "label": "Accuracy", "color": "r", "marker": "^", "filename": f'accuracy_plot_scene_{scene_id + 1}.png'}
    ]

    # Save each plot as a separate figure
    for plot in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(plot["values"], marker=plot["marker"], color=plot["color"], label=plot["label"])
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{plot["label"]} Over Steps')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(save_img_folder, folder_name, plot["filename"])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # High resolution
        plt.close()

    # Save pcd
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors / 255.0)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(os.path.join(save_img_folder, folder_name, "pcd_final.ply"), point_cloud)

    # Open the CSV file in append mode
    csv_file = "coverage_ratios.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + coverage_values.tolist())

    # Open the CSV file in append mode
    csv_file = "cd.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + cd_values.tolist())

    # Open the CSV file in append mode
    csv_file = "accuracy.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + accuracy_values.tolist())

    if scene_id+1 == end:
        # Calculate the average values up to the current scene_id
        average_values = np.mean(coverage_ratio_rec, axis=0)
        print("Average Values:", average_values)
        # Plot the averaged values
        plt.figure(figsize=(10, 6))  # Optional: Specify figure size for better resolution
        plt.plot(average_values, marker='o', color='b', label='Average Curve')
        plt.xlabel('Step')
        plt.ylabel('Average Value')
        plt.title('Average Line Curve of 20 Steps')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file instead of showing it
        plot_filename = f'average_curve_plot_scene.png'  # Dynamic filename based on scene_id
        plt.savefig(os.path.join(save_img_folder, plot_filename), dpi=300, bbox_inches='tight')  # dpi=300 for high-resolution
        plt.close()
        

def get_seen_face(occ_grid_xyz, camera_xyz, grid_size, device):
    rays = camera_xyz - occ_grid_xyz

    # Normalize rays
    rays_norm = rays.to(device) / (torch.norm(rays, dim=-1, keepdim=True)+1e-10).to(device)

    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).to(device)
    face_grid = torch.zeros((grid_size, grid_size, grid_size, 6), dtype=torch.bool).to(device)

    # Check visibility for each face
    for i, face in enumerate(faces):
        dot_product = torch.sum(rays_norm * face, dim=-1)
        face_grid[occ_grid_xyz[:, 0], occ_grid_xyz[:, 1], occ_grid_xyz[:, 2], i] = dot_product > 0

    return face_grid

def _get_observations(probability_grid, rgb_image, pose, obv_face):
        
        # pose
        current_pose = pose
        # norm
        current_pose[:3] /= ENV_SIZE
        current_pose /= 3.15
        h_limit = (torch.ones(NUM_ENVS).to(DEVICE)*10*GRID_SIZE/ENV_SIZE).int()
        h_limit_expanded = h_limit.unsqueeze(1) / GRID_SIZE
        pose_step = torch.cat([current_pose.reshape(NUM_ENVS, -1), h_limit_expanded], dim=1)


        obv_occ = torch.ones(NUM_ENVS, GRID_SIZE, GRID_SIZE, GRID_SIZE, 4, device=DEVICE)*0.5
        # generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        y_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        z_coords = torch.linspace(0.0, ENV_SIZE, GRID_SIZE, device=DEVICE)
        
        # create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        # xyz
        obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1) / ENV_SIZE
        # occ grid
        obv_occ[:, :, :, :, 0] = probability_grid.clone()
        # face
        occ_face = torch.cat((obv_occ, obv_face), dim=-1)
        # image
        # Normalize to [0, 1] and resize
        resized_rgb_image = F.interpolate(rgb_image.permute(0, 3, 1, 2)/255.0, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
        resized_rgb_image = resized_rgb_image.permute(0, 2, 3, 1)
        #rgb_image = rgb_image/ 255.
        # img: N, H, W, 1
        gray_scale_img = torch.mean(resized_rgb_image[0, :, :, :], (2))

        center = torch.zeros((NUM_ENVS, 3), device=DEVICE)

        obs = {"pose_step": pose_step,
               "img": gray_scale_img.reshape(-1, 1, TARGET_HEIGHT, TARGET_WIDTH),
               "occ": occ_face.permute(0, 4, 1, 2, 3),
               "env_size": torch.ones((NUM_ENVS, 1)) * ENV_SIZE,
               "aux_center": center
               }

        return obs

def main():
    # Define the input file
    input_file = args_cli.input
    # load data path    
    scene_paths = []
    with open(input_file, "r") as file:
        for line in file:
        #for line in self.files:
            # Strip newline and split by space
            line = line.strip()
            if line:  # Ignore empty lines
                *path_parts, _ = line.rsplit(maxsplit=1)  # Split by spaces, last part is the value
                hollow_occ_path = " ".join(path_parts)  # Recombine the path parts with spaces

                # Check if the path is a hollow_occ.npy
                if hollow_occ_path.endswith("hollow_occ.npy"):
                    directory = os.path.dirname(hollow_occ_path)  # Extract directory from the path

                    # Search for the corresponding `.usd` file
                    usd_files = glob.glob(os.path.join(directory, "*.usd"))
                    usd_files = [file for file in usd_files if "non_metric" not in file]

                    # Search for `faces.npy` and `fill_occ_set.pkl` in the same directory
                    faces_path = os.path.join(directory, "faces.npy")
                    fill_occ_set_path = os.path.join(directory, "fill_occ_set.pkl")

                    # Ensure the required files exist
                    if usd_files and os.path.exists(faces_path) and os.path.exists(fill_occ_set_path):
                        #scene_paths.append((usd_files[0], hollow_occ_path, fill_occ_set_path, faces_path, UsdFileCfg(usd_path=usd_files[0])))
                        scene_paths.append((usd_files[0], hollow_occ_path, fill_occ_set_path, faces_path))
    
    # coverage ratio record
    coverage_ratio_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)
    # chamfer distance record
    cd_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)
    # accuracy record
    acc_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)

    # world initialization
    world = World(stage_units_in_meters=1.0, backend="torch", device="cpu")
    # user viewport
    set_camera_view(eye=np.array([-40, -40, 60]), target=np.array([15, -15, 8]))
    # create agent
    #agent = make_env()
    
    log_path = os.path.join("source","standalone","mad3d","neural_rendering", "logs", 'dtu_training')
    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)
    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)
    # set agent to neu_nbv model
    agent = PretrainedModel(cfg["model"], ckpt_file, 'cuda', [0])

    
    # start scan the shapes
    for i, (scene_path, hollow_occ_path, fill_occ_set_path, faces_path) in enumerate(scene_paths):
        print(f"{i}: {scene_path}")
        
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(args_cli.input)))
    
        # setup ground, light, and camera
        scene_entities = setup_scene(world, scene_path, i)

        # output dir
        output_dir = os.path.dirname(scene_path)
        
        world.reset()
        
        # TODO: update this
        gt_pcd_path = None
        run_simulator(world, scene_entities, agent, hollow_occ_path, gt_pcd_path, coverage_ratio_rec, cd_rec, acc_rec, i, len(scene_paths), dataset_name)

        '''
        # remove camera
        prim_path = f"/World/Camera_{i}"
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path):
            stage.RemovePrim(Sdf.Path(prim_path))
        scene_entities[f"camera_{i}"].__del__()
        
        del scene_entities[f"camera_{i}"]       
        
        # remove camera
        scene_entities[f"camera_{i}"].__del__()
        scene_entities[f"camera_{i}"].reset()
        prim_path = f"/World/Camera_{i}"
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path):
            stage.RemovePrim(Sdf.Path(prim_path))
        del scene_entities[f"camera_{i}"]       
        '''
        world.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
