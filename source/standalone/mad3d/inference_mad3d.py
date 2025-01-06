import argparse
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("--input", type=str, required=True, help="Path to input USD files.")
parser.add_argument("--vis", action="store_true", help="If set, visualize occupancy grid.")


parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
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
import omni
import omni.isaac.lab.sim as sim_utils
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.lab.sensors import CameraCfg, Camera
from omni.isaac.lab.utils.math import transform_points, unproject_depth
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from PIL import Image
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.lab.utils.math import convert_camera_frame_orientation_convention, euler_xyz_from_quat

#from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import sys
sys.path.append("/home/hat/Documents/IsaacLab")
from sb3_ppo_cus import PPO_Cus
import matplotlib.pyplot as plt

# Env
num_steps = 20
num_envs = 1
GRID_SIZE = 20
ENV_SIZE = 20
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300
camera_offset = [0.1, 0.0, 0.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sensor
CameraCfg = CameraCfg(
    prim_path="/World/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), convention="world"),
    update_period=0,  # Update every physical step
    data_types=["distance_to_image_plane","rgb"],
    spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.8, # in cm default 24, dji 1.38
            #focus_distance=1.0, # in m 
            horizontal_aperture=24., # in mm 
            clipping_range=(0.01, 60.0) # near and far plane in meter
        ),
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
)


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


def setup_scene(world, scene_path, scene_prim_root="/World/Scene"):
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
    
    scene_entities = {}
    scene_entities["camera"] = Camera(CameraCfg)

    return scene_entities


def save_occupancy_grid_as_image(occupancy_grid, filename):
    colors = {1: (0, 0, 0), 0: (255, 255, 255)}
    image_data = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        image_data[occupancy_grid == value] = color
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(image_data, "RGB").save(filename)
    print(f"Saved occupancy map as image: {filename}")


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


def run_simulator(sim, scene_entities, output, agent, hollow_occ_path, coverage_ratio_avg, scene_id):
    # env 
    grid_size = GRID_SIZE
    env_size = ENV_SIZE
    # grid
    grid = OccupancyGrid(env_size, (1, grid_size, grid_size, grid_size))
    # sim time step
    sim_dt = sim.get_physics_dt()
    # camera
    camera = scene_entities["camera"]   
    # circular path
    index = 0
    theta = np.linspace(0, 6 * np.pi, 200)
    #x = env_size  
    #y = env_size
    x = 0
    y = 0
    z = 2
    
    obv_face = torch.zeros(num_envs, GRID_SIZE, GRID_SIZE, GRID_SIZE, 6, device=DEVICE)
    #pose: N, T, 5 (xyz + pitch yaw)
    #pose = torch.ones(1, 5).to(DEVICE)
    
    # initial pose
    target_position = torch.tensor([[x, y, z]], dtype=torch.float32)
    # look at (0, 0, 2)
    camera.set_world_poses_from_view(target_position, torch.tensor([0, 0, 2], dtype=torch.float32).unsqueeze(0))
    pose = torch.tensor([[x, y, z, 0, 0]]).to(DEVICE)

    occ_gt = torch.tensor(np.load(hollow_occ_path))
    save_img_folder = os.path.join(os.path.dirname(args_cli.checkpoint), "images")
    folder_name = os.path.basename(os.path.dirname(hollow_occ_path))
    
    os.makedirs(os.path.join(save_img_folder, folder_name), exist_ok=True)
    
    # start scanning
    while index < num_steps-1:
        # set camera to the target position
        #target_position = torch.tensor([[x[index], y[index], z[index]]], dtype=torch.float32)
        # look at (0, 0, 2)
        #camera.set_world_poses_from_view(target_position, torch.tensor([0, 0, 2], dtype=torch.float32).unsqueeze(0))
        # simulate
        for _ in range(10):
            sim.step()
            camera.update(dt=sim_dt)
        # next step
        index += 1
        # depth image
        if "distance_to_image_plane" in camera.data.output:
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
        
        #plt.imsave(depth_image, os.path.join(save_img_folder, folder_name, f'depth_{index}_{}.png')
        probability_grid = grid.log_odds_to_prob()
        hard_occ = torch.where(probability_grid[:, :, :, 0] >= 0.6, 1, 0)

        hard_occ = hard_occ[ :, :, 1:]
        num_match_occ = torch.sum(torch.logical_and(hard_occ, occ_gt[:, :, 1:]), dim=(0, 1, 2))
        total_occ = torch.sum(occ_gt[:, :, 1:], dim=(0, 1, 2))
        coverage_ratio = (num_match_occ/total_occ).reshape(-1, 1)
        coverage_ratio_avg[scene_id, index] = coverage_ratio.cpu().numpy()[0][0]
        
        print(coverage_ratio)
        rgb_image = camera.data.output["rgb"].clone()        
        obs = _get_observations(probability_grid, rgb_image, pose, obv_face)
        
        actions, _  = agent.predict(obs, deterministic=True)
        
        new_positions, orientation_camera, yaw, pitch = process_action(actions)
        print(depth_image.shape)
        plt.imsave(os.path.join(save_img_folder, folder_name, f'depth_{index}_{new_positions[0,0]}_{new_positions[0,1]}_{new_positions[0,2]}_{yaw.cpu().numpy()[0]}_{pitch.cpu().numpy()[0]}_{coverage_ratio.cpu().numpy()[0][0]}.png'),
                np.clip(depth_image[0,:,:,0].detach().cpu().numpy(),0,ENV_SIZE*2).astype(np.uint8),
                           cmap='gray',
                           vmin=0,
                           vmax=ENV_SIZE)
        plt.imsave(os.path.join(save_img_folder, folder_name, f'rgb_{index}_{new_positions[0,0]}_{new_positions[0,1]}_{new_positions[0,2]}_{yaw.cpu().numpy()[0]}_{pitch.cpu().numpy()[0]}_{coverage_ratio.cpu().numpy()[0][0]}.png'), rgb_image[0].detach().cpu().numpy().astype(np.uint8))
        # this may slow down the process
        org_x, org_y, org_z = env_size/2., env_size/2., 0
        cell_size = env_size/grid_size  # meters per cell
        slice_height = env_size / grid_size  # height of each slice in meters
        if args_cli.vis:
            for i in range(grid_size):
                occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
                create_blocks_from_occupancy(0, [0, 0, 0], occupancy_grid.cpu().numpy(), cell_size, i * cell_size, i, env_size, 1, 25)
        
        # apply action
        camera.set_world_poses(new_positions, orientation_camera)


    if scene_id+1==100:
        average_values = np.mean(coverage_ratio_avg[:scene_id+1], axis=0)
        print(average_values)
        # Plot the averaged values
        plt.plot(average_values, marker='o', color='b', label='Average Curve')
        plt.xlabel('Step')
        plt.ylabel('Average Value')
        plt.title('Average Line Curve of 20 Steps')
        plt.legend()
        plt.grid(True)
        plt.show()

    if args_cli.vis:
        for i in range(grid_size):
            occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
            save_occupancy_grid_as_image(occupancy_grid.cpu().numpy(), os.path.join(output, f"circular_occupancy_map_slice_{i}.png"))
    
        occupancy_grid = (probability_grid[0] > 0.5).int()
        np.save(os.path.join(output, "circular_occ.npy"), occupancy_grid.cpu().numpy())
        # screenshot
        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))
 
def process_action(actions):
    actions = torch.tensor(actions).to(DEVICE)
    _xyz = actions[:, :3]
    _xyz = (_xyz + torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)

    #if self.cfg.used_nearest:
    real_xyz = actions[:, 6:9]
    real_xyz = (real_xyz + torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)

    # 0~1
    lookatxyz = actions[:, 3:6]
    # -1~1
    lookatxyz = lookatxyz*2-1
    # to real-world xyz
    lookatxyz = (lookatxyz+torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)
    # compute yaw and pitch
    dxyz = lookatxyz - _xyz + 1e-6
    # calculate yaw using torch functions
    # -pi~pi
    _yaw = torch.atan2(dxyz[:, 1], dxyz[:, 0])
    # calculate pitch using torch functions
    # -pi/2~pi/2
    _pitch = torch.atan2(dxyz[:, 2], torch.sqrt(dxyz[:, 0]**2 + dxyz[:, 1]**2))
    # to positive: downward, negative: upward
    _pitch *= -1
    # normalize pitch as specified
    # -pi/3~pi/2 which is -60-90
    _pitch = torch.clamp(_pitch, min=-torch.pi/3, max=torch.pi/2)

    target_position = real_xyz
    pitch_radians = _pitch
    yaw = _yaw
    
    #print(pitch_radians, yaw)
    # roll, pitch, yaw
    target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(yaw.shape[0],1), pitch_radians.unsqueeze(1).cpu(), yaw.cpu().unsqueeze(1)],dim=1).numpy(), degrees=False)
    # setup camera position
    orientation_camera = convert_camera_frame_orientation_convention(torch.tensor(target_orientation).float(), origin="world", target="ros") 
    x_new = real_xyz[:, 0] + camera_offset[0] * torch.cos(_yaw) - camera_offset[1] * torch.sin(_yaw)
    y_new = real_xyz[:, 1] + camera_offset[0] * torch.sin(_yaw) + camera_offset[1] * torch.cos(_yaw)
    z_new = real_xyz[:, 2] + camera_offset[2]

    new_positions = torch.stack([x_new, y_new, z_new], dim=1)

    #self._camera.set_world_poses(new_positions, orientation_camera)

    return new_positions, orientation_camera, _yaw, _pitch

def compute_weighted_centroid(obv_occ, gt_occ):
    """
    Compute the weighted centroid of XYZ from gt_occ, ignoring faces already observed in obv_occ.

    Parameters:
        obv_occ (torch.Tensor): Observed occupancy grid of shape [n_envs, grid_x, grid_y, grid_z, 6].
        gt_occ (torch.Tensor): Ground truth occupancy grid of shape [n_envs, grid_x, grid_y, grid_z, 6].

    Returns:
        torch.Tensor: Weighted centroid of shape [n_envs, 3].
    """
    # Validate inputs
    assert obv_occ.shape == gt_occ.shape, "obv_occ and gt_occ must have the same shape"
    assert obv_occ.ndim == 5 and obv_occ.shape[-1] == 6, "Input dimensions must match [n_envs, grid_x, grid_y, grid_z, 6]"

    n_envs, grid_x, grid_y, grid_z, _ = obv_occ.shape

    # Mask gt_occ to ignore faces already observed in obv_occ
    valid_occ = gt_occ * (1 - obv_occ)

    # Compute the weights for each voxel (sum of valid faces)
    weights = torch.sum(valid_occ, dim=-1)

    # Create a grid of XYZ coordinates
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(grid_x, device=obv_occ.device),
        torch.arange(grid_y, device=obv_occ.device),
        torch.arange(grid_z, device=obv_occ.device),
        indexing="ij"
    )

    # Expand coordinates to match the shape of weights
    x_coords = x_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]
    y_coords = y_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]
    z_coords = z_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]

    # Compute weighted sums for each axis
    total_weights = torch.sum(weights, dim=(1, 2, 3), keepdim=True)
    total_weights = torch.clamp(total_weights, min=1e-6)  # Avoid division by zero

    weighted_x = torch.sum(weights * x_coords, dim=(1, 2, 3)) / total_weights.squeeze()
    weighted_y = torch.sum(weights * y_coords, dim=(1, 2, 3)) / total_weights.squeeze()
    weighted_z = torch.sum(weights * z_coords, dim=(1, 2, 3)) / total_weights.squeeze()

    # Combine weighted coordinates into centroids
    centroids = torch.stack((weighted_x, weighted_y, weighted_z), dim=-1)

    return centroids

def get_seen_face(occ_grid_xyz, camera_xyz, grid_size, device):
    #print(occ_grid_xyz.shape)
    #print(camera_xyz.shape)

    rays = camera_xyz - occ_grid_xyz

    # Normalize rays
    rays_norm = rays.to(device) / (torch.norm(rays, dim=-1, keepdim=True)+1e-10).to(device)

    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).to(device)
    face_grid = torch.zeros((grid_size, grid_size, grid_size, 6), dtype=torch.bool).to(device)

    # Check visibility for each face
    for i, face in enumerate(faces):
        dot_product = torch.sum(rays_norm * face, dim=-1)
        face_grid[occ_grid_xyz[:, 0], occ_grid_xyz[:, 1], occ_grid_xyz[:, 2], i] = dot_product > 0

    #print(face_grid)
    #print(face_grid.shape)
    #exit()
    return face_grid

def _get_observations(probability_grid, rgb_image, pose, obv_face):
        # pose: N, T, 7 (xyz + pitch yaw + height limit)
        # current pose
        current_pose = pose
        # reshape and normalize env_step
        #env_step_normalized = (self.env_step.to(self.device) / self.cfg.total_img).unsqueeze(1)
        # expand h_limit to match batch dimensions
        # num_envs = 1

        h_limit = (torch.ones(num_envs).to(DEVICE)*10*GRID_SIZE/ENV_SIZE).int()
        h_limit_expanded = h_limit.unsqueeze(1) / GRID_SIZE
        # concatenate current pose, normalized step, and h_limit
        #num_envs = 1
        #import pdb; pdb.set_trace()
        pose_step = torch.cat([current_pose.reshape(num_envs, -1), h_limit_expanded], dim=1)
        #pose_step = current_pose

        obv_occ = torch.ones(num_envs, GRID_SIZE, GRID_SIZE, GRID_SIZE, 4, device=DEVICE)*0.5
        # generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        y_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        z_coords = torch.linspace(0.0, ENV_SIZE, GRID_SIZE, device=DEVICE)
        
        # create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1) / ENV_SIZE
        obv_occ[:, :, :, :, 0] = probability_grid.clone()
        
        #obv_face = torch.zeros(num_envs, GRID_SIZE, GRID_SIZE, GRID_SIZE, 6, device=DEVICE)
        # occ: N, grid_size, grid_size, grid_size, occ + coordinate + face occ
        occ_face = torch.cat((obv_occ, obv_face), dim=-1)

        #print(rgb_image.shape)
        rgb_image = rgb_image/ 255.
        # img: N, H, W, 1
        gray_scale_img = torch.mean(rgb_image[0, :, :, :], (2))

        gt_faces = torch.zeros((num_envs, GRID_SIZE, GRID_SIZE, GRID_SIZE, 6), device=DEVICE)
        center = compute_weighted_centroid(occ_face[:, :, :, 1:, 4:], gt_faces[:, :, :, 1:, :])
        center /= occ_face.shape[1]

        obs = {"pose_step": pose_step,
               "img": gray_scale_img.reshape(-1, 1, CAMERA_HEIGHT, CAMERA_WIDTH),
               "occ": occ_face.permute(0, 4, 1, 2, 3),
               "env_size": torch.ones((num_envs, 1)) * ENV_SIZE,
               "aux_center": center
               }

        #observations = {"policy": obs}

        return obs

def make_env():
    """Play with stable-baselines agent."""
    # parse configuration

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    #agent = PPO.load(checkpoint_path, env, print_system_info=True)

    agent = PPO_Cus.load(checkpoint_path, print_system_info=True) 
    return agent


def main():
    # load non-metric usd
    #scenes_path = sorted(glob.glob(os.path.join(args_cli.input, "**", "*[!_non_metric].usd"), recursive=True))
    #print(scenes_path)
    #exit()

    # Define the input file
    input_file = args_cli.input
    
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
    
    coverage_ratio_avg = np.zeros((len(scene_paths), num_steps), dtype=np.float32)

    # world initialization
    world = World(stage_units_in_meters=1.0, backend="torch", device="cpu")
    # user viewport
    set_camera_view(eye=np.array([40, 40, 60]), target=np.array([-15, 15, 8]))
    agent = make_env()

    # start scan the shapes
    for i, (scene_path, hollow_occ_path, fill_occ_set_path, faces_path) in enumerate(scene_paths):
        print(f"{i}: {scene_path}")
        # setup ground, light, and camera
        scene_entities = setup_scene(world, scene_path)

        # output dir
        output_dir = os.path.dirname(scene_path)

        # run simulation (capture)
        world.reset()
        run_simulator(world, scene_entities, output_dir, agent, hollow_occ_path, coverage_ratio_avg, i)
        world.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
