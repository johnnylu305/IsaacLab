import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Make pretrain data")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import glob
import time

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg, spawn_from_multiple_usd_env_id, spawn_from_usd
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth, convert_orientation_convention



@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(0.75, 0.75, 0.75))
    )

    
    # robot
    drone = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, # won't drop
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            copy_from_source=False, # mirror
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint": 200.0,
                "m2_joint": -200.0,
                "m3_joint": 200.0,
                "m4_joint": -200.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=[0.5, 0.0, 0.0], convention="world"),
        update_period=0, # update every physical step
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.8,
            horizontal_aperture=24.,
            clipping_range=(0.2, 60.0)
        ),
        width=300,
        height=300
    )


def add_building(num_envs, i):
    scenes_path = []
    # Loop over each batch number
    for batch_num in range(1, 7):  # Range goes from 1 to 6 inclusive
        # Generate the path pattern for the glob function
        path_pattern = os.path.join(f'../Dataset/Raw_Rescale_USD/BATCH_{batch_num}', '**', '*[!_non_metric].usd')
        # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
        scenes_path.extend(sorted(glob.glob(path_pattern, recursive=True)))
        # only use one building
    scene_path = scenes_path[i]
    cfg = UsdFileCfg(usd_path=scene_path)
    spawn_from_usd("/World/envs/env_0/Scene", cfg)
    

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["drone"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    building_i = 0
    camera_offset = [0.5, 0., 0.]
    # Simulation loop
    while simulation_app.is_running():
        current_time = time.time()
        # Reset
        #if count % 500 == 0:
        #    count = 0
        #    building_i += 1
        #    scene.reset()
        #    delete_prim('/World/envs/env_0/Scene')
        #    add_building(scene.num_envs, building_i)
        #    print("[INFO]: Resetting robot state...")
        
        # Apply random action
        x = torch.distributions.Uniform(-9.5, 9.5).sample((scene.num_envs,))
        y = torch.distributions.Uniform(-9.5, 9.5).sample((scene.num_envs,))
        z = torch.distributions.Uniform(0, 10).sample((scene.num_envs,))
        _xyz = torch.stack((x, y, z), dim=1)
        # Sampling for yaw (-pi to pi)
        _yaw = torch.distributions.Uniform(-torch.pi, torch.pi).sample((scene.num_envs,))
        # Sampling for pitch (-pi/3 to pi/2)
        _pitch = torch.distributions.Uniform(-torch.pi/3, torch.pi/2).sample((scene.num_envs,))
        # drone itself only needs yaw, pitch and roll should be 0
        target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(_yaw.shape[0],1), torch.zeros(_yaw.shape[0],1), _yaw.unsqueeze(1)],dim=1).numpy(), degrees=False)
        target_orientation = torch.from_numpy(target_orientation)
        # -- apply action to the robot
        robot_pos = _xyz + scene.env_origins
        robot_ori = torch.zeros((scene.num_envs, 3))
        robot_ori[:, 0] = _pitch
        robot_ori[:, 1] = _yaw
        # setup robot position
        root_state = torch.zeros((scene.num_envs, 13))
        root_state[:, :3] = robot_pos
        root_state[:,3:7] = target_orientation
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        # setup camera position
        drone_euler = torch.cat([torch.zeros(_yaw.shape[0],1), _pitch.unsqueeze(1), _yaw.unsqueeze(1)], dim=1)
        pitch_quat = torch.from_numpy(rot_utils.euler_angles_to_quats(drone_euler, degrees=False)).float()
        orientation_camera = convert_orientation_convention(pitch_quat, origin="world", target="ros")

        x_new = root_state[:, 0] + camera_offset[0] * torch.cos(_yaw) - camera_offset[1] * torch.sin(_yaw)
        y_new = root_state[:, 1] + camera_offset[0] * torch.sin(_yaw) + camera_offset[1] * torch.cos(_yaw)
        z_new = root_state[:, 2] + camera_offset[2]
 
        new_positions = torch.stack([x_new, y_new, z_new], dim=1)

        scene["camera"].set_world_poses(new_positions, orientation_camera)


        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)
        if count%50==0:
            print(f"fps: {scene.num_envs*1.0/(time.time()-current_time)}")

def main():
    """Main function."""
    # Load kit helper
    # TODO: enable gpu
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=60)
    scene = InteractiveScene(scene_cfg)
    add_building(args_cli.num_envs, 0)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
