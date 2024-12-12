import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg

CRAZYFLIE_CFG = ArticulationCfg(
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


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    # env
    env_size = 20
    grid_size = 20
    preplan = False
    save_img = True
    vis_occ = False
    vis_pointcloud = False
    save_env_ids = [0, 1]
    save_img_freq = 30 #10
    random_initial = True #False
    test = True
    duster = False

    num_envs = 2 # this might be overwrote by parser
    env_spacing = 60 #30 # in meter, 2 cells is one unit

    decimation = 2 #5 # _apply_action will run decimation time
    #decimation = 10 # _apply_action will run decimation time
    num_actions = 5 # x, y, z, yaw, pitch
    num_states = 0
    episode_length_s = 20000 # timeout
    debug_vis = False #True

    # obv
    img_t = 2 
    total_img = 50 #50
    goal = 0.9 #0.92

    # occ grid
    # TODO need to tune free threshold
    decrement = 0.01 #0.2
    increment = 1.0
    max_log_odds = 10.
    min_log_odds = -10.

    # sensor
    # TODO FIX OFFSET
    camera_offset = [0.5, 0.0, 0.0]
    camera_w, camera_h = 300, 300 #300, 300 # try 400 x 400 in the future
    #camera_w, camera_h = 1000, 1000 # try 400 x 400 in the future

    # obv
    num_observations = total_img * 5
    img_observations = [img_t, camera_h, camera_w, 3]
    pose_observations = [total_img, 5] # N, xyz + yaw + pitch
    occ_observations = [grid_size, grid_size, grid_size, 4] # label + xyz

    # reward scales
    occ_reward_scale = 30.0
    col_reward_scale = -1 #-30.0 #-10.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/10, #50, # physical simulation step
        disable_contact_processing=True,
        #physx=sim_utils.PhysxCfg(), # with use_gpu, the buffer cannot grow dynamically
        #physics_material=sim_utils.RigidBodyMaterialCfg(
        #    friction_combine_mode="multiply",
        #    restitution_combine_mode="multiply",
        #    static_friction=1.0,
        #    dynamic_friction=1.0,
        #    restitution=0.0,
        #),
        gravity=(0.0, 0.0, 0.0), # disable gravity
        enable_scene_query_support=False, # disable collision query
        render_interval=decimation
    )

    # ground	
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        #collision_group=-1,
        #physics_material=sim_utils.RigidBodyMaterialCfg(
        #    friction_combine_mode="multiply",
        #    restitution_combine_mode="multiply",
        #    static_friction=1.0,
        #    dynamic_friction=1.0,
        #    restitution=0.0,
        #)
    )
    
    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG

    # sensor    
    #camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Camera",
    #     offset=CameraCfg.OffsetCfg(pos=camera_offset, convention="world"),
    #     update_period=0, # update every physical step
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=13.8, # in cm
             #focus_distance=1.0, # in m 
   #          horizontal_aperture=24., # in mm 
             #clipping_range=(0.1, 20.0) # near and far plane in meter
   #          clipping_range=(0.5, 60.0) # near and far plane in meter
   #      ),
   #      width=camera_w,
   #      height=camera_h,
   #)
    
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_offset, convention="world"),
        update_period=0, # update every physical step
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.8, # in cm default 24, dji 1.38
            #focus_distance=1.0, # in m 
            horizontal_aperture=24., # in mm 
            clipping_range=(0.2, 60.0) # near and far plane in meter
        ),
        width=camera_w,
        height=camera_h,
      
    )

    if duster:
        camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_offset, convention="world"),
        update_period=0, # update every physical step
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.8, # in cm default 24, dji 1.38
            #focus_distance=1.0, # in m 
            horizontal_aperture=24., # in mm 
            clipping_range=(0.2, 60.0) # near and far plane in meter
        ),
        width=512,
        height=384,
      
        )

    # setup interactive scene for rl training
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, 
                                                     env_spacing=env_spacing,
                                                     lazy_sensor_update=False, # update sensor data every time
                                                     replicate_physics=True)
